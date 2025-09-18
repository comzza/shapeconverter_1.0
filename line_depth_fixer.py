from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import os
import math
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from itertools import groupby
from operator import itemgetter

from shapely.strtree import STRtree
from pyproj import CRS
import trimesh
import csv

###############################
# 결과 Object
###############################
@dataclass
class FixedPipeFeature:
    x: float
    y: float
    z: float
    diam: float
    fid: int
    feature_id: str # 관로번호
    flag: bool # False: 시작/끝/내분점, True: 심도점
    point_type: str = "" # 시작점/끝점/내분점/심도점 구분
    facility_type: Optional[str] = None # (NEW) 관로 종류

###############################
# 내부 보조 유틸
###############################

# EPSG:5186 좌표계로 변환 유틸
def _ensure_epsg5186(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # gdf.crs: None | CRS | dict
    if gdf.crs is None:
        return gdf
    target = CRS.from_epsg(5186)
    if gdf.crs.to_epsg() != 5186:
        return gdf.to_crs(target)
    return gdf

# 관로 feature의 parts에 심도점(points) 삽입 유틸
def _line_insert_points(line: LineString, insert_pts: List[Point]) -> List[Point]:
    if not insert_pts:
        return [Point(c) for c in line.coords]

    def proj(p: Point) -> float:
        return line.project(p)

    eps = 1e-8
    projected = [(max(0.0, min(line.length, proj(p))), p) for p in insert_pts]
    projected.sort(key=lambda t: t[0])

    base_pts = [Point(c) for c in line.coords]
    base_proj = [(line.project(p), p) for p in base_pts]

    merged = base_proj + projected
    merged.sort(key=lambda t: t[0])
    out: List[Point] = []
    for _, p in merged:
        if not out or out[-1].distance(p) > eps:
            out.append(p)
    return out

# 선형 보간 함수
def _linear_interpolate_z(x0, z0, x1, z1, x):
    if abs(x1 - x0) < 1e-9:
        return z0
    t = (x - x0) / (x1 - x0)
    return z0 * (1 - t) + z1 * t


# === [NEW] Shapely 1.x / 2.x 호환용 STRtree query 정규화 유틸 ===
def _strtree_query_geoms(tree: STRtree, geoms: List, query_geom) -> List:
    """
    Shapely 1.x: list of geometries 반환
    Shapely 2.x: ndarray of integer indices 반환
    → 항상 '지오메트리 리스트'로 변환해 돌려준다.
    """
    res = tree.query(query_geom)
    if res is None:
        return []
    try:
        arr = np.asarray(res)
        # 정수형 인덱스 배열이면, 입력 geoms에서 지오메트리로 매핑
        if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.integer):
            return [geoms[int(i)] for i in arr.tolist()]
    except Exception:
        pass
    # 이미 지오메트리(Shapely 1.x 스타일)인 경우
    return list(res)

###############################
# 핵심 유틸리티 클래스
###############################
class LineDepthFixer:
    def __init__(self,
                 facility_type: str,
                 line_shp_file_path: str,
                 depth_shp_file_path: str,
                 save_path: str,
                 avg_depth_field: str,
                 pipe_size_field: str,
                 point_depth_field: str,
                 feature_id_field: str,
                 snap_tol_m: float = 0.3,
                 connect_branch_mode: str = "inherit",   # 'inherit' | 'average'
                 interp_mode: str = "linear"             # 'linear' | 'idw'
                ):
        self.facility_type = facility_type
        self.line_shp_file_path = line_shp_file_path
        self.depth_shp_file_path = depth_shp_file_path
        self.save_path = save_path
        self.avg_depth_field = avg_depth_field
        self.pipe_size_field = pipe_size_field
        self.point_depth_field = point_depth_field
        self.feature_id_field = feature_id_field
        self.snap_tol_m = float(snap_tol_m)
        self.connect_branch_mode = connect_branch_mode
        self.interp_mode = interp_mode

        self.result_points: List[Dict] = []
        self.result_objects: List[FixedPipeFeature] = []

    def shape_union_run(self) -> None:
        """
        Perform shape union operation.

        이 메서드는 Line Shapefile와 Depth Shapefile를 읽은 후, 합집합 연산 수행
        합집합 결과는 `self.result_points`에 저장
        """
        # --- 1. 입력 데이터 준비 ---
        # 입력 shapefile 읽기
        lines = gpd.read_file(self.line_shp_file_path)
        depth_points = gpd.read_file(self.depth_shp_file_path)

        # 좌표계를 EPSG:5186 (중부원점 UTM-K)으로 통일
        lines = _ensure_epsg5186(lines)
        depth_points = _ensure_epsg5186(depth_points)

        # MultiLineString 타입을 LineString으로 분해하여 각 라인을 개별적으로 처리
        if hasattr(lines, "explode"):
            lines = lines.explode(index_parts=False, ignore_index=True)
        
        # 유효하지 않은 지오메트리(비어있거나, LineString이 아닌 경우) 제거
        lines = lines[
            ~lines.geometry.is_empty & (lines.geometry.geom_type == "LineString")
        ].reset_index(drop=True)

        # --- 2. 필수 필드 유효성 검사 ---
        # 연산에 필요한 필드들이 각 shapefile에 존재하는지 확인
        required_line_fields = [self.avg_depth_field, self.pipe_size_field, self.feature_id_field]
        for f in required_line_fields:
            if f not in lines.columns:
                raise KeyError(f"Line Shapefile에 필수 필드가 없습니다: {f}")
        if self.point_depth_field not in depth_points.columns:
            raise KeyError(f"Depth-Point Shapefile에 필수 필드가 없습니다: {self.point_depth_field}")

        # --- 3. 심도점을 가장 가까운 관로에 스냅 ---
        # 각 심도점을 어느 관로에 스냅할지 결정하는 단계
        line_geoms = list(lines.geometry)
        
        # 효율적인 공간 검색을 위해 STRtree(Sort-Tile-Recursive tree) 인덱스를 생성
        strtree = STRtree(line_geoms)
        
        # 각 관로(line_index)에 스냅된 심도점(snapped_point, point_index) 목록을 저장할 딕셔너리
        line_to_snapped: Dict[int, List[Tuple[Point, int]]] = {i: [] for i in range(len(lines))}
        
        # 어떤 심도점이 사용되었는지 추적하기 위한 플래그
        point_flags: List[bool] = [False] * len(depth_points)
        
        # [개선] list.index()의 성능 문제를 해결하기 위해 지오메트리와 인덱스를 미리 매핑
        line_geom_to_idx = {geom.wkb: i for i, geom in enumerate(line_geoms)}
        
        # 각 심도점에 대해 가장 가까운 관로를 찾아 스냅
        
        # 모든 심도점을 순회하며 가장 가까운 관로 탐색
        for pi, p in enumerate(depth_points.geometry):
            # snap_tol_m 버퍼 내에 있는 후보 관로들을 빠르게 검색
            candidates = _strtree_query_geoms(strtree, line_geoms, p.buffer(self.snap_tol_m))
            
            best_idx = None
            best_dist = float('inf')
            # 후보 관로들 중 실제 거리가 가장 가까운 관로를 탐색
            for cand in candidates:
                d = cand.distance(p)
                if d < best_dist:
                    best_dist = d
                    # [개선] 미리 만들어둔 딕셔너리를 이용해 인덱스를 빠르게 조회
                    best_idx = line_geom_to_idx.get(cand.wkb)

            # 가장 가까운 관로가 허용 오차(snap_tol_m) 내에 있을 경우에만 스냅핑 수행
            if best_idx is None or best_dist > self.snap_tol_m:
                continue
            
            # 스냅핑 계산: 점을 선에 투영(project)하여 선 위의 가장 가까운 지점을 찾음(interpolate)
            line = line_geoms[best_idx]
            m = line.project(p)       # 선의 시작점부터의 거리
            snapped = line.interpolate(m) # 거리 m에 해당하는 좌표
            
            # 스냅된 결과를 딕셔너리에 저장
            line_to_snapped[best_idx].append((snapped, pi))
            point_flags[pi] = True

        # --- 4. 각 관로의 Z값 계산 및 점 생성 ---
        # 모든 피처(관로)에 대해 최종 점(정점)들을 저장할 리스트
        all_feature_points: List[Tuple[int, List[Point], List[bool]]] = []

        # 모든 관로를 순회
        for li, row in lines.iterrows():
            line: LineString = row.geometry
            
            # 관로의 속성 정보 추출 (결측치 처리 포함)
            feature_id_raw = row[self.feature_id_field]
            feature_id_str = None if pd.isna(feature_id_raw) else str(feature_id_raw)
            fid = int(feature_id_raw) if not pd.isna(feature_id_raw) and str(feature_id_raw).isdigit() else li
            diam_mm = float(row[self.pipe_size_field]) if not pd.isna(row[self.pipe_size_field]) else 0.0
            avg_depth = float(row[self.avg_depth_field]) if not pd.isna(row[self.avg_depth_field]) else 0.0
            
            # 관 상단(z값) 계산을 위해 관경의 절반을 미터 단위로 변환
            diam_m_half = (diam_mm / 2.0) / 1000.0

            # 현재 관로에 스냅된 점들의 목록을 가져옴
            snapped_list = line_to_snapped.get(li, [])
            snapped_pts = [sp for sp, _ in snapped_list]
            
            # 기존 관로의 정점과 스냅된 점들을 합쳐 새로운 정점 리스트 생성
            part_pts = _line_insert_points(line, snapped_pts)
            # 각 정점들이 관로 시작점으로부터 얼마나 떨어져 있는지(거리) 계산
            mvals = [line.project(pt) for pt in part_pts]

            # --- 4.1. Z값 보간을 위한 기준(앵커) Z값 설정 ---
            # 스냅된 점의 개수에 따라 관로의 시작점/끝점 Z값을 결정
            if len(snapped_pts) == 0: # 스냅된 점이 없으면 관로의 평균심도 사용
                z_start = avg_depth + diam_m_half
                z_end = avg_depth + diam_m_half
            elif len(snapped_pts) == 1: # 스냅된 점이 하나면 해당 점의 심도를 전체 관로에 적용
                depth_val = float(depth_points.iloc[snapped_list[0][1]][self.point_depth_field])
                z_start = depth_val + diam_m_half
                z_end = depth_val + diam_m_half
            else: # 스냅된 점이 둘 이상이면, 관로상의 첫번째/마지막 스냅점의 심도를 양 끝의 심도로 사용
                snapped_sorted = sorted(snapped_list, key=lambda t: line.project(t[0]))
                first_depth = float(depth_points.iloc[snapped_sorted[0][1]][self.point_depth_field])
                last_depth = float(depth_points.iloc[snapped_sorted[-1][1]][self.point_depth_field])
                z_start = first_depth + diam_m_half
                z_end = last_depth + diam_m_half

            # Z값 보간에 사용할 앵커(기준점) 리스트 생성: (거리, Z값)
            anchors: List[Tuple[float, float]] = []
            anchors.append((0.0, z_start)) # 관로 시작점
            anchors.append((line.length, z_end)) # 관로 끝점

            snapped_m_set = set() # 스냅된 지점인지 확인하기 위한 거리값 집합
            for snapped, pidx in snapped_list:
                m = line.project(snapped)
                snapped_m_set.add(round(m, 6)) # 부동소수점 오차를 고려해 반올림
                depth_val = float(depth_points.iloc[pidx][self.point_depth_field])
                anchors.append((m, depth_val + diam_m_half)) # 스냅된 심도점들도 앵커에 추가
            
            # 앵커들을 거리순으로 정렬
            anchors.sort(key=lambda x: x[0])

            # --- 4.2. Z값 보간 함수 정의 ---
            def interp_z(mq: float) -> float:
                """ 앵커들을 기준으로 주어진 거리(mq)에 해당하는 Z값을 보간하여 반환 """
                # mq를 감싸는 좌우 앵커 탐색
                left = anchors[0]
                right = anchors[-1]
                for i in range(len(anchors) - 1):
                    a0, a1 = anchors[i], anchors[i + 1]
                    if a0[0] <= mq <= a1[0]:
                        left, right = a0, a1
                        break

                # 보간 모드에 따라 Z값 계산
                if self.interp_mode == "idw": # 역거리 가중 보간
                    d0 = max(1e-9, abs(mq - left[0])) # 0으로 나누는 것을 방지
                    d1 = max(1e-9, abs(right[0] - mq))
                    w0 = 1.0 / d0
                    w1 = 1.0 / d1
                    return (left[1] * w0 + right[1] * w1) / (w0 + w1)
                else: # 기본: 선형 보간
                    return _linear_interpolate_z(left[0], left[1], right[0], right[1], mq)

            # --- 4.3. 최종 결과점 리스트에 추가 ---
            flags = []
            for m in mvals:
                # 각 정점이 스냅된 심도점에 의해 생성된 것인지 여부를 플래그로 저장
                is_depth_split = (round(m, 6) in snapped_m_set)
                flags.append(is_depth_split)
            
            # (피처ID, 정점목록, 플래그목록) 형태로 저장. 추후 분기점 연결에 사용
            all_feature_points.append((fid, part_pts, flags))
            
            # 각 정점의 Z값을 계산하여 최종 결과 리스트(`self.result_points`)에 추가
            for m, pt, flg in zip(mvals, part_pts, flags):
                z_val = interp_z(m)
                self.result_points.append({
                    'x': float(pt.x), 'y': float(pt.y), 'z': float(z_val),
                    'diam': float(diam_mm), 'fid': int(fid), 'feature_id': feature_id_str, 'flag': bool(flg)
                })

        # --- 5. 분기 관로 연결 처리 ---
        # 관로들이 만나는 지점의 X, Y, Z 좌표를 일치시키는 후처리 단계
        
        # 모든 점들을 끝점(endpoint)과 내부점(interior)으로 분류
        endpoint_records: List[Tuple[int, int, Point, float]] = [] # (전체 인덱스, fid, 지오메트리, z값)
        interior_records: List[Tuple[int, int, Point, float]] = []

        idx_cursor = 0
        for fid, pts, flgs in all_feature_points:
            n = len(pts)
            for j in range(n):
                rp = self.result_points[idx_cursor + j]
                p_geom = Point(rp['x'], rp['y'])
                if j == 0 or j == n - 1: # 라인의 양 끝점
                    endpoint_records.append((idx_cursor + j, fid, p_geom, rp['z']))
                else: # 라인의 중간점
                    interior_records.append((idx_cursor + j, fid, p_geom, rp['z']))
            idx_cursor += n

        # 끝점과 내부점에 대해 각각 STRtree를 생성하여 빠른 검색 준비
        endpoint_geoms = [rec[2] for rec in endpoint_records]
        endpoint_tree = STRtree(endpoint_geoms)
        interior_geoms = [rec[2] for rec in interior_records] if interior_records else []
        interior_tree = STRtree(interior_geoms) if interior_records else None
        
        # [개선] 성능 향상을 위해 지오메트리-인덱스 맵을 미리 생성
        endpoint_geom_to_record_idx = {geom.wkb: i for i, geom in enumerate(endpoint_geoms)}
        interior_geom_to_record_idx = {geom.wkb: i for i, geom in enumerate(interior_geoms)}

        # 모든 끝점을 순회하며 주변의 다른 관로와 연결되는지 확인
        for ei, (gidx, fid, geom, zval) in enumerate(endpoint_records):
            
            # --- 5.1. 끝점(Endpoint) -> 내부점(Interior) 연결 ---
            # 다른 관로의 중간 부분에 현재 끝점이 연결되는 경우
            if interior_tree is not None:
                candidates = _strtree_query_geoms(interior_tree, interior_geoms, geom.buffer(self.snap_tol_m))
                target_cand = min(candidates, key=lambda c: c.distance(geom)) if candidates else None

                if target_cand and target_cand.distance(geom) <= self.snap_tol_m:
                    # [개선] 딕셔너리를 사용하여 인덱스를 즉시 조회
                    ti = interior_geom_to_record_idx.get(target_cand.wkb)
                    tgidx, tfid, tgeom, tz = interior_records[ti]
                    
                    # 연결 모드에 따라 좌표 조정
                    if self.connect_branch_mode == "average": # 두 점의 평균값으로 일치
                        avgx = (geom.x + tgeom.x) / 2.0
                        avgy = (geom.y + tgeom.y) / 2.0
                        avgz = (zval + tz) / 2.0
                        self.result_points[gidx]['x'], self.result_points[gidx]['y'], self.result_points[gidx]['z'] = avgx, avgy, avgz
                    else: # 'inherit': 끝점의 좌표를 내부점의 좌표로 변경
                        self.result_points[gidx]['x'], self.result_points[gidx]['y'], self.result_points[gidx]['z'] = tgeom.x, tgeom.y, tz
                    continue # 연결 처리가 완료되었으므로 다음 끝점으로 넘어감

            # --- 5.2. 끝점(Endpoint) -> 끝점(Endpoint) 연결 ---
            # 다른 관로의 끝점과 현재 끝점이 만나는 경우
            candidates = _strtree_query_geoms(endpoint_tree, endpoint_geoms, geom.buffer(self.snap_tol_m))
            
            # 자기 자신을 제외하고 가장 가까운 후보 탐색
            bestd = float('inf')
            target_cand = None
            for cand in candidates:
                d = cand.distance(geom)
                if d < 1e-12: continue # 거의 같은 점은 자기 자신으로 간주
                if d < bestd:
                    bestd = d
                    target_cand = cand

            if target_cand and bestd <= self.snap_tol_m:
                # [개선] 딕셔너리를 사용하여 인덱스를 즉시 조회
                ti = endpoint_geom_to_record_idx.get(target_cand.wkb)
                tgidx, tfid, tgeom, tz = endpoint_records[ti]
                
                # 두 끝점의 좌표를 평균값으로 통일
                avgx = (geom.x + tgeom.x) / 2.0
                avgy = (geom.y + tgeom.y) / 2.0
                avgz = (zval + tz) / 2.0
                self.result_points[gidx]['x'], self.result_points[gidx]['y'], self.result_points[gidx]['z'] = avgx, avgy, avgz
                self.result_points[tgidx]['x'], self.result_points[tgidx]['y'], self.result_points[tgidx]['z'] = avgx, avgy, avgz

        # --- 6. 최종 결과 객체 생성 ---
        # 딕셔너리 리스트를 최종 데이터 클래스 객체 리스트로 변환
        self.result_objects = [
            FixedPipeFeature(
                x=rp['x'], y=rp['y'], z=rp['z'], diam=rp['diam'],
                fid=rp['fid'], feature_id=rp.get('feature_id'), flag=rp['flag']
            )
            for rp in self.result_points
        ]

    def get_list_result(self) -> List[Dict]:
        return self.result_points

    def get_object_result(self) -> List[FixedPipeFeature]:
        return self.result_objects

    #############################################
    # 결과 내보내기 추가 기능
    #############################################

    def configure_output_file(self, export_filetype='gpkg'):
        # export 폴더 경로 생성
        export_dir = os.path.join(self.save_path, 'export')
        
        # 폴더 존재하지 않으면 생성
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)
        
        # 파일명에서 확장자 분리
        base_name_with_ext = os.path.basename(self.line_shp_file_path)
        base_name, _ = os.path.splitext(base_name_with_ext)
        base_name = f"{self.facility_type}_{base_name}"
        # 출력 파일명 구성
        out_file = f"{base_name}.{export_filetype}"
        
        return os.path.join(export_dir, out_file)



    def export_to_shapefile(self, out_path: str) -> None:
        """
        결과 데이터를 3D 포인트 Shapefile로 내보냅니다.
        주의: Shapefile 형식의 한계로 지오메트리의 Z값은 저장되지 않고, 'z' 필드에만 값이 저장됩니다.
        """
        try:
            print(f"Shapefile 내보내기 시작: {out_path}")
            # GeoDataFrame 생성
            gdf = gpd.GeoDataFrame([
                {
                    'fid': rp['fid'],
                    'facility_type': self.facility_type,
                    'feature_id': rp.get('feature_id'),
                    'diam': rp['diam'],
                    'flag': rp['flag'],
                    'z': rp['z'],
                    'geometry': Point(rp['x'], rp['y'], rp['z']) # 3D 포인트 생성
                } for rp in self.result_points
            ], crs="EPSG:5186")
            
            # 파일로 저장
            gdf.to_file(out_path, driver='ESRI Shapefile', encoding='utf-8')
            print(f"✔️ Shapefile 내보내기 완료: {out_path}")

        except Exception as e:
            # 파일 쓰기 중 오류 발생 시 처리
            print(f"❌ Shapefile 내보내기 실패: {out_path}")
            print(f"  오류 원인: {e}")

    def export_points_to_geopackage(self, out_path: str, layer_name: str = "pipe_points") -> None:
        """
        [기존 함수] 결과 데이터를 3D 포인트 GeoPackage로 내보냅니다.
        """
        try:
            print(f"Point GeoPackage 내보내기 시작: {out_path}")
            gdf = gpd.GeoDataFrame([
                {
                    'fid': rp['fid'],
                    'facility_type': self.facility_type,
                    'feature_id': rp.get('feature_id'),
                    'diam': rp['diam'],
                    'flag': rp['flag'],
                    'z': rp['z'],
                    'geometry': Point(rp['x'], rp['y'], rp['z'])
                } for rp in self.result_points
            ], crs="EPSG:5186")
            
            gdf.to_file(out_path, layer=layer_name, driver='GPKG')
            print(f"✔️ Point GeoPackage 내보내기 완료: {out_path}")

        except Exception as e:
            print(f"❌ Point GeoPackage 내보내기 실패: {out_path}")
            print(f"  오류 원인: {e}")

    def export_lines_to_geopackage(self, out_path: str, layer_name: str = "pipe_lines") -> None:
        """
        [추가된 함수] 결과 포인트들을 fid 기준으로 그룹화하여 3D LineString GeoPackage로 내보냅니다.
        """
        try:
            print(f"LineString GeoPackage 내보내기 시작: {out_path}")

            # 1. 'fid'를 기준으로 데이터를 정렬 (groupby를 위해 필수)
            sorted_points = sorted(self.result_points, key=itemgetter('fid'))
            
            line_features = []
            # 2. 'fid'가 동일한 포인트들을 하나의 그룹으로 묶기
            for fid, group in groupby(sorted_points, key=itemgetter('fid')):
                point_group = list(group)

                # 3. 라인을 생성하려면 최소 2개 이상의 포인트가 필요
                if len(point_group) < 2:
                    continue

                # 4. 그룹 내 포인트들의 (x, y, z) 좌표 리스트를 생성
                coords = [(pt['x'], pt['y'], pt['z']) for pt in point_group]
                
                # 5. 좌표 리스트로 3D LineString 지오메트리 생성
                line_geom = LineString(coords)
                
                # 라인의 속성 정보는 그룹의 첫 번째 포인트에서 가져옴
                first_point = point_group[0]
                line_features.append({
                    'fid': fid,
                    'facility_type': self.facility_type,
                    'feature_id': first_point.get('feature_id'),
                    'diam': first_point['diam'],
                    'geometry': line_geom
                })

            if not line_features:
                print("⚠️ 경고: 생성된 라인이 없어 파일을 저장하지 않습니다.")
                return

            # 6. 생성된 라인 피처들로 GeoDataFrame 생성
            gdf = gpd.GeoDataFrame(line_features, crs="EPSG:5186")
            
            gdf.to_file(out_path, layer=layer_name, driver='GPKG')
            print(f"✔️ LineString GeoPackage 내보내기 완료: {out_path}")

        except Exception as e:
            print(f"❌ LineString GeoPackage 내보내기 실패: {out_path}")
            print(f"  오류 원인: {e}")

    def export_to_csv(self, out_path: str) -> None:
        """
        결과 데이터를 CSV 파일로 내보냅니다.
        """
        try:
            print(f"CSV 내보내기 시작: {out_path}")
            # pandas DataFrame 생성
            df = pd.DataFrame(self.result_points)
            
            # 파일로 저장 (utf-8-sig는 Excel에서 한글 깨짐 방지)
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"✔️ CSV 내보내기 완료: {out_path}")

        except Exception as e:
            # 파일 쓰기 중 오류 발생 시 처리
            print(f"❌ CSV 내보내기 실패: {out_path}")
            print(f"  오류 원인: {e}")


#############################################
# 외부 호출용 래퍼 함수
#############################################

def shape_union_run(line_shp_file_path: str,
                     depth_shp_file_path: str,
                     save_path: str,
                     avg_depth_field: str,
                     pipe_size_field: str,
                     point_depth_field: str,
                     feature_id_field: str,
                     facility_type: str,
                     snap_tol_m: float = 0.3) -> LineDepthFixer:
    fixer = LineDepthFixer(
        line_shp_file_path=line_shp_file_path,
        depth_shp_file_path=depth_shp_file_path,
        save_path=save_path,
        avg_depth_field=avg_depth_field,
        pipe_size_field=pipe_size_field,
        point_depth_field=point_depth_field,
        feature_id_field=feature_id_field,
        facility_type=facility_type,
        snap_tol_m=snap_tol_m,
    )
    fixer.shape_union_run()
    return fixer

def get_list_result(fixer: LineDepthFixer) -> List[Dict]:
    return fixer.get_list_result()

def get_object_result(fixer: LineDepthFixer) -> List[FixedPipeFeature]:
    return fixer.get_object_result()

###############################
# 충돌 분석 유틸리티
###############################
class CollisionAnalyzer:
    def __init__(self,
                 fixed_features: List[FixedPipeFeature],
                 *,
                 overlap_mode: str = "interval",   # 'center' | 'interval'
                 export_format: str = "both",      # 'obj' | 'glb' | 'both'
                 cylinder_length: float = 1.0,
                 slices: int = 24,
                 z_up_is_positive: bool = False):
        """
        overlap_mode:
          - 'center'   : |zA - zB| < rA + rB 기준
          - 'interval' : [zA-rA, zA+rA]와 [zB-rB, zB+rB] 구간 겹침 기준(기본)
        export_format: OBJ/GLB 선택
        cylinder_length: 생성 원통 길이(m)
        slices: 원통 분해수
        z_up_is_positive: 모델 z축 양(+)을 위로 볼지 여부
        """
        self.fixed_features = fixed_features
        self.overlap_mode = overlap_mode
        self.export_format = export_format
        self.cylinder_length = float(cylinder_length)
        self.slices = int(slices)
        self.z_up_is_positive = bool(z_up_is_positive)

        self._fid_map: Dict[int, List[FixedPipeFeature]] = {}
        self._lines: Dict[int, LineString] = {}

    # ----------------------- 내부 빌드 -----------------------
    def _build_lines(self) -> None:
        fid_map: Dict[int, List[FixedPipeFeature]] = {}
        for f in self.fixed_features:
            fid_map.setdefault(f.fid, []).append(f)
        # 정렬: 기존 입력 순서를 신뢰하되, 동일 좌표 중복을 제거
        for fid, pts in fid_map.items():
            dedup = []
            for p in pts:
                if not dedup or (dedup[-1].x != p.x or dedup[-1].y != p.y):
                    dedup.append(p)
            self._fid_map[fid] = dedup
            if len(dedup) >= 2:
                self._lines[fid] = LineString([(p.x, p.y) for p in dedup])

    @staticmethod
    def _interp_z_on_polyline(poly: List[FixedPipeFeature], inter: Point) -> float:
        """poly 상 교차점(inter)에서 선형보간 z 계산."""
        if len(poly) == 0:
            return 0.0
        if len(poly) == 1:
            return poly[0].z
        # 가장 가까운 세그먼트 찾기
        best = (None, None, float('inf'))
        for i in range(len(poly) - 1):
            p0, p1 = poly[i], poly[i+1]
            seg = LineString([(p0.x, p0.y), (p1.x, p1.y)])
            d = seg.distance(inter)
            if d < best[2]:
                best = (i, i+1, d)
        i0, i1, _ = best
        if i0 is None:
            return poly[0].z
        p0, p1 = poly[i0], poly[i1]
        seg = LineString([(p0.x, p0.y), (p1.x, p1.y)])
        m = seg.project(inter)
        if seg.length < 1e-9:
            return p0.z
        t = m / seg.length
        return p0.z * (1 - t) + p1.z * t

    @staticmethod
    def _radius_m(diam_mm: float) -> float:
        return float(diam_mm) / 2000.0

    def _overlap(self, zA: float, rA: float, zB: float, rB: float) -> Tuple[bool, float]:
        """충돌 여부와 간섭량(>0 이면 겹침 깊이)을 반환."""
        if self.overlap_mode == "center":
            margin = (rA + rB) - abs(zA - zB)
            return (margin > 0), margin
        # interval 모드: 두 구간이 겹치는 길이로 간섭량 계산
        a0, a1 = zA - rA, zA + rA
        b0, b1 = zB - rB, zB + rB
        overlap_len = min(a1, b1) - max(a0, b0)
        return (overlap_len > 0), overlap_len

    # ----------------------- 공개 API -----------------------
    def detect_collisions(self) -> List[Dict]:
        """교차점/충돌 목록을 반환. 각 항목: dict(meta 포함)."""
        self._build_lines()
        fids = list(self._fid_map.keys())
        results: List[Dict] = []
        for i in range(len(fids)):
            fid_i = fids[i]
            line_i = self._lines.get(fid_i)
            if line_i is None:
                continue
            for j in range(i+1, len(fids)):
                fid_j = fids[j]
                line_j = self._lines.get(fid_j)
                if line_j is None:
                    continue
                inter = line_i.intersection(line_j)
                pts: List[Point] = []
                if inter.is_empty:
                    continue
                if inter.geom_type == 'Point':
                    pts = [inter]
                elif inter.geom_type == 'MultiPoint':
                    pts = list(inter.geoms)
                else:
                    # 공유 구간은 스킵
                    continue
                for ip in pts:
                    z_i = self._interp_z_on_polyline(self._fid_map[fid_i], ip)
                    z_j = self._interp_z_on_polyline(self._fid_map[fid_j], ip)
                    # 반지름: 교차점 주변 평균 직경 사용
                    diam_i = np.mean([p.diam for p in self._fid_map[fid_i]]) if len(self._fid_map[fid_i]) else 0.0
                    diam_j = np.mean([p.diam for p in self._fid_map[fid_j]]) if len(self._fid_map[fid_j]) else 0.0
                    r_i = self._radius_m(diam_i)
                    r_j = self._radius_m(diam_j)

                    hit, overlap_amt = self._overlap(z_i, r_i, z_j, r_j)
                    if not hit:
                        continue

                    # 메타 추출 (facility_type/feature_id)
                    p_i0 = self._fid_map[fid_i][0]
                    p_j0 = self._fid_map[fid_j][0]
                    fac_i = p_i0.facility_type or ""
                    fac_j = p_j0.facility_type or ""

                    results.append({
                        'x': float(ip.x), 'y': float(ip.y),
                        'fid_1': fid_i, 'fid_2': fid_j,
                        'feature_id_1': p_i0.feature_id, 'feature_id_2': p_j0.feature_id,
                        'facility_1': fac_i, 'facility_2': fac_j,
                        'z_center_1': float(z_i), 'z_center_2': float(z_j),
                        'radius_1': float(r_i), 'radius_2': float(r_j),
                        'top_1': float(max(0.0, z_i - r_i)), 'bottom_1': float(z_i + r_i),
                        'top_2': float(max(0.0, z_j - r_j)), 'bottom_2': float(z_j + r_j),
                        'overlap_amount': float(overlap_amt),
                    })
        return results

    def export_models_and_csv(self, collisions: List[Dict], out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for c in collisions:
            x2 = math.trunc(c['x'] * 100.0) / 100.0
            y2 = math.trunc(c['y'] * 100.0) / 100.0
            base = f"{c['facility_1']}_{c['facility_2']}_{x2:.2f}_{y2:.2f}"
            safe_base = base.replace('/', '-').replace('\\', '-').replace(' ', '_')

            # 3D 모델 (두 원통)
            # z축 정렬 원통을 교차점 중심에 배치
            ra, rb = c['radius_1'], c['radius_2']
            za, zb = c['z_center_1'], c['z_center_2']
            cylA = trimesh.creation.cylinder(radius=ra, height=self.cylinder_length, sections=self.slices)
            cylA.apply_translation((c['x'], c['y'], za if self.z_up_is_positive else -za))
            cylB = trimesh.creation.cylinder(radius=rb, height=self.cylinder_length, sections=self.slices)
            cylB.apply_translation((c['x'], c['y'], zb if self.z_up_is_positive else -zb))
            scene = trimesh.util.concatenate([cylA, cylB])

            if self.export_format in ("obj", "both"):
                obj_path = os.path.join(out_dir, f"{safe_base}.obj")
                scene.export(obj_path)
                written.append(obj_path)
            if self.export_format in ("glb", "both"):
                glb_path = os.path.join(out_dir, f"{safe_base}.glb")
                scene.export(glb_path)
                written.append(glb_path)

            # CSV 메타 (요구 스키마 + 확장 항목)
            csv_path = os.path.join(out_dir, f"{safe_base}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow([
                    'x', 'y',
                    'facility_type_1', 'feature_id_field_1', 'z_with_radius_1',
                    'facility_type_2', 'feature_id_field_2', 'z_with_radius_2',
                    # 확장 필드
                    'center_z_1', 'center_z_2', 'radius_1(m)', 'radius_2(m)',
                    'top_1', 'bottom_1', 'top_2', 'bottom_2', 'overlap_amount'
                ])
                # z_with_radius는 보수적으로 bottom(더 깊은 면) 사용 예시
                z1_wr = c['bottom_1']
                z2_wr = c['bottom_2']
                w.writerow([
                    f"{x2:.2f}", f"{y2:.2f}",
                    c['facility_1'], c['feature_id_1'], f"{z1_wr:.3f}",
                    c['facility_2'], c['feature_id_2'], f"{z2_wr:.3f}",
                    f"{c['z_center_1']:.3f}", f"{c['z_center_2']:.3f}", f"{c['radius_1']:.3f}", f"{c['radius_2']:.3f}",
                    f"{c['top_1']:.3f}", f"{c['bottom_1']:.3f}", f"{c['top_2']:.3f}", f"{c['bottom_2']:.3f}", f"{c['overlap_amount']:.3f}"
                ])
            written.append(csv_path)
        return written
