from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import os
import math
import csv
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pyproj import CRS


#############################################
# 충돌 결과 객체
#############################################
@dataclass
class CollisionHit:
    x: float
    y: float
    z_a: float  # 파이프 A 중심 심도(m)
    z_b: float  # 파이프 B 중심 심도(m)
    z_a_adj: float  # 반지름 고려 후 깊이(상단/하단 범위 반영)
    z_b_adj: float
    fid_a: int
    fid_b: int
    feature_id_a: str
    feature_id_b: str
    fac_a: str
    fac_b: str
    diam_a_mm: float
    diam_b_mm: float


#############################################
# 내부 유틸
#############################################

def _ensure_epsg5186(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf
    target = CRS.from_epsg(5186)
    if gdf.crs.to_epsg() != 5186:
        return gdf.to_crs(target)
    return gdf


def _truncate2(v: float) -> float:
    # 소수점 둘째 자리까지 절삭
    return math.trunc(v * 100.0) / 100.0


def _interp_z_on_line(line: LineString, anchors_mz: List[Tuple[float, float]], p: Point) -> float:
    """라인 상의 임의 점 p에 대해 anchors(m,z)로 선형보간하여 z 반환."""
    m = line.project(p)
    if not anchors_mz:
        return 0.0
    anchors = sorted(anchors_mz, key=lambda x: x[0])
    left = anchors[0]
    right = anchors[-1]
    for i in range(len(anchors) - 1):
        a0, a1 = anchors[i], anchors[i + 1]
        if a0[0] <= m <= a1[0]:
            left, right = a0, a1
            break
    # 선형보간
    if abs(right[0] - left[0]) < 1e-9:
        return left[1]
    t = (m - left[0]) / (right[0] - left[0])
    return left[1] * (1 - t) + right[1] * t


def _tangent_at(line: LineString, p: Point, eps: float = 0.05) -> Tuple[float, float]:
    """교차점에서 라인의 2D 단위 접선 벡터를 근사."""
    m = line.project(p)
    m0 = max(0.0, m - eps)
    m1 = min(line.length, m + eps)
    p0 = line.interpolate(m0)
    p1 = line.interpolate(m1)
    vx, vy = (p1.x - p0.x), (p1.y - p0.y)
    L = math.hypot(vx, vy)
    if L < 1e-9:
        return 1.0, 0.0
    return vx / L, vy / L


def _build_obj_cylinder(center_xyz: Tuple[float, float, float],
                        axis_dir_xy: Tuple[float, float],
                        radius: float,
                        length: float = 1.0,
                        slices: int = 24,
                        z_up_is_positive: bool = True) -> Tuple[List[str], List[str]]:
    """간단한 원통 메시 OBJ 텍스트 생성 (z축은 깊이 방향으로 사용).
    - center_xyz: (x, y, z) 에서 z는 '깊이'를 양(+)으로 둘지 결정(z_up_is_positive)
    - axis_dir_xy: 관로의 수평 방향 (단위 벡터)
    - radius: 미터
    - length: 원통 길이(미터)
    - slices: 분할 수
    반환: (vertices_lines, faces_lines)
    """
    cx, cy, cz = center_xyz
    # 2D 수평 방향을 기준으로 로컬 직교 벡터 구성
    tx, ty = axis_dir_xy
    # 수평 법선
    nx, ny = -ty, tx
    # 양 끝 중심
    half = length / 2.0
    # z축 방향은 지면 기준 깊이 축. 시각화를 위해 양(+)을 위로 두고 싶으면 반전
    zsign = 1.0 if z_up_is_positive else -1.0

    verts: List[str] = []
    faces: List[str] = []
    # 원 둘레 생성
    top_ring = []
    bot_ring = []
    for i in range(slices):
        ang = 2.0 * math.pi * i / slices
        rx = math.cos(ang) * radius
        ry = math.sin(ang) * radius
        # 로컬 (nx,ny)와 (tx,ty)를 사용해 수평 원 둘레 좌표 만들기
        px = cx + rx * nx + ry * tx
        py = cy + rx * ny + ry * ty
        # 상/하단 z 좌표
        pz_top = cz + zsign * (half)
        pz_bot = cz - zsign * (half)
        top_ring.append((px, py, pz_top))
        bot_ring.append((px, py, pz_bot))

    # OBJ vertices
    for v in top_ring + bot_ring:
        verts.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # 사이드 faces (quad → 두 삼각형)
    n = slices
    for i in range(n):
        a = i + 1
        b = (i + 1) % n + 1
        c = n + b
        d = n + a
        # 두 삼각면
        faces.append(f"f {a} {b} {c}")
        faces.append(f"f {a} {c} {d}")

    return verts, faces


#############################################
# CollisionAnalyzer
#############################################
class CollisionAnalyzer:
    def __init__(self,
                 line_path: str,
                 feature_id_field: str,
                 facility_type_field: str,
                 pipe_size_field: str,
                 fixed_points: List[Dict],
                 z_overlap_rule: str = "center_radius_overlap",  # 현재는 |zA - zB| < rA + rB
                 min_xy_gap: float = 0.01):  # 교차점 주변 분해용 작은 길이(m)
        self.line_path = line_path
        self.feature_id_field = feature_id_field
        self.facility_type_field = facility_type_field
        self.pipe_size_field = pipe_size_field
        self.fixed_points = fixed_points
        self.z_overlap_rule = z_overlap_rule
        self.min_xy_gap = float(min_xy_gap)

        self.collisions: List[CollisionHit] = []

    def _prepare(self):
        lines = gpd.read_file(self.line_path)
        lines = _ensure_epsg5186(lines)
        # explode MultiLineString → LineString
        if hasattr(lines, "explode"):
            lines = lines.explode(index_parts=False, ignore_index=True)
        lines = lines[~lines.geometry.is_empty & (lines.geometry.geom_type == "LineString")].reset_index(drop=True)
        self.lines_gdf = lines

        # feature_id(str) → 라인 인덱스 매핑 (여러 세그먼트일 수 있으므로 list)
        self.feature_lines: Dict[str, List[int]] = {}
        for idx, row in lines.iterrows():
            fid_raw = row[self.feature_id_field]
            fid_str = None if fid_raw is None else str(fid_raw)
            if fid_str is None:
                fid_str = f"__IDX_{idx}"
            self.feature_lines.setdefault(fid_str, []).append(idx)

        # fixed_points 를 feature_id 별로 m,z 앵커로 재구성
        # 같은 feature_id에 대해 라인 세그먼트가 여럿이면 세그먼트별로 별도 anchors 계산
        self.anchors_by_feature: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
        self.meta_by_feature: Dict[str, Dict] = {}

        # 메타: facility_type/diam(mm) 중 대표값 하나 저장 (동일 feature_id 내 동일하다는 가정)
        # 앵커 구성: 각 세그먼트 라인에 대해 (m,z) 리스트 생성
        for fid_str, idx_list in self.feature_lines.items():
            # 대표 메타 찾기
            # fixed_points 중 해당 feature_id의 첫 항목 사용
            meta = None
            for rp in self.fixed_points:
                if str(rp.get('feature_id')) == fid_str:
                    meta = {
                        'facility_type': None,
                        'diam_mm': rp.get('diam', 0.0),
                        'fid_int': rp.get('fid', -1),
                        'feature_id': fid_str,
                    }
                    break
            # 라인에서 facility_type/diam 갱신 (가능하면)
            for idx in idx_list:
                row = lines.iloc[idx]
                if meta is None:
                    meta = {
                        'facility_type': row.get(self.facility_type_field, None),
                        'diam_mm': row.get(self.pipe_size_field, 0.0),
                        'fid_int': -1,
                        'feature_id': fid_str,
                    }
                else:
                    if meta.get('facility_type') is None:
                        meta['facility_type'] = row.get(self.facility_type_field, None)
                    if not meta.get('diam_mm'):
                        meta['diam_mm'] = row.get(self.pipe_size_field, 0.0)
            if meta is None:
                meta = {'facility_type': None, 'diam_mm': 0.0, 'fid_int': -1, 'feature_id': fid_str}
            self.meta_by_feature[fid_str] = meta

            # 세그먼트별 앵커 구성
            for idx in idx_list:
                line = lines.geometry.iloc[idx]
                anchors: List[Tuple[float, float]] = []
                # 해당 라인과 가까운 fixed_points를 m값으로 투영 후 사용
                for rp in self.fixed_points:
                    if str(rp.get('feature_id')) != fid_str:
                        continue
                    p = Point(rp['x'], rp['y'])
                    # 해당 세그먼트 라인에서 포인트가 충분히 가까운지 검사
                    if p.distance(line) < 0.5:  # 50cm 이내면 같은 세그먼트로 간주
                        m = line.project(p)
                        anchors.append((m, rp['z']))
                if anchors:
                    self.anchors_by_feature[(fid_str, idx)] = sorted(anchors, key=lambda t: t[0])

    def _z_with_radius(self, z_center: float, radius_m: float) -> Tuple[float, float]:
        """센터 z에 대해 상단/하단 깊이 반환 (top, bottom). z는 지표로부터 깊이(+)."""
        top = max(0.0, z_center - radius_m)  # 지표 위로는 0으로 클램프
        bottom = z_center + radius_m
        return top, bottom

    def _is_overlap(self, zA: float, rA: float, zB: float, rB: float) -> Tuple[bool, float, float]:
        # 두 파이프의 수직 구간 [zA-rA, zA+rA], [zB-rB, zB+rB]가 겹치면 충돌
        return (abs(zA - zB) < (rA + rB), zA - rA, zB - rB)

    def analyze(self) -> List[CollisionHit]:
        self._prepare()
        lines = self.lines_gdf
        tree = STRtree(list(lines.geometry))

        # 라인 쌍 후보(교차) 탐색
        for i, geom_i in enumerate(lines.geometry):
            # 동일 feature_id의 세그먼트끼리는 스킵
            fid_i = str(lines.iloc[i][self.feature_id_field])
            fac_i = lines.iloc[i].get(self.facility_type_field, None)
            diam_i_mm = lines.iloc[i].get(self.pipe_size_field, 0.0)
            cand = tree.query(geom_i)
            for geom_j in cand:
                j = list(lines.geometry).index(geom_j)
                if j <= i:
                    continue
                fid_j = str(lines.iloc[j][self.feature_id_field])
                if fid_j == fid_i:
                    continue
                fac_j = lines.iloc[j].get(self.facility_type_field, None)
                diam_j_mm = lines.iloc[j].get(self.pipe_size_field, 0.0)

                inter = geom_i.intersection(geom_j)
                if inter.is_empty:
                    continue
                # 포인트/멀티포인트만 처리 (공유구간은 스킵)
                points: List[Point] = []
                if inter.geom_type == 'Point':
                    points = [inter]
                elif inter.geom_type == 'MultiPoint':
                    points = list(inter.geoms)
                else:
                    continue

                for ip in points:
                    # 각 세그먼트에 대해 앵커 확보
                    anchors_i = self.anchors_by_feature.get((fid_i, i), [])
                    anchors_j = self.anchors_by_feature.get((fid_j, j), [])
                    if not anchors_i or not anchors_j:
                        continue
                    # z 보간 (센터 심도)
                    z_i = _interp_z_on_line(geom_i, anchors_i, ip)
                    z_j = _interp_z_on_line(geom_j, anchors_j, ip)

                    r_i = float(diam_i_mm) / 2000.0
                    r_j = float(diam_j_mm) / 2000.0

                    is_hit, z_i_top, z_j_top = self._is_overlap(z_i, r_i, z_j, r_j)
                    if not is_hit:
                        continue

                    meta_i = self.meta_by_feature.get(fid_i, {})
                    meta_j = self.meta_by_feature.get(fid_j, {})

                    hit = CollisionHit(
                        x=float(ip.x), y=float(ip.y),
                        z_a=float(z_i), z_b=float(z_j),
                        z_a_adj=float(z_i), z_b_adj=float(z_j),  # center 기준 (필요시 top/bottom로 확장 가능)
                        fid_a=int(meta_i.get('fid_int', -1)),
                        fid_b=int(meta_j.get('fid_int', -1)),
                        feature_id_a=str(meta_i.get('feature_id')),
                        feature_id_b=str(meta_j.get('feature_id')),
                        fac_a=str(meta_i.get('facility_type')),
                        fac_b=str(meta_j.get('facility_type')),
                        diam_a_mm=float(diam_i_mm), diam_b_mm=float(diam_j_mm),
                    )
                    self.collisions.append(hit)

        return self.collisions

    #############################################
    # 3D 내보내기 (OBJ)
    #############################################
    def export_collision_models(self,
                                out_dir: str,
                                length_m: float = 1.0,
                                slices: int = 24,
                                z_up_is_positive: bool = False,
                                also_csv: bool = True) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []

        # 메타 CSV 준비 (파일별이 아니라 충돌건별 CSV를 요구했으므로 파일마다 1 csv)
        for hit in self.collisions:
            # 파일명 구성
            x2 = _truncate2(hit.x)
            y2 = _truncate2(hit.y)
            base = f"{hit.fac_a}_{hit.fac_b}_{x2:.2f}_{y2:.2f}"
            safe_base = base.replace("/", "-").replace("\\", "-").replace(" ", "_")
            obj_path = os.path.join(out_dir, f"{safe_base}.obj")

            # 라인/탱전 구하기
            # fid → 현재 세그먼트 찾기는 복잡하므로, 단순히 해당 feature_id의 첫 세그먼트 기준 접선 사용
            # (시각화 목적)
            idx_i = self.feature_lines.get(hit.feature_id_a, [None])[0]
            idx_j = self.feature_lines.get(hit.feature_id_b, [None])[0]
            if idx_i is None or idx_j is None:
                continue
            line_i = self.lines_gdf.geometry.iloc[idx_i]
            line_j = self.lines_gdf.geometry.iloc[idx_j]
            tx_i, ty_i = _tangent_at(line_i, Point(hit.x, hit.y))
            tx_j, ty_j = _tangent_at(line_j, Point(hit.x, hit.y))

            # 중심 z는 깊이를 +로 보되, OBJ z축 양(+)을 위로 볼지 여부는 옵션
            center_a = (hit.x, hit.y, hit.z_a)
            center_b = (hit.x, hit.y, hit.z_b)
            ra = hit.diam_a_mm / 2000.0
            rb = hit.diam_b_mm / 2000.0

            v1, f1 = _build_obj_cylinder(center_a, (tx_i, ty_i), ra, length=length_m, slices=slices, z_up_is_positive=z_up_is_positive)
            offset = len(v1)
            v2, f2 = _build_obj_cylinder(center_b, (tx_j, ty_j), rb, length=length_m, slices=slices, z_up_is_positive=z_up_is_positive)

            # OBJ 작성
            with open(obj_path, "w", encoding="utf-8") as fo:
                fo.write("# Collision pair OBJ\n")
                fo.write("o pipeA\n")
                fo.write("\n".join(v1) + "\n")
                fo.write("g pipeA\n")
                fo.write("\n".join(f1) + "\n")
                # 두 번째 오브젝트는 인덱스 오프셋 필요
                # f 인덱스 보정
                f2_fixed = []
                n1 = len(v1)
                for line in f2:
                    parts = line.split()
                    ids = [int(p) + n1 for p in parts[1:]]
                    f2_fixed.append("f " + " ".join(str(i) for i in ids))
                fo.write("o pipeB\n")
                fo.write("\n".join(v2) + "\n")
                fo.write("g pipeB\n")
                fo.write("\n".join(f2_fixed) + "\n")

            written.append(obj_path)

            if also_csv:
                csv_path = os.path.join(out_dir, f"{safe_base}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf:
                    w = csv.writer(cf)
                    # 헤더
                    w.writerow([
                        "x", "y",
                        "facility_type_field_1", "feature_id_field_1", "z_with_radius_1",
                        "facility_type_field_2", "feature_id_field_2", "z_with_radius_2",
                    ])
                    # 반지름 반영 z: 센터 z를 기준으로 상/하단 범위가 겹치는 시나리오 → 보수적으로 더 가까운 상단값 사용 가능
                    z1_with_r = hit.z_a  # 센터 기준(요구에 맞춰 반지름 고려시 변형 가능)
                    z2_with_r = hit.z_b
                    w.writerow([
                        f"{_truncate2(hit.x):.2f}", f"{_truncate2(hit.y):.2f}",
                        hit.fac_a, hit.feature_id_a, f"{z1_with_r:.3f}",
                        hit.fac_b, hit.feature_id_b, f"{z2_with_r:.3f}",
                    ])
                written.append(csv_path)

        return written


#############################################
# 외부 호출 함수
#############################################

def run_collision_analysis(line_path: str,
                           feature_id_field: str,
                           facility_type_field: str,
                           pipe_size_field: str,
                           fixed_points: List[Dict],
                           out_dir: Optional[str] = None,
                           export_models: bool = True) -> Tuple[List[CollisionHit], List[str]]:
    analyzer = CollisionAnalyzer(
        line_path=line_path,
        feature_id_field=feature_id_field,
        facility_type_field=facility_type_field,
        pipe_size_field=pipe_size_field,
        fixed_points=fixed_points,
    )
    hits = analyzer.analyze()
    written: List[str] = []
    if export_models and out_dir is not None:
        written = analyzer.export_collision_models(out_dir)
    return hits, written


#############################################
# 사용 예시 (주석)
#############################################
# from line_depth_fixer import shape_union_run, get_list_result
# fixer = shape_union_run(...)
# points = get_list_result(fixer)
# hits, files = run_collision_analysis(
#     line_path="/path/pipe_lines.shp",
#     feature_id_field="FID",
#     facility_type_field="FTYPE",
#     pipe_size_field="PIPE_D",
#     fixed_points=points,
#     out_dir="/tmp/collisions",
#     export_models=True,
# )
# print(len(hits), "collisions; files:", files)
