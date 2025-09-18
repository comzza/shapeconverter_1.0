# sewageconverter.py
# -*- coding: utf-8 -*-
"""
SewageConverter: 하수관로 2D 선형 Shapefile을 3D 메쉬로 변환하여 OBJ/GLB/GLTF로 저장.
규칙 요약:
 1) 박스형 판단: 가로길이/세로길이 필드 값이 둘 다 0이 아니면 박스, 그 외는 원형
 2) z값 = (심도 + 보정치). 박스=세로길이/2, 원형=직경(mm)/2 → m 로 환산
 3) 선형 시작/끝점에 보정된 z값을 우선 할당
 4) 이웃 관의 연결 끝점 z가 다르면 평균으로 보정
 5) 중간점 간격(intervals)은 원형에만 적용, 시작/끝 z 기준으로 거리 보간
 6) 원형은 circle_seg 만큼 분할된 링, 박스는 사각 링(quad) 생성
 7) 인접 링을 삼각형 두 개로 연결하여 메쉬 생성
 8) 내보내기: OBJ/GLB/GLTF
 9) 좌표계: 입력 .prj가 EPSG:5186과 다르면 EPSG:5186으로 자동 변환
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

import fiona
from shapely.geometry import shape, Point, LineString, MultiLineString
from shapely.ops import linemerge
from shapely.strtree import STRtree
from pyproj import CRS, Transformer
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class PipeFeature:
    geom: LineString   # EPSG:5186 좌표계의 2D 선형
    start_depth_m: float  # 시점 심도 (m)
    end_depth_m: float    # 종점 심도 (m)
    diameter_m: float     # 원형 직경 (m) - mm 입력값을 m로 변환 저장
    width_m: float        # 박스 가로(폭, m)
    height_m: float       # 박스 세로(높이, m)
    pipe_row: int         # 관로열의 갯수
    is_box: bool          # 박스 여부
    fid: int
    feature_id: str       # 관로번호  

class SewerageConverter(QObject):
    # ----- 진행/완료 시그널 -----
    progress_changed = Signal(int)
    message = Signal(str)
    finished = Signal(bool)   # True=정상완료, False=취소
    error = Signal(str)

    def __init__(self,
                 facility_type: str,
                 line_shp: str,
                 output_folder: str,
                 start_depth_field: str,
                 end_depth_field: str,
                 pipe_size_field: str,
                 hol_width_field: str,
                 ver_len_field: str,
                 feature_id_field: str,
                 pipe_row_field: str,
                 circle_seg: int,
                 intervals: float,
                 tolerance: float,
                 export_format: str,
                 ):
        super().__init__()
        self.facility_type = facility_type
        self.line_shp = line_shp
        self.output_folder = output_folder
        self.start_depth_field = start_depth_field
        self.end_depth_field = end_depth_field
        self.pipe_size_field = pipe_size_field
        self.hol_width_field = hol_width_field
        self.ver_len_field = ver_len_field
        self.feature_id_field = feature_id_field
        self.pipe_row_field = pipe_row_field
        self.circle_seg = int(circle_seg)
        self.intervals = float(intervals)
        self.tolerance = float(tolerance)
        self.export_format = export_format.lower()
        self._cancel = False
        self._is_running = True
        # 노드별 프레임 캐시: key -> (N, B) 또는 (N, B, T)
        self._frame_cache = {}

    # ----- 편의: 진행표시 -----
    def _tick(self, msg: str, pct: int):
        if self._cancel:
            self.message.emit("진행 중인 작업이 사용자에 의해 취소되었습니다.")
            # 작업이 중단되었음을 알리기 위해 finished 시그널을 보낼 수 있습니다.
            # self.finished.emit(True)
            return False  # 중단 신호 반환
        if pct is not None:
            self.progress_changed.emit(int(pct))
        if msg:
            self.message.emit(msg)
        return True # 계속 진행
    
    # --------- 변환 취소 스레드 제어 ---------
    def request_cancel(self):
        print("진행 중인 작업 취소하도록 요청합니다.")
        self._cancel = True
        self._is_running = False
        self.progress_changed.emit(0)  # 진행률 초기화
        self.message.emit("작업 취소 요청됨...")     

    # ----- 실행 -----
    def run(self):
        try:
            if not self._tick("선형 Shapefile 읽는 중.....", 5): return
            pipes = self._read_lines()
            if not pipes:
                raise RuntimeError("선형 피처가 없습니다.")
            print(f"선형 피처 갯수: {len(pipes)}")

            if not self._tick("파트 포인트 및 z 계산 중.....", 10): return
            parts: Dict[int, Dict] = {}

            for pf in pipes:
                add_interval = (not pf.is_box)  # 원형만 중간점 간격 적용
                xs, ys, s, total_len = self._sample_polyline(pf.geom, add_interval, self.intervals)
                z0, z1 = self._endpoint_z(pf)
                zs = self._interpolate_z(s, total_len, z0, z1)
                parts[pf.fid] = {"xs": xs, "ys": ys, "zs": zs, "s": s, "pf": pf}

            if not self._tick("연결 끝점 z 보정 중.....", 20): return
            self._harmonize_endpoints(parts)

            if not self._tick("단면 링 생성 및 메쉬 조립 중.....", 40): return
            meshes = []
            total = len(pipes)
            idx = 0
            with ThreadPoolExecutor() as executor:
                future_list = [executor.submit(self._create_mesh_worker, data) for _, data in parts.items()]
                try:
                    for idx, future in enumerate(as_completed(future_list), 1):
                        # ▼▼▼ 취소 체크 로직 ▼▼▼
                        # _tick 메서드가 False를 반환하면 사용자 취소를 의미합니다.
                        if not self._tick("관(Tube) 생성 중.....", 40 + int(30 * (idx + 1) / max(1, total))): 
                            executor.shutdown(wait=False, cancel_futures=True)
                            print("사용자 요청으로 작업을 중단합니다. 대기 중인 작업을 취소합니다...")
                            return
                        try:
                            mesh = future.result()
                            if mesh is not None:
                                meshes.append(mesh)
                                print(f"{idx}번째 mesh를 만들었습니다. >>>>>>")
                        except Exception as e:
                            print(f"메쉬를 생성하는 동안 오류가 발생했습니다: {e}")
                            executor.shutdown(wait=False, cancel_futures=True)
                            return
                except KeyboardInterrupt:
                    print("강제 종료 요청. 대기 중인 작업을 취소합니다...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

            if not meshes:
                raise RuntimeError("생성된 메쉬가 없습니다.")
            print(f"생성된 메쉬의 개수: {len(meshes)}")
            
            if not self._tick("메시 정리 중.....", 70): return            

            all_vertices = []
            for m in meshes:
                if m.vertices is not None and len(m.vertices) > 0:
                    all_vertices.append(m.vertices)
            if not all_vertices:
                raise RuntimeError("모든 메쉬가 비어있습니다.")
            
            all_vertices = np.vstack(all_vertices)

            # ----- 후처리: 정렬 옵션 -----
            if not self._tick("오브젝트 정리 중.....", 74): return
            # 2) 각 메시에 동일 변환 적용 + 경량 정리 + 메타데이터 부여
            #    - pf 정보를 함께 보관했다면 meshes를 [(pf, mesh)] 형태로 운용하는 것이 이상적
            #    - 현재 코드는 mesh만 append했으므로 이름은 인덱스로 생성
            mesh_records = []
            for i, m in enumerate(meshes):
                if not self._tick("오브젝트 정리 중 ......", 74 + int(16 * (idx + 1) / max(1, len(meshes)))): return

                # 정리/노멀
                digits = max(0, int(round(-math.log10(max(1e-4, float(getattr(self, 'tolerance', 1e-4)))))))
                m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)
                m.remove_duplicate_faces()
                m.remove_degenerate_faces()
                m.remove_unreferenced_vertices()
                m.fix_normals()

                # (선택) 재질: glTF/GLB에서 doubleSided로 백페이스 컬링 회피
                try:
                    from trimesh.visual.material import PBRMaterial
                    from trimesh.visual import TextureVisuals
                    mat = PBRMaterial(baseColorFactor=(0.62, 0.69, 1.0, 1.0), doubleSided=True)
                    if m.visual is None or m.visual.kind == 'none':
                        m.visual = TextureVisuals(material=mat)
                    else:
                        m.visual.material = mat
                except Exception:
                    pass

                # 메쉬 개체별 메타데이터 지정
                m.metadata = {"fid": pf.fid, 
                                 "is_box": pf.is_box,
                                 "width_m": pf.width_m,
                                 "height_m": pf.height_m,
                                 "diameter_m": pf.diameter_m,
                                 "start_depth_m": pf.start_depth_m,
                                 "end_depth_m": pf.end_depth_m,
                                 "feature_id": pf.feature_id
                                 }
                name = f"feature_{i}"  # pf가 있으면 f"pipe_fid_{pf.fid}" 권장
                mesh_records.append((name, m))

            # ----- 내보내기 -----
            if not self._tick(f"{self.export_format.upper()}로 내보내는 중.....", 90): return

            out_dir = Path(self.output_folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(self.line_shp).stem
            ext = {"obj": ".obj", "glb": ".glb", "gltf": ".gltf"}[self.export_format]
            out_file = str(out_dir / f"{base_name}{ext}")

            # 관별로 분리된 오브젝트로 추가
            scene = trimesh.Scene()
            for name, m in mesh_records:
                scene.add_geometry(m, node_name=name, geom_name=name)

            if self.export_format == "obj":
                # OBJ도 Scene 단위 내보내기로 객체 분리(o 그룹) 유지
                scene.export(out_file)
            elif self.export_format in ("glb", "gltf"):
                scene.export(out_file)
            else:
                raise RuntimeError(f"지원하지 않는 내보내기 형식: {self.export_format}")

            if not self._tick("변환작업 완료", 100):
                return

            if self._is_running:
                self.message.emit(f"변환 완료: {out_file}")
                self.finished.emit(True)
            else:
                self.message.emit("작업이 사용자에 의해 중지되었습니다.")
                self.finished.emit(False)
        except Exception as e:
            if not self._cancel: # 취소된 경우가 아니라 실제 에러일 때만 보고
                self.error.emit(str(e))
            else:
                self.finished.emit(False) # 취소 완료 시그널

    def _create_mesh_worker(self, data):
        """단일 파이프에 대한 메쉬를 생성하는 작업 단위 함수"""
        try:
            if not self._cancel:
                # None 값을 0.0으로 변환
                pf: PipeFeature = data["pf"]
                xs, ys, zs = data["xs"], data["ys"], data["zs"]
                xs = [0.0 if x is None else x for x in data["xs"]]
                ys = [0.0 if y is None else y for y in data["ys"]]
                zs = [0.0 if z is None else z for z in data["zs"]]

                # 노드 키
                tol = float(self.tolerance)
                k0 = (round(xs[0]/tol)*tol, round(ys[0]/tol)*tol)
                k1 = (round(xs[-1]/tol)*tol, round(ys[-1]/tol)*tol)
                start_frame = None
                if k0 in self._frame_cache:
                    start_frame = (self._frame_cache[k0][0], self._frame_cache[k0][1])

                if pf.is_box:
                    # 타입별 평균 치수에서 시작/끝 폭/높이를 가져와 테이퍼(박스↔박스)에만 사용
                    w0, h0 = pf.width_m, pf.height_m
                    w1, h1 = pf.width_m, pf.height_m

                    if k0 in self._node_size_map and "box" in self._node_size_map[k0]:
                        if self._node_size_map[k0]["box"]["w"]:
                            w0 = self._node_size_map[k0]["box"]["w"]
                        if self._node_size_map[k0]["box"]["h"]:
                            h0 = self._node_size_map[k0]["box"]["h"]
                    if k1 in self._node_size_map and "box" in self._node_size_map[k1]:
                        if self._node_size_map[k1]["box"]["w"]:
                            w1 = self._node_size_map[k1]["box"]["w"]
                        if self._node_size_map[k1]["box"]["h"]:
                            h1 = self._node_size_map[k1]["box"]["h"]

                    rings, segN, end_frame = self._make_rings(
                        xs, ys, zs,
                        circle=False,
                        radius=0.0,
                        width=(w0, w1),
                        height=(h0, h1),
                        circle_seg=self.circle_seg,
                        start_frame=start_frame,
                        return_end_frame=True
                    )
                else:
                    # 원형: 노드 평균 직경으로 테이퍼(원↔원). 박스와는 평균을 섞지 않음.
                    d0 = pf.diameter_m
                    d1 = pf.diameter_m
                    if k0 in self._node_size_map and "circle" in self._node_size_map[k0]:
                        d0 = self._node_size_map[k0]["circle"]["d"]
                    if k1 in self._node_size_map and "circle" in self._node_size_map[k1]:
                        d1 = self._node_size_map[k1]["circle"]["d"]
                    r0 = max(d0/2.0, 1e-6)
                    r1 = max(d1/2.0, 1e-6)

                    rings, segN, end_frame = self._make_rings(
                        xs, ys, zs,
                        circle=True,
                        radius=(r0, r1),
                        width=0.0, height=0.0,
                        circle_seg=self.circle_seg,
                        start_frame=start_frame,
                        return_end_frame=True
                    )
                # 링 연결 → 메쉬     
                mesh = self._rings_to_mesh(rings, segN)

                # ▼▼▼ 방어 코드 추가 ▼▼▼
                # mesh가 성공적으로 생성되었는지, 정점(vertices)이 있는지 확인
                if mesh is None or mesh.is_empty:
                    print(f"피처 {pf.fid}에 대한 메쉬 생성에 실패했거나 비어있어 건너뜁니다.")
                    return None 
            return mesh
        except Exception as e:
            print(f"피처 {pf.fid}에 대한 메쉬 생성에 실패했거나 비어있어 건너뜁니다. ({e})")
            return None

    # ----- 좌표계 변환기 -----
    def _get_transform_to_5186(self, src_wkt: str):
        target = CRS.from_epsg(5186)  # Korea 2000 / Central Belt
        try:
            src = CRS.from_wkt(src_wkt) if src_wkt else None
        except Exception:
            src = None
        if src is None or src == target:
            return None  # 변환 불필요
        return Transformer.from_crs(src, target, always_xy=True)

    # ----- 선형 읽기 -----
    def _read_lines(self) -> List[PipeFeature]:
        feats: List[PipeFeature] = []
        with fiona.open(self.line_shp, 'r') as src:
            transformer = self._get_transform_to_5186(src.crs_wkt)
            for i, rec in enumerate(src):
                geom = shape(rec["geometry"])
                # MultiLineString의 경우 병합
                if isinstance(geom, MultiLineString):
                    geom = linemerge(geom)
                    if isinstance(geom, MultiLineString):
                        # 여전히 분리된 경우 가장 긴 라인만 사용
                        geom = max(list(geom), key=lambda g: g.length)
                if transformer:
                    # 좌표 배열로 변환 후 일괄 변환
                    xs, ys = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
                    x2, y2 = transformer.transform(xs, ys)
                    geom = LineString(np.column_stack([x2, y2]))

                props = rec["properties"]
                feature_id = props[self.feature_id_field]
                def get_float(props, key, default=0.0):
                    try:
                        return float(props[key])
                    except Exception:
                        return float(default)

                start_depth = get_float(props, self.start_depth_field, 0.0)  # m
                end_depth = get_float(props, self.end_depth_field, 0.0)      # m
                diam_mm = get_float(props, self.pipe_size_field, 0.0)        # mm
                width_m = get_float(props, self.hol_width_field, 0.0)        # m
                height_m = get_float(props, self.ver_len_field, 0.0)         # m
                pipe_row_val = get_float(props, self.pipe_row_field, 0.0)            # 관로열수
                pipe_row = int(pipe_row_val)              # 관로열수        

                is_box = (width_m != 0.0 and height_m != 0.0)
                feats.append(PipeFeature(
                    geom=geom,
                    start_depth_m=start_depth,
                    end_depth_m=end_depth,
                    diameter_m=diam_mm / 1000.0,  # mm → m
                    width_m=width_m * pipe_row,
                    height_m=height_m,
                    pipe_row=pipe_row,
                    is_box=is_box,
                    fid=i,
                    feature_id=feature_id
                ))
        return feats

    # ----- 선형 샘플링 (원형/박스 공용) -----
    def _sample_polyline(self, line: LineString, add_interval: bool, interval_m: float):
        """
        반환: (xs, ys, s)  - s는 선형 길이 누적값(거리)
        - 기본 샘플: 원본 선형의 꼭짓점
        - add_interval=True 이면 interval에 맞춰 균등 샘플 추가 (원형에만 적용)
        """
        coords = list(line.coords)
        xs = np.array([p[0] for p in coords], dtype=float)
        ys = np.array([p[1] for p in coords], dtype=float)

        # 꼭짓점 기준 s(누적 거리) 계산
        seg_dx = np.diff(xs)
        seg_dy = np.diff(ys)
        seg_len = np.hypot(seg_dx, seg_dy)
        s_base = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = float(s_base[-1])

        s_list = list(s_base)
        if add_interval and interval_m > 0.0 and total_len > 0:
            n = int(total_len // interval_m)
            for k in range(1, n):
                s_new = round(k * interval_m, 6)
                if s_new not in s_list:
                    s_list.append(s_new)

        s_sorted = np.array(sorted(s_list))
        # s 위치별 좌표를 실제 선형에서 보간
        pts = [line.interpolate(float(s)) for s in s_sorted]
        xs2 = np.array([p.x for p in pts], dtype=float)
        ys2 = np.array([p.y for p in pts], dtype=float)
        return xs2, ys2, s_sorted, total_len

    # ----- 시작/끝 z 계산 -----
    def _endpoint_z(self, pf: PipeFeature) -> Tuple[float, float]:
        if pf.is_box:
            dz = pf.height_m / 2.0
        else:
            dz = (pf.diameter_m / 2.0)
        return pf.start_depth_m + dz, pf.end_depth_m + dz

    # ----- z 배열 생성 (거리 보간) -----
    def _interpolate_z(self, s: np.ndarray, total_len: float, z0: float, z1: float) -> np.ndarray:
        if total_len <= 0:
            return np.full_like(s, z0, dtype=float)
        t = s / total_len
        return z0 + (z1 - z0) * t

    # ----- 끝점 z 조화(평균) -----
    def _harmonize_endpoints(self, parts: Dict[int, Dict]):
        """
        parts[fid] = {"xs","ys","zs","s","pf"}
        - 끝점 XY가 허용오차 내로 만나는 경우, 해당 끝점들의 (XY, Z)를 평균으로 조정
        - 노드별로 타입별 치수(원형: 직경, 박스: 폭/높이)의 평균치를 별도로 계산
        """
        tol = float(self.tolerance)
        if tol <= 0:
            return

        endpoints = []  # (Point(x,y), fid, idx)
        for fid, d in parts.items():
            xs, ys = d["xs"], d["ys"]
            endpoints.append((Point(xs[0], ys[0]), fid, 0))
            endpoints.append((Point(xs[-1], ys[-1]), fid, len(xs) - 1))

        pts = [p for p, _, _ in endpoints]
        tree = STRtree(pts)

        # 1) 클러스터링: XY 근접 끝점 묶기
        clusters = []
        visited = set()

        def _to_indices(raw):
            if raw is None:
                return []
            try:
                items = list(raw)
            except TypeError:
                items = [raw]
            idxs = []
            for it in items:
                if hasattr(it, "distance"):
                    idxs.append(pts.index(it))
                else:
                    idxs.append(int(it))
            return idxs

        for i, (p, fid_i, idx_i) in enumerate(endpoints):
            if (fid_i, idx_i) in visited:
                continue
            cand_idx = _to_indices(tree.query(p.buffer(tol)))
            cluster = [(fid_i, idx_i)]
            for j in cand_idx:
                pfid, pidx = endpoints[j][1], endpoints[j][2]
                if (pfid, pidx) != (fid_i, idx_i):
                    q = pts[j]
                    if p.distance(q) <= tol:
                        cluster.append((pfid, pidx))
            for item in cluster:
                visited.add(item)
            clusters.append(cluster)

        # 2) 각 노드 클러스터별 평균 XY/Z 및 타입별 치수 평균 계산
        #    type_map: node_key -> {'xy':(x,y), 'z':avg_z, 'circle':{'d':..}, 'box':{'w':..,'h':..}}
        self._node_size_map = {}  # 후속 테이퍼 계산용
        for cluster in clusters:
            xs_all, ys_all, zs_all = [], [], []
            circ_ds = []
            box_ws, box_hs = [], []
            for (fid, idx) in cluster:
                d = parts[fid]
                xs_all.append(d["xs"][idx]); ys_all.append(d["ys"][idx]); zs_all.append(d["zs"][idx])
                pf: PipeFeature = d["pf"]
                if pf.is_box:
                    if pf.width_m > 0:  box_ws.append(pf.width_m)
                    if pf.height_m > 0: box_hs.append(pf.height_m)
                else:
                    if pf.diameter_m > 0: circ_ds.append(pf.diameter_m)
            x_avg = float(np.mean(xs_all)); y_avg = float(np.mean(ys_all)); z_avg = float(np.mean(zs_all))

            # 적용: XY, Z 평균치로 스냅
            for (fid, idx) in cluster:
                d = parts[fid]
                d["xs"][idx] = x_avg; d["ys"][idx] = y_avg; d["zs"][idx] = z_avg

            # 노드 키와 타입별 치수 평균 저장
            key = (round(x_avg / tol) * tol, round(y_avg / tol) * tol)
            info = {"xy": (x_avg, y_avg), "z": z_avg}
            if circ_ds:
                info["circle"] = {"d": float(np.mean(circ_ds))}
            if box_ws or box_hs:
                info["box"] = {"w": float(np.mean(box_ws)) if box_ws else None,
                            "h": float(np.mean(box_hs)) if box_hs else None}
            self._node_size_map[key] = info
        return

    def _orthonormalize_nb(self, T, N_in, B_in, eps=1e-12):
        """
        T에 대해 (N,B)를 정규직교화(Gram–Schmidt).
        - 가능한 한 입력 B_in의 '회전 상태'를 보존
        - 우수(right-handed): cross(N,B) · T > 0 되도록 B의 부호 조정
        """
        T = self._safe_unit(T)
        # 1) N: T에 수직화
        N = np.asarray(N_in, dtype=float)
        N = N - T * float(np.dot(N, T))
        Nn = np.linalg.norm(N)
        if not np.isfinite(Nn) or Nn < eps:
            # N이 부실하면 B를 이용해 본다
            if B_in is not None:
                N = np.cross(B_in, T)
                Nn = np.linalg.norm(N)
        if not np.isfinite(Nn) or Nn < eps:
            # 여전히 부실하면 T 기반 기본 기저
            up = np.array([0.0, 0.0, 1.0], float)
            if abs(float(np.dot(T, up))) > 0.99:
                up = np.array([1.0, 0.0, 0.0], float)
            N = np.cross(up, T)
        N = self._safe_unit(N, fallback=[1.0, 0.0, 0.0])

        # 2) B: 입력 B_in을 우선 보존하면서 T,N에 수직화
        if B_in is not None:
            B = np.asarray(B_in, dtype=float)
            B = B - T * float(np.dot(B, T)) - N * float(np.dot(B, N))
            B = self._safe_unit(B, fallback=None)
        else:
            B = None

        if B is None or not np.isfinite(np.linalg.norm(B)) or np.linalg.norm(B) < eps:
            # 입력 B를 못 쓰면 N×T로 생성(주의: 방향은 뒤에서 맞춤)
            B = np.cross(N, T)
            B = self._safe_unit(B, fallback=[0.0, 1.0, 0.0])

        # 3) 우수(right-handed) 보장: cross(N,B)와 T가 같은 방향
        if float(np.dot(np.cross(N, B), T)) < 0.0:
            B = -B
        return N, B

    # 안전 정규화 헬퍼
    def _safe_unit(self, v, eps=1e-12, fallback=None):
        v = np.asarray(v, dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < eps:
            if fallback is None:
                # 아무거나 기본 단위벡터 하나 반환
                return np.array([1.0, 0.0, 0.0], dtype=float)
            return np.asarray(fallback, dtype=float)
        return v / n

    # ----- 단면 링 생성 (원형/박스) -----
    def _make_rings(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                    circle: bool, radius, width, height, circle_seg: int,
                    start_frame=None, return_end_frame=False):
        """단면 링 생성(원형/박스), Parallel Transport 기반 프레임.
        circle=True:
            - radius: float | (r0,r1) | len(N) 배열 | callable(i,t01)->r
        circle=False (box):
            - width,height: float | (w0,w1)/(h0,h1) | len(N) 배열
        """
        import math as _math
        import numpy as np

        # T(접선)과 수직인 (N,B) 직교기저 생성
        def _basis_from_tangent(t):
            t = self._safe_unit(t, fallback=[1.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, 1.0], dtype=float)
            # t와 up이 평행하면 다른 up 선택
            if abs(float(np.dot(t, up))) > 0.99:
                up = np.array([1.0, 0.0, 0.0], dtype=float)
            b = np.cross(t, up)
            b = self._safe_unit(b, fallback=[0.0, 1.0, 0.0])
            n = np.cross(b, t)
            n = self._safe_unit(n, fallback=[0.0, 0.0, 1.0])
            # 수치오차 줄이기 위해 한 번 더 재정렬
            b = np.cross(t, n)
            b = self._safe_unit(b, fallback=[0.0, 1.0, 0.0])
            return n, b

        pts = np.column_stack([xs, ys, zs])
        Np = pts.shape[0]
        if Np < 2:
            raise ValueError("링 생성을 위한 점이 부족합니다.")

        # 접선 T 및 누적길이
        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        s_edge = np.concatenate(([0.0], np.cumsum(seg_len)))
        L = float(s_edge[-1]) if len(s_edge) else 0.0

        T = np.zeros_like(pts)
        # 앞/뒤단 접선 계산 (0 나눗셈 방어)
        T[0]  = seg[0]  / max(seg_len[0], 1e-12)
        T[-1] = seg[-1] / max(seg_len[-1], 1e-12)
        for i in range(1, Np-1):
            a = seg[i-1] / max(seg_len[i-1], 1e-12)
            b = seg[i]   / max(seg_len[i],   1e-12)
            v = a + b
            if np.linalg.norm(v) < 1e-12:
                v = b
            T[i] = v / max(np.linalg.norm(v), 1e-12)

        # ===== 초기 프레임(N,B) — 경고 방어 버전 =====
        if start_frame is not None:
            # 주어진 (N,B)를 T[0]에 직교화 + 정상화
            N_in = np.asarray(start_frame[0], float)
            B_in = np.asarray(start_frame[1], float)
            N0, B0 = self._orthonormalize_nb(T[0], N_in, B_in)
        else:
            # start_frame 없으면 기존 로직과 동일하지만 유틸을 사용
            up = np.array([0.0, 0.0, 1.0], float)
            if abs(float(np.dot(T[0], up))) > 0.99:
                up = np.array([1.0, 0.0, 0.0], float)
            # up을 이용해 초기 N,B 구성
            N_seed = np.cross(up, T[0])
            B_seed = np.cross(N_seed, T[0])
            N0, B0 = self._orthonormalize_nb(T[0], N_seed, B_seed)

        # ===== 사이즈 배열 준비(기존 로직 그대로) =====
        if circle:
            if callable(radius):
                R = np.array([float(radius(i, s_edge[i]/L if L>0 else 0.0)) for i in range(Np)], float)
            else:
                if hasattr(radius, '__len__') and not isinstance(radius, (str, bytes)):
                    rarr = np.asarray(radius, float)
                    if len(rarr) == 2:
                        r0, r1 = float(rarr[0]), float(rarr[1])
                        R = r0 + (r1 - r0) * (s_edge / max(L, 1e-12))
                    elif len(rarr) == Np:
                        R = rarr
                    else:
                        raise ValueError("radius 배열 길이가 경로 점수와 맞지 않습니다.")
                else:
                    R = np.full(Np, float(radius), dtype=float)
            if np.any(R <= 0):
                raise ValueError("반지름은 모두 양수여야 합니다.")
        else:
            def _interp_size(val):
                if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                    arr = np.asarray(val, float)
                    if len(arr) == 2:
                        a0, a1 = float(arr[0]), float(arr[1])
                        return a0 + (a1 - a0) * (s_edge / max(L, 1e-12))
                    elif len(arr) == Np:
                        return arr
                    else:
                        raise ValueError("박스 치수 배열 길이가 경로 점수와 맞지 않습니다.")
                return np.full(Np, float(val), dtype=float)
            W = _interp_size(width)
            H = _interp_size(height)
            if np.any(W <= 0) or np.any(H <= 0):
                raise ValueError("박스 폭/높이는 모두 양수여야 합니다.")

        # ===== Parallel Transport로 링 생성(기존 로직) =====
        rings = []
        Ncur = N0.copy(); Bcur = B0.copy(); t_prev = T[0].copy()
        if circle:
            segN = max(6, int(circle_seg))
            angles = np.linspace(0, 2*np.pi, segN, endpoint=False)
        else:
            segN = 4

        for i in range(Np):
            if i > 0:
                t_cur = T[i]
                cross = np.cross(t_prev, t_cur)
                nrm = np.linalg.norm(cross)
                if nrm > 1e-12:
                    axis = cross / nrm
                    dot = float(np.clip(np.dot(t_prev, t_cur), -1.0, 1.0))
                    ang = _math.acos(dot)
                    def _rot(v, ax, ang):
                        ax = np.asarray(ax, float); v = np.asarray(v, float)
                        n = np.linalg.norm(ax)
                        if n < 1e-15 or abs(ang) < 1e-15:
                            return v
                        k = ax / n
                        c = _math.cos(ang); s = _math.sin(ang)
                        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)
                    Ncur = _rot(Ncur, axis, ang)
                    Bcur = _rot(Bcur, axis, ang)
                t_prev = t_cur

            p = pts[i]
            if circle:
                r = R[i]
                ring = np.array([p + r * (np.cos(a) * Ncur + np.sin(a) * Bcur) for a in angles])
            else:
                hw = W[i] / 2.0; hh = H[i] / 2.0
                ring = np.array([
                    p + (-hw) * Ncur + (-hh) * Bcur,
                    p + ( hw) * Ncur + (-hh) * Bcur,
                    p + ( hw) * Ncur + ( hh) * Bcur,
                    p + (-hw) * Ncur + ( hh) * Bcur,
                ])
            rings.append(ring)

        rings = np.asarray(rings)
        if return_end_frame:
            # T[-1] 기준으로 Ncur, Bcur를 정리해 일관된 종단 프레임 제공
            N_end, B_end = self._orthonormalize_nb(T[-1], Ncur, Bcur)
            return rings, segN, (N_end, B_end, self._safe_unit(T[-1]))
        return rings, segN

# ----- 링 연결 → 메쉬 -----
    def _rings_to_mesh(self, rings: np.ndarray, seg: int) -> trimesh.Trimesh:
        """
        인접한 두 링을 사각형으로 연결하고, 삼각형 두 개로 분할하여 faces 생성
        """
        N = rings.shape[0]
        vertices = rings.reshape(-1, 3)
        faces = []
        for i in range(N - 1):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + ((j + 1) % seg)
                c = (i + 1) * seg + j
                d = (i + 1) * seg + ((j + 1) % seg)
                faces.append([a, b, d])
                faces.append([a, d, c])
        faces = np.array(faces, dtype=np.int64)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

