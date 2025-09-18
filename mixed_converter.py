# mixedconverter.py
# -*- coding: utf-8 -*-
"""
스레드 기반 변환기: 선형/포인트 Shapefile을 읽어 포인트 심도를 가장 가까운 선형에 매핑하고,
각 선형에 대해 3D 관 메쉬를 생성하여 OBJ/GLB/GLTF로 내보냅니다.


주요 규칙 (스펙에 따라 구현됨):
1) 각 포인트를 가장 가까운 선형에 매핑하고 해당 선형에 부분 포인트로 삽입한다.
2) z = (심도 + 반지름). 선형 끝점은 평균 심도 + 반지름을 사용한다.
3) 인접한 선형 끝점들이 (허용오차 내에서) 서로 다른 z 값을 가질 경우 평균값을 사용한다.
4) 사용자 지정 "intervals"마다 중간 부분 포인트를 삽입하고, 인접한 "알려진 z" 포인트들(끝점 & 매핑된 포인트) 사이를 선형 보간한다.
5) "circle_sections" 분할 수를 가진 원형 링을 생성하고, 링들을 연결하여 삼각형 메쉬를 만든다.
6) OBJ/GLB/GLTF 형식으로 내보낸다. 선택적으로 원점으로 이동하거나 최소 z 값을 지표면(z=0)에 맞춘다.
7) 소스 좌표계가 다를 경우 EPSG:5186으로 재투영한다.
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

import fiona
from shapely.geometry import shape, Point, LineString
from shapely.strtree import STRtree
from shapely.ops import nearest_points, linemerge
from pyproj import CRS, Transformer
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
@dataclass
class PipeFeature:
    feature_id: str
    geom: LineString             # 2D in EPSG:5186
    avg_depth_m: float           # meters (to pipe top)
    diameter_m: float            # meters
    fid: int

class MixedConverter(QObject):
    # ----- 진행/완료 시그널 -----
    progress_changed = Signal(int)
    message = Signal(str)
    finished = Signal(bool) # True=정상완료, False=취소
    error = Signal(str)

    def __init__(self,
                 facility_type: str,
                 line_shp: str,
                 point_shp: str,
                 output_folder: str,
                 feature_id_field: str,
                 avrg_depth_field: str,
                 pipe_size_field: str,
                 point_depth_field: str,
                 circle_sections: int,
                 tolerance: float,
                 export_format: str,
                 node_tol=1e-4):
        super().__init__()
        self.facility_type = facility_type
        self.line_shp = line_shp
        self.point_shp = point_shp
        self.output_folder = output_folder
        self.feature_id_field = feature_id_field
        self.avrg_depth_field = avrg_depth_field
        self.pipe_size_field = pipe_size_field
        self.point_depth_field = point_depth_field
        self.circle_sections = circle_sections
        self.tolerance = tolerance
        self.export_format = export_format.lower()
        self._cancel = False
        self._is_running = True
        self.node_tol = float(node_tol)
        # 노드별 프레임 캐시: key -> (N, B) 또는 (N, B, T)
        self._frame_cache = {}

    # --------- 진행률 스레드 제어 ---------
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

    # --------- 메인 엔트리 ---------
    def run(self):
        """이 메서드는 QThread.start()에 의해 실행되는 메인 작업입니다."""
        try:
            print("변환 작업 시작")
            self._is_running = True # 작업이 진행 중임을 표시

            if not self._tick("선형 Shapefile 읽는 중.....", 2): return
            lines = self._read_lines()
            if not lines:
                raise RuntimeError("선형 피처가 없습니다.")
            print(f"선형 피처 갯수: {len(lines)}")

            if not self._tick("포인트 Shapefile 읽는 중.....", 5): return
            points = self._read_points()
            print(f"포인트 피처 갯수: {len(points)}")
            if not points:
                raise RuntimeError("포인트 피처가 없습니다.")
            
            if not self._tick("포인트 심도를 가장 가까운 선형에 매핑 중.....", 10): return
            mapped = self._map_points_to_lines(lines, points)
            print(f"매핑된 선형 수: {len(mapped)}")
            if not mapped:
                raise RuntimeError("포인트가 어떤 선형에도 매핑되지 않았습니다.")

             # 각 선형에 대해 부분 포인트 생성 및 z 보간
            if not self._tick("간격을 포함한 part point 생성 및 z 계산 중.....", 15): return
            lines_parts: Dict[int, Dict] = {}
            for pf in lines:
                if self._cancel:
                    self.finished.emit(False); return
                xs, ys, zs, svals = self._build_part_points(pf, mapped.get(pf.fid, []))
                lines_parts[pf.fid] = {"xs": xs, "ys": ys, "zs": zs, "s": svals, "pf": pf}

            if not self._tick("연결된 선형 끝점의 z값 보정 중.....", 25): return
            self._harmonize_endpoints(lines_parts)

            if not self._tick("관 생성 및 mesh 조립 시작.....", 35): return
            meshes = []
            total = len(lines)
            idx = 0

            with ThreadPoolExecutor() as executor:
                # 각 작업에 대한 future 객체를 생성합니다.
                # 나중에 future 객체에 직접 접근하기 위해 리스트로 저장합니다.
                future_list = [executor.submit(self._create_mesh_worker, data) for _, data in lines_parts.items()]

                try:
                    # 작업이 완료되는 순서대로 결과를 처리합니다.
                    for idx, future in enumerate(as_completed(future_list), 1):
                        # ▼▼▼ 취소 체크 로직 ▼▼▼
                        # _tick 메서드가 False를 반환하면 사용자 취소를 의미합니다.
                        if not self._tick("관(Tube) 생성 중 ......", 35 + int(30 * idx / max(1, total))):
                            print("사용자 요청으로 작업을 중단합니다. 대기 중인 작업을 취소합니다...")
                            
                            # 아직 시작되지 않은 대기 중인 모든 future를 취소합니다. (Python 3.9+)
                            # 이 호출 후 with 블록이 종료되면서 executor는 완전히 정리됩니다.
                            executor.shutdown(wait=False, cancel_futures=True)
                            
                            # (참고) Python 3.8 이하 버전의 경우 수동으로 취소:
                            # for f in future_list:
                            #     if not f.done():
                            #         f.cancel()
                            
                            return # 함수 실행을 즉시 종료

                        # ▼▼▼ 기존 결과 처리 로직 ▼▼▼
                        try:
                            mesh = future.result()
                            if mesh:
                                meshes.append(mesh)
                                print(f"{len(meshes)}번째 mesh를 만들었습니다")
                        except Exception as e:
                            # 개별 작업에서 발생한 예외 처리
                            print(f"메쉬를 생성하는 동안 오류가 발생했습니다: {e}")
                except KeyboardInterrupt:
                    # Ctrl+C와 같은 강제 종료 신호 처리
                    print("강제 종료 요청. 대기 중인 작업을 취소합니다...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise # 예외를 다시 발생시켜 프로그램이 종료되도록 함

            print(f"생성된 메쉬 개수: {len(meshes)}")

            if not meshes:
                raise RuntimeError("생성된 메쉬가 없습니다.")

            if not self._tick("메시 정리 중.....", 65): return
            # 1) 전체 정렬/원점 이동을 위한 공통 변환 계산
            #    - 모든 메쉬의 버텍스를 모아 한 번에 minZ/centroid를 구함
            all_vertices = []
            for m in meshes:
                if m is not None and m.vertices is not None and len(m.vertices) > 0:
                    all_vertices.append(m.vertices)
            if not all_vertices:
                raise RuntimeError("모든 메쉬가 비어있습니다.")

            all_vertices = np.vstack(all_vertices)
 
            if not self._tick("오브젝트 정리 중.....", 70): return
            # 2) 각 메시에 동일 변환 적용 + 경량 정리 + 메타데이터 부여
            #    - pf 정보를 함께 보관했다면 meshes를 [(pf, mesh)] 형태로 운용하는 것이 이상적
            #    - 현재 코드는 mesh만 append했으므로 이름은 인덱스로 생성
            mesh_records = []
            for i, m in enumerate(meshes):
                if not self._tick("오브젝트 정리 중 ......", 70 + int(16 * (idx + 1) / max(1, len(meshes)))): return

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

                # 메타데이터(필요 항목을 자유롭게 추가)
                m.metadata = {
                    "feature_index": i,
                    "fid": pf.fid,
                    "feature_id": pf.feature_id,                                  # pf를 함께 보관하면 주석 해제
                    "diameter_m": pf.diameter_m,   # "
                    "avg_depth_m": pf.avg_depth_m, # "
                }

                name = f"feature_{i}"  # pf가 있으면 f"pipe_fid_{pf.fid}" 권장
                mesh_records.append((name, m))

            if not self._tick(f"파일 내보내는 중 {self.export_format.upper()}.....", 86): return

            # 3) Scene으로 관별 오브젝트 추가 → 분리된 객체 유지
            base_name = Path(self.line_shp).stem
            out_dir = Path(self.output_folder)
            out_dir.mkdir(parents=True, exist_ok=True)
            ext = {"obj": ".obj", "glb": ".glb", "gltf": ".gltf"}[self.export_format]
            out_file = str(out_dir / f"{base_name}{ext}")

            scene = trimesh.Scene()
            for name, m in mesh_records:
                scene.add_geometry(m, node_name=name, geom_name=name)

            # 4) Export: Scene 그대로 내보내면 Blender에서 오브젝트가 개별로 들어옵니다.
            if self.export_format in ("obj", "glb", "gltf"):
                scene.export(out_file)
            else:
                raise RuntimeError(f"지원하지 않는 내보내기 형식: {self.export_format}")

            print("파일 내보내기 완료.")
            if not self._tick("변환작업 완료", 100): return
            self.message.emit(f"변환 완료: {out_file}")

            if self._is_running:
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
            pf = data["pf"]
            # None 값을 0.0으로 변환
            xs = [0.0 if x is None else x for x in data["xs"]]
            ys = [0.0 if y is None else y for y in data["ys"]]
            zs = [0.0 if z is None else z for z in data["zs"]]

            # 포인트 데이터 생성 및 NaN, Inf 값 처리
            pts3 = np.column_stack([xs, ys, zs])
            pts3 = np.nan_to_num(pts3, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 너무 짧거나 유효하지 않은 데이터는 건너뛰기
            if len(pts3) < 2:
                return None

            start_xy = (xs[0], ys[0])
            end_xy = (xs[-1], ys[-1])
            rad = max(pf.diameter_m / 2.0, 1e-6)

            # 핵심 작업: 메쉬 생성
            mesh = self._sweep_with_node_frames(pts3, start_xy, end_xy, radius=rad, sections=self.circle_sections)

            # 비어 있거나 유효하지 않은 메쉬는 None 반환
            if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
                return None
            
            return mesh
        except Exception as e:
            # 각 스레드에서 발생한 예외를 로깅하거나 처리할 수 있습니다.
            print(f"메쉬 생성 중 오류 발생: {e}")
            return None

    # --------- 좌표계 처리 ---------
    def _get_transform_to_5186(self, src_wkt: str):
        target = CRS.from_epsg(5186)  # Korea 2000 / Central Belt
        try:
            src = CRS.from_wkt(src_wkt) if src_wkt else None
        except Exception:
            src = None
        if src is None or src == target:
            return None  # no transform needed
        return Transformer.from_crs(src, target, always_xy=True)

    # --------- Shapefile 읽기 ---------
    def _read_lines(self) -> List[PipeFeature]:
        feats: List[PipeFeature] = []
        with fiona.open(self.line_shp, 'r') as src:
            transformer = self._get_transform_to_5186(src.crs_wkt)
            for i, rec in enumerate(src):
                geom = shape(rec["geometry"])
                if geom.is_empty:
                    continue
                if transformer:
                    x, y = np.array(geom.coords.xy[0]), np.array(geom.coords.xy[1])
                    x2, y2 = transformer.transform(x, y)
                    geom = LineString(np.column_stack([x2, y2]))

                props = rec["properties"]
                feature_id = props[self.feature_id_field]
                try:
                    avg_depth = float(props[self.avrg_depth_field])
                except Exception:
                    avg_depth = 0.0
                try:
                    diam_mm = float(props[self.pipe_size_field])
                except Exception:
                    diam_mm = 0.0
                feats.append(PipeFeature(
                    feature_id=feature_id,
                    geom=geom,
                    avg_depth_m=avg_depth,
                    diameter_m=diam_mm / 1000.0,  # mm → m
                    fid=i
                ))
        return feats

    def _read_points(self) -> List[Tuple[Point, float]]:
        pts: List[Tuple[Point, float]] = []
        with fiona.open(self.point_shp, 'r') as src:
            transformer = self._get_transform_to_5186(src.crs_wkt)
            for rec in src:
                geom = shape(rec["geometry"])
                if geom.is_empty:
                    continue
                if transformer:
                    x, y = geom.x, geom.y
                    x2, y2 = transformer.transform(x, y)
                    geom = Point(x2, y2)
                props = rec["properties"]
                try:
                    depth = float(props[self.point_depth_field])
                except Exception:
                    depth = 0.0
                pts.append((geom, depth))
        return pts

    # --------- 포인트를 최근접 선형에 매핑 ---------
    def _map_points_to_lines(self, lines: List[PipeFeature], points: List[Tuple[Point, float]]):
        """
        Shapely 2.x STRtree.query / nearest는 정수 인덱스를 반환하고,
        Shapely 1.x에서는 기하 객체를 반환한다. 이 구현은 두 버전을 모두 지원한다.
        """
        line_geoms = [pf.geom for pf in lines]
        index = STRtree(line_geoms)
        mapping: Dict[int, List[Tuple[float, float]]] = {}  # fid -> list of (s_along_line_m, z)

        for pt, depth in points:
            # nearest가 기하 객체 또는 인덱스일 수 있음
            nearest_raw = index.nearest(pt)
            if hasattr(nearest_raw, "distance"):
                nearest_idx = line_geoms.index(nearest_raw)
            else:
                nearest_idx = int(nearest_raw)

            # query도 버전에 따라 인덱스-indices (Shapely 2) or 객체-geometries (Shapely 1) 반환
            cand_raw = index.query(pt.buffer(self.tolerance if self.tolerance > 0 else 0.01))
            # 인덱스 리스트로 정규화
            cand_idxs = []
            if cand_raw is None:
                cand_idxs = []
            else:
                try:
                    # numpy array of indices or objects
                    for item in list(cand_raw):
                        if hasattr(item, "distance"):
                            cand_idxs.append(line_geoms.index(item))
                        else:
                            cand_idxs.append(int(item))
                except TypeError:
                    # single value
                    if hasattr(cand_raw, "distance"):
                        cand_idxs = [line_geoms.index(cand_raw)]
                    else:
                        cand_idxs = [int(cand_raw)]

            if len(cand_idxs) == 0:
                cand_idxs = [nearest_idx]

            # 실제 거리로 가장 가까운 선형 선택
            best_idx = None
            best_d = float('inf')
            for idx in cand_idxs:
                geom = line_geoms[idx]
                d = geom.distance(pt)
                if d < best_d:
                    best_d = d
                    best_idx = idx

            fid = best_idx
            pf = lines[fid]
            s = line_geoms[fid].project(pt)
            radius = pf.diameter_m / 2.0
            z = depth + radius
            mapping.setdefault(fid, []).append((s, z))
        return mapping
        
    # --------- z값이 포함된 부분 포인트 생성 ---------
    def _build_part_points(self, pf: PipeFeature, mapped: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (xs, ys, zs) 배열을 반환하며, 선형 거리(s) 순서대로 정렬된다.
        'mapped'는 특정 거리(s)에서 알려진 z 값을 포함한다.
        """
        line = pf.geom
        total_len = line.length
        # 기본 부분 포인트 수집: 시작 & 끝점 (평균 심도 + 반지름)
        known = [(0.0, pf.avg_depth_m + pf.diameter_m / 2.0)]
        known.append((total_len, pf.avg_depth_m + pf.diameter_m / 2.0))
        # 매핑된 포인트-심도 추가
        for s, z in (mapped or []):
            s = max(0.0, min(total_len, float(s)))
            known.append((s, float(z)))

        # s 기준 정렬
        known.sort(key=lambda t: t[0])

        # 각 s의 좌표 계산
        xs, ys = [], []
        for s, _ in known:
            pt = line.interpolate(s)
            xs.append(pt.x); ys.append(pt.y)

        # z 보간 (양방향)
        zs = [v for _, v in known]
        # 순방향 채우기
        last_known_idx = None
        for i, z in enumerate(zs):
            if z is not None:
                last_known_idx = i
            elif last_known_idx is not None:
                # look ahead to next known
                j = i + 1
                while j < len(zs) and zs[j] is None:
                    j += 1
                if j < len(zs):
                    # linear interpolate between last_known_idx and j
                    s0, z0 = known[last_known_idx][0], zs[last_known_idx]
                    s1, z1 = known[j][0], zs[j]
                    si = known[i][0]
                    t = 0.0 if s1 == s0 else (si - s0) / (s1 - s0)
                    zs[i] = z0 + t * (z1 - z0)
                else:
                    # no future known; extend last known
                    zs[i] = zs[last_known_idx]

        # 역방향 채우기
        next_known_idx = None
        for i in range(len(zs) - 1, -1, -1):
            if zs[i] is not None:
                next_known_idx = i
            elif next_known_idx is not None:
                zs[i] = zs[next_known_idx]
            else:
                zs[i] = pf.avg_depth_m + pf.diameter_m / 2.0

        return np.array(xs), np.array(ys), np.array(zs), np.array([s for s, _ in known])

    # --------- 연결된 선형 끝점 보정 ---------
    def _harmonize_endpoints(self, lines_parts: Dict[int, Dict]):
        """
        lines_parts[fid] = {"xs","ys","zs","s","pf"}
        끝점 XY가 허용오차 내로 만나는 경우, 해당 끝점들의 (XY,Z)를 평균으로 조정한다.
        """
        tol = float(self.tolerance)
        if tol <= 0:
            return

        # 1) 끝점 수집
        eps = []  # (Point(x,y), fid, idx)
        for fid, data in lines_parts.items():
            xs, ys, zs = data["xs"], data["ys"], data["zs"]
            eps.append((Point(xs[0], ys[0]), fid, 0))
            eps.append((Point(xs[-1], ys[-1]), fid, len(xs) - 1))

        pts = [p for p, _, _ in eps]
        tree = STRtree(pts)

        visited = set()
        for i, (p, fid_i, idx_i) in enumerate(eps):
            if (fid_i, idx_i) in visited:
                continue
            # 후보 검색
            cand_raw = tree.query(p.buffer(tol))
            # 인덱스로 정규화
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
            cand_idx = _to_indices(cand_raw)

            cluster = [(fid_i, idx_i)]
            for j in cand_idx:
                pfid, pidx = eps[j][1], eps[j][2]
                if (pfid, pidx) != (fid_i, idx_i) and p.distance(pts[j]) <= tol:
                    cluster.append((pfid, pidx))

            if len(cluster) > 1:
                # 평균 XY 및 Z
                xs_all, ys_all, zs_all = [], [], []
                for pfid, pidx in cluster:
                    d = lines_parts[pfid]
                    xs_all.append(d["xs"][pidx]); ys_all.append(d["ys"][pidx]); zs_all.append(d["zs"][pidx])
                x_avg = float(np.mean(xs_all)); y_avg = float(np.mean(ys_all)); z_avg = float(np.mean(zs_all))
                for pfid, pidx in cluster:
                    d = lines_parts[pfid]
                    d["xs"][pidx] = x_avg; d["ys"][pidx] = y_avg; d["zs"][pidx] = z_avg
                    visited.add((pfid, pidx))
        return

    # --------- 노드 프레임/키 유틸 ---------
    def _node_key(self, xy):
        if xy is None or xy[0] is None or xy[1] is None:
            return None  # 또는 예외/특수키
        t = max(float(self.tolerance), 1e-6)
        return (round(float(xy[0]) / t) * t, round(float(xy[1]) / t) * t)
    
    # 1) 벡터 정규화/정제 유틸
    def _as_unit_vec(self, v, fallback=None):
        v = np.asarray(v, dtype=float)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float) if fallback is None else np.asarray(fallback, dtype=float)
        return v / n

    # 2) 프레임 유효성 검사 & 정제
    def _sanitize_frame(self, frame):
        """(N, B, T) 또는 (N, B) 형태의 프레임을 안전한 float 단위벡터로 변환.
        (N,B)만 오면 T는 N×B로 유도(가능하면) 또는 기본값."""
        if frame is None:
            return None
        try:
            N = self._as_unit_vec(frame[0], fallback=[1,0,0])
            B = self._as_unit_vec(frame[1], fallback=[0,1,0])
            if len(frame) >= 3 and frame[2] is not None:
                T = self._as_unit_vec(frame[2], fallback=[0,0,1])
            else:
                # N, B가 유효하면 T ~ N×B (평행이면 폴백)
                T = np.cross(N, B)
                T = self._as_unit_vec(T, fallback=[0,0,1])
            return (N, B, T)
        except Exception as e:
            print("sanitize_frame error:", e, "frame=", frame)
            return None

    def _sweep_with_node_frames(self, pts3, start_xy, end_xy, radius, sections=24, resample_step=None, seam_align=True):
        P = np.asarray(pts3, float)
        k_start = self._node_key(start_xy)
        k_end   = self._node_key(end_xy)

        start_frame, which = self._pick_start_frame(P, start_xy, end_xy)
        flip = (which == "tail")
        if flip:
            P = P[::-1].copy()
            if hasattr(radius, '__len__') and not callable(radius):
                r = np.asarray(radius, float)
                if len(r) == 2:
                    radius = (float(r[1]), float(r[0]))
                else:
                    radius = r[::-1].copy()
            k_start, k_end = k_end, k_start

        mesh, (N_end, B_end, T_end) = self._tube_sweep(P, radius=radius, sections=sections, start_frame=start_frame, resample_step=resample_step,
            seam_align=seam_align, return_end_frame=True)
        N_end_s, B_end_s, T_end_s = self._sanitize_frame((N_end, B_end, T_end)) or (np.array([1,0,0.],float), np.array([0,1,0.],float), np.array([0,0,1.],float))
        self._frame_cache[k_end] = (N_end_s, B_end_s, T_end_s)
        return mesh

    def _tube_sweep(self, pts, radius=1.0, sections=16, start_frame=None, resample_step=None, seam_align=True, return_end_frame=False):
        import math as _math
        if pts is None:
            raise ValueError("경로가 없습니다.")
        # 원본 점/길이 저장(리샘플 실패 시 복구용)
        P0 = np.asarray(pts, dtype=float)
        if len(P0) < 2:
            raise ValueError("경로점이 부족합니다.")
        # 전체 길이(L0) 계산
        _seg0 = P0[1:] - P0[:-1]
        _len0 = float(np.sum(np.linalg.norm(_seg0, axis=1))) if len(_seg0) else 0.0
        P = P0.copy()
        sections = max(3, int(sections))

        # (선택) 등간격 리샘플
        def _resample_polyline(points, step):
            if step is None or step <= 0:
                return points
            d = np.linalg.norm(points[1:] - points[:-1], axis=1)
            if np.all(d <= 1e-12):
                return points
            s = np.concatenate(([0.0], np.cumsum(d)))
            L = float(s[-1])
            if L <= step:
                return points
            new_s = np.arange(0.0, L, step, dtype=float)
            if new_s[-1] < L:
                new_s = np.append(new_s, L)
            new_pts = []
            j = 0
            for ss in new_s:
                while j+1 < len(s) and s[j+1] < ss:
                    j += 1
                if j+1 >= len(s):
                    new_pts.append(points[-1])
                else:
                    t = (ss - s[j]) / max(s[j+1]-s[j], 1e-12)
                    new_pts.append(points[j]*(1-t) + points[j+1]*t)
            return np.asarray(new_pts)

        # 과대 리샘플 방지: step이 전체 길이보다 크면 자동 보정
        if resample_step is not None and _len0 > 0.0:
            resample_step = min(float(resample_step), max(_len0 * 0.5, 1e-9))
        P = _resample_polyline(P0, resample_step)
        # 리샘플 후 2점 미만이면 원본 복구
        if len(P) < 2 and len(P0) >= 2:
            P = P0.copy()

        seg = P[1:] - P[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        keep = seg_len > 1e-12
        if not np.all(keep):
            P = P[np.concatenate(([True], keep))]
            if len(P) < 2:
                # 안전 탈출: 퇴화 경로는 스킵(호출부에서 None 체크 후 continue)
                return (None, (None, None, None)) if return_end_frame else None
            seg = P[1:] - P[:-1]
            seg_len = np.linalg.norm(seg, axis=1)

        s_edge = np.concatenate(([0.0], np.cumsum(seg_len)))
        L_tot = float(s_edge[-1])
        if L_tot <= 0:
            raise ValueError("경로 길이가 0입니다.")

        T = np.zeros_like(P)
        T[0] = seg[0] / seg_len[0]
        T[-1] = seg[-1] / seg_len[-1]
        for i in range(1, len(P) - 1):
            a = seg[i - 1] / seg_len[i - 1]
            b = seg[i] / seg_len[i]
            v = a + b
            if np.linalg.norm(v) < 1e-12:
                v = b
            T[i] = v / np.linalg.norm(v)

        # 초기 프레임
        if start_frame is not None:
            N = np.asarray(start_frame[0], float)
            B = np.asarray(start_frame[1], float)
            N = N - T[0] * float(np.dot(N, T[0]))
            nrm = np.linalg.norm(N)
            if nrm < 1e-12:
                up = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(T[0], up)) > 0.99:
                    up = np.array([1.0, 0.0, 0.0])
                B = np.cross(T[0], up); B /= np.linalg.norm(B)
                N = np.cross(B, T[0])
            else:
                N /= nrm
                B = np.cross(T[0], N); B /= np.linalg.norm(B)
        else:
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(T[0], up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            B = np.cross(T[0], up); B /= np.linalg.norm(B)
            N = np.cross(B, T[0])

        # 반지름 배열 만들기
        M = len(P)
        if callable(radius):
            R = np.array([float(radius(i, s_edge[i]/L_tot, s_edge[i])) for i in range(M)], dtype=float)
        else:
            if hasattr(radius, '__len__') and not isinstance(radius, (str, bytes)):
                rad_arr = np.asarray(radius, float)
                if len(rad_arr) == 2:
                    r0, r1 = float(rad_arr[0]), float(rad_arr[1])
                    R = r0 + (r1 - r0) * (s_edge / L_tot)
                elif len(rad_arr) == M:
                    R = rad_arr.astype(float)
                else:
                    raise ValueError("radius 배열 길이가 경로 점수와 맞지 않습니다.")
            else:
                R = np.full(M, float(radius), dtype=float)
        if np.any(R <= 0):
            raise ValueError("반지름은 모두 양수여야 합니다.")

        rings = []
        t_prev = T[0].copy()
        N_cur = N.copy()
        B_cur = B.copy()
        for i in range(M):
            if i > 0:
                t_cur = T[i]
                cross = np.cross(t_prev, t_cur)
                norm_c = np.linalg.norm(cross)
                if norm_c > 1e-12:
                    axis = cross / norm_c
                    dot = float(np.clip(np.dot(t_prev, t_cur), -1.0, 1.0))
                    angle = _math.acos(dot)
                    N_cur = self._rotate_vector(N_cur, axis, angle)
                    B_cur = self._rotate_vector(B_cur, axis, angle)
                t_prev = t_cur

            r = R[i]
            ring = []
            for k in range(sections):
                theta = 2.0 * _math.pi * (k / sections)
                offset = r * (np.cos(theta) * N_cur + np.sin(theta) * B_cur)
                ring.append(P[i] + offset)
            rings.append(np.asarray(ring))

        rings = np.asarray(rings)
        V = rings.reshape((-1, 3))
        faces = []
        for i in range(M - 1):
            base0 = i * sections
            base1 = (i + 1) * sections
            for k in range(sections):
                k2 = (k + 1) % sections
                a = base0 + k
                b = base1 + k
                c = base1 + k2
                d = base0 + k2
                faces.append([a, b, d])
                faces.append([a, d, c])

        mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(faces), process=False)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        if return_end_frame:
            return mesh, (N_cur, B_cur, T[-1])
        return mesh

    def _pick_start_frame(self, pts3, start_xy, end_xy):
        k0 = self._node_key(start_xy)
        k1 = self._node_key(end_xy)
        f0 = self._frame_cache.get(k0)
        f1 = self._frame_cache.get(k1)
        print(f"[pick] k0={k0} has={f0 is not None}, k1={k1} has={f1 is not None}")

        T0 = self._as_unit_vec(self._initial_tangent(pts3))  # T0도 보장
        # 디버그용
        # def _dbg_frame(name, f):
        #     if f is None:
        #         print(f"[pick] {name}=None"); return
        #     print(f"[pick] {name} types:", type(f), [None if v is None else getattr(np.asarray(v), 'dtype', None) for v in f])
        #     # 위험 요소 출력
        #     try:
        #         arrs = [np.asarray(v, dtype=object) for v in f]
        #         print(f"[pick] {name} contains None:", any(np.any(a == None) for a in arrs))  # noqa: E711
        #     except Exception as e:
        #         print(f"[pick] {name} check error:", e)

        # _dbg_frame("f0", f0); _dbg_frame("f1", f1)

        def score(frame):
            f = self._sanitize_frame(frame)  # ← 여기서 프레임 정제
            if f is None: 
                return None
            # f는 (N,B,T)
            try:
                c = float(np.clip(np.dot(f[2], T0), -1.0, 1.0))
                return np.arccos(c)
            except Exception as e:
                print("score() error:", e, "frame=", frame)
                return None

        if f0 is None and f1 is None:
            return None, "head"

        sf0 = self._sanitize_frame(f0) if f0 is not None else None
        sf1 = self._sanitize_frame(f1) if f1 is not None else None

        s0 = score(f0); s1 = score(f1)

        if sf0 is not None and (sf1 is None or (s0 is not None and s1 is not None and s0 <= s1)):
            return (sf0[0], sf0[1]), "head"
        return (sf1[0], sf1[1]), "tail"

    def _initial_tangent(self, pts):
        """경로 시작 접선 T0 계산 (입력 정제 및 방어 로직 포함)"""
        P = self._coerce_points_to_float_array(pts)

        # 충분한 점 개수 확인
        if len(P) < 2:
            # 점이 1개 이하인 경우: 기본 접선 반환(또는 예외)
            # 상황에 맞게 정책 결정
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # 첫 세그먼트
        seg = P[1] - P[0]
        n = np.linalg.norm(seg)

        # 첫 세그먼트가 너무 짧거나 비정상일 때 다음 세그먼트로 대체
        if not np.isfinite(n) or n < 1e-12:
            for i in range(1, len(P) - 1):
                seg = P[i + 1] - P[i]
                n = np.linalg.norm(seg)

                if np.isfinite(n) and n > 1e-12:
                    break

        # 모든 세그먼트가 0에 가깝거나 비정상일 때의 최종 방어
        if not np.isfinite(n) or n < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        return seg / n
    
    def _coerce_points_to_float_array(self, pts):
        """
        pts를 (N,3) float64 배열로 강제 변환.
        - None, NaN, Inf → 0.0으로 대체
        - dtype=object → float64로 캐스팅
        """
        # 1) 우선 object로 받아 None을 잡아낸다
        P = np.asarray(pts, dtype=object)

        # 2) None → 0.0 치환
        #    (== 비교는 object 배열에서만 안전; float 배열로 바꾼 뒤에는 쓰지 말 것)
        P = np.where(P == None, 0.0, P)  # noqa: E711

        # 3) float64로 캐스팅 시도 (문자 등 변환 불가 값이 있으면 여기서 에러)
        #    필요 시 try/except로 더 관대하게 처리 가능
        try:
            P = P.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            # 변환 불가한 원소를 0.0으로 대체하는 느슨한 처리
            # 문자열 등 비수치 원소가 섞여 있을 때를 대비
            P_flat = []
            for v in P.ravel():
                try:
                    P_flat.append(float(v))
                except (TypeError, ValueError):
                    P_flat.append(0.0)
            P = np.array(P_flat, dtype=np.float64).reshape(P.shape)

        # 4) NaN/Inf 정리
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # 5) 형상 점검: (N,3) 보장
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"pts must have shape (N,3), got {P.shape}")

        return P
    
    # --------- 폴리라인을 따라 튜브 메쉬 생성 ---------
    def _rotate_vector(self, v, axis, angle):
        v = np.asarray(v, dtype=float)
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-15 or abs(angle) < 1e-15:
            return v
        k = axis / n
        c = math.cos(angle)
        s = math.sin(angle)
        return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)

