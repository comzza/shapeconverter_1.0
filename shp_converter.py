# ===== File: converter.py =====
# 변환 작업을 수행하는 워커 클래스 (QThread에서 실행)

from PySide6.QtCore import QObject, Signal
import os
import math
import numpy as np
import fiona
from shapely.geometry import shape, LineString
from shapely.ops import linemerge
from pyproj import Transformer, CRS
import trimesh
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError
import threading
import time

class ConversionWorker(QObject):
    # ----- 진행/완료 시그널 -----
    progress_changed = Signal(int)
    message = Signal(str)
    finished = Signal(bool)   # True=정상완료, False=취소
    error = Signal(str)

    def __init__(self, 
                 facility_type: str,
                 shapefile: str, 
                 output_folder: str, 
                 feature_id_field: str,
                 avrg_depth_field: str, 
                 pipe_size_field: str,
                 circle_sections=16, 
                 resample_interval=1.0, 
                 export_format='glb',
                 tolerance=0.05, 
                 node_tol=1e-4,
                 preserve_vertices=False):
        super().__init__()
        self.facility_type = facility_type
        self.shapefile = shapefile
        self.output_folder = output_folder
        self.feature_id_field = feature_id_field
        self.depth_field = avrg_depth_field
        self.pipe_size_field = pipe_size_field
        self.circle_sections = int(circle_sections)
        self.resample_interval = float(resample_interval)
        self.export_format = export_format.lower()
        self.tolerance = float(tolerance)
        self.preserve_vertices = preserve_vertices
        self._cancel = False
        self._is_running = True
        self.node_tol = float(node_tol)
        self._frame_cache = {}
        self._cancel_event = threading.Event()

    # --------- 진행률 스레드 제어 ---------
    def _tick(self, pct=None, msg=None):
        if self._cancel:
            self.message.emit("진행 중인 작업이 사용자에 의해 취소되었습니다.")
            # 작업이 중단되었음을 알리기 위해 finished 시그널을 보낼 수 있습니다.
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
            base = os.path.splitext(os.path.basename(self.shapefile))[0]
            out_ext = {'obj': '.obj', 'glb': '.glb', 'gltf': '.gltf'}.get(self.export_format, '.glb')
            out_file = os.path.join(self.output_folder, base + out_ext)

            if not self._tick(2, "Shapefile 읽는 중....."): return
            geoms, attrs, crs = self._read_lines()
            print(f"선형 피처 갯수: {len(geoms)}")
            if not geoms:
                raise RuntimeError("LineString 피처가 없습니다.")
            
            if not self._tick(10, "좌표계 준비 중....."): return
            coords_metric, _ = self._to_local_metric(geoms, crs)
            print(f"좌표계 변환 후 피처 개수: {len(coords_metric)}")
            if not coords_metric:
                raise RuntimeError("유효한 좌표계로 변환된 피처가 없습니다.")
            
            if not self._tick(15, "끝점 연결 분석 및 심도 보정 중....."): return
            lines3d = self._build_lines_with_z(coords_metric, attrs)
            print(f"Z 적용 후 피처 개수: {len(lines3d)}")
            if not lines3d:
                raise RuntimeError("Z 좌표가 적용된 피처가 없습니다.")
            
            if not self._tick(20, "라인 리샘플링 중....."): return
            resampled_paths = [self._resample_line(ls, self.resample_interval, self.preserve_vertices) for ls in lines3d]
            print(f"리샘플링 후 피처 개수: {len(resampled_paths)}")
            if not resampled_paths:
                raise RuntimeError("리샘플링된 피처가 없습니다.")
            
            # -------------------------
            # 1) 튜브 생성 단계 (병렬)
            # -------------------------
            meshes = [None] * len(resampled_paths)

            if not self._tick(25, f"튜브 생성 준비 중... ({len(resampled_paths)}개)"): return

            max_workers = min(10, max(1, len(resampled_paths)))  # 필요 시 조정
            ex = ThreadPoolExecutor(max_workers=max_workers)
            print(f"생성된 작업스레드 수: {max_workers}")

            # 튜브 생성작업 병렬처리
            futures = []
            future_to_index = {}
            try:
                for i, path in enumerate(resampled_paths):
                    fut = ex.submit(self._build_tube_for_path, i, path, self.circle_sections)
                    futures.append(fut)
                    future_to_index[fut] = i
                completed = 0
                last_reported = 24  # 시작 이전 값(25부터 표시될 것이므로 24로 초기화)
                total = len(resampled_paths)
                report_every = max(1, total // 100)  # 최대 100번 정도만 UI 갱신(너무 잦은 호출 방지)
                last_tick_time = 0.0                  # 시간 스로틀링용
                tick_interval = 0.05                  # 최소 50ms 간격으로만 UI 갱신 시도

                # 폴링 루프 (취소 신호를 즉시 반영)
                pending = set(futures)
                errors_in_build = 0
                while pending:
                    # 취소가 요청되었으면 대기 중단하고 미시작 작업 취소
                    if self._cancel_event.is_set():
                        # 아직 시작 안 한 작업 취소
                        for f in pending:
                            f.cancel()
                        # 실행 중/대기중인 작업 취소 예약 + 비대기 셧다운
                        ex.shutdown(wait=False, cancel_futures=True)
                        return
                    done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    # 완료된 것 처리
                    for fut in done:
                        idx = future_to_index.get(fut, -1)
                        try:
                            i, tube, had_error = fut.result()
                        except CancelledError:
                            continue
                        except Exception as e:
                            print(f"{idx}번째 튜브 생성 작업 예외 고지: {e!r}")
                            i, tube, had_error = -1, None, True   # 안전 기본값

                        if had_error:
                            errors_in_build += 1

                        if tube is not None and i >= 0:
                            meshes[i] = tube
                            print(f"{i}번째 mesh 변환 작업 완료 >>>>>>> ")
                        completed += 1
                        # 진행률 계산: 
                        # 1) 완료 기반 스로틀: 1% 단위(or report_every)로만 시도
                        should_try_by_count = (completed % report_every == 0) or (completed == total)

                        # 2) 시간 스로틀: 최소 tick_interval 간격 유지
                        now = time.perf_counter()
                        should_try_by_time = (now - last_tick_time) >= tick_interval

                        if should_try_by_count or should_try_by_time:
                            # 소수로 계산(내림 최소화), 단 표시 직전에만 int로 변환
                            progress_f = 25.0 + (45.0 * completed / max(1, total))
                            new_progress = int(progress_f)  # 신호는 정수여야 한다면 마지막에만 int
                            if new_progress > last_reported:
                                if not self._tick(new_progress):
                                    self._cancel_event.set()
                                    break
                                last_reported = new_progress
                                last_tick_time = now  # 성공적으로 반영했을 때만 갱신
                # 정상 완료 시 스레드풀 정리
                ex.shutdown(wait=True)
            except Exception:
                # 예기치 못한 예외
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError("튜브 생성 중에 오류가 발생하였습니다. 다시 시도해주세요.")
            finally:
                # 예외/취소 등으로 빠질 때 안전장치
                if not ex._shutdown:
                    # Python 3.9+: cancel_futures=True로 미시작 작업 제거
                    ex.shutdown(wait=False, cancel_futures=True)

            if self._cancel_event.is_set():
                return

            # 유효한 메시 존재 확인
            valid_meshes = [m for m in meshes if m is not None]
            if not valid_meshes:
                raise RuntimeError("생성된 메시가 없습니다.")

            if not self._tick(71, "메시 정리 중....."): return

            # -------------------------
            # 공통 변환 계산 대비: 전체 버텍스 스택
            # -------------------------
            all_vertices = []
            for m in valid_meshes:
                if m.vertices is not None and len(m.vertices) > 0:
                    all_vertices.append(m.vertices)

            if not all_vertices:
                raise RuntimeError("모든 메쉬가 비어있습니다.")

            all_vertices = np.vstack(all_vertices)

            # (필요 시 여기서 all_vertices 기반 정렬/공통변환을 계산/적용)
            if not self._tick(75, "3D 모델 개별 개체 경량화 처리 중....."): return

            # -------------------------
            # 2) 개별 메시 경량화 단계 (병렬)
            # -------------------------
            mesh_records = [None] * len(valid_meshes)
            # tolerance → digits 한 번만 계산
            digits = max(0, int(round(-math.log10(max(1e-4, float(getattr(self, 'tolerance', 1e-4)))))))
            # 개별 메시 경량화 병렬 처리
            ex = ThreadPoolExecutor(max_workers=max_workers)   # 새로 생성
            try:
                futures = []
                for i, m in enumerate(meshes):
                    if m is None:
                        continue
                    futures.append(ex.submit(self._lighten_mesh_record, i, m, digits, attrs[i]))

                completed = 0
                last_reported = 74  # 시작 이전 값(25부터 표시될 것이므로 24로 초기화)
                total_to_lighten = len(futures)
                report_every = max(1, total_to_lighten // 100)  # 최대 100번 정도만 UI 갱신(너무 잦은 호출 방지)

                # 폴링 루프 (취소 신호를 즉시 반영)
                pending = set(futures)
                errors_in_build = 0

                while pending:
                    # 취소가 요청되었면 대기 중단하고 미시작 작업 취소
                    if self._cancel_event.is_set():
                        # 아직 시작 안 한 작업 취소
                        for f in pending:
                            f.cancel()
                        # 실행 중/대기중인 작업 취소 예약 + 비대기 셧다운
                        ex.shutdown(wait=False, cancel_futures=True)
                        return

                    done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)

                    # 취소 상태면 남은 결과만 회수하고 중단
                    for fut in done:
                        if not done:
                            continue
                        try:
                            i, rec = fut.result()
                        except CancelledError:
                            # cancel()된 경우
                            continue
                        except Exception:
                            # 예기치 못한 예외
                            i, rec = -1, None  # 예외 시 사용할 안전한 인덱스

                        if rec is not None:
                            mesh_records.append(rec)

                        completed += 1
                        # 진행률 계산: 75 -> 94 구간(약 18포인트)
                        if (completed % report_every == 0) or (completed == total_to_lighten):
                            new_progress = 75 + int(18 * completed / max(1, total_to_lighten))
                            if new_progress > last_reported:
                                if not self._tick(new_progress):
                                    self._cancel_event.set()
                                    break
                                last_reported = new_progress    

                # 정상 완료 시 스레드풀 정리
                ex.shutdown(wait=True)
            finally:
                # 예외/취소 등으로 빠질 때 안전장치
                if not ex._shutdown:
                    # Python 3.9+: cancel_futures=True로 미시작 작업 제거
                    ex.shutdown(wait=False, cancel_futures=True)

            if self._cancel_event.is_set():
                return

            # None 제거 및 압축
            mesh_records = [r for r in mesh_records if r is not None]

            if not mesh_records:
                raise RuntimeError("메시 경량화 결과가 비어있습니다.")

            print(f"정렬/정리 후 메시 개수: {len(mesh_records)}")

            # 3) Scene에 개별 오브젝트로 추가
            scene = trimesh.Scene()
            for name, m in mesh_records:
                scene.add_geometry(m, node_name=name, geom_name=name)

            # (선택) 노멀 강제 계산 - 씬엔 필요 없지만 호환성 위해
            # 각 메시에서 이미 fix_normals 호출, 추가 작업 불필요

            # Export
            if not self._tick(93, f"{self.export_format.upper()} 파일 내보내는 중....."): return
            self._export_scene(scene, out_file)

            if not self._tick(100, "변환작업 완료"): return

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

    def _build_tube_for_path(self, i, path, circle_sections):
        """
        개별 path -> tube 생성 (스레드에서 실행)
        반환: (i, tube or None, error_flag)
        """
        try:
            # 취소 요청되었으면 즉시 반환 (미완료로 처리)
            if self._cancel_event.is_set():
                return (i, None, False)
            
            if len(path) < 2:
                return (i, None, False)

            pts = path[:, :3]
            if self._cancel_event.is_set():
                return (i, None, False)
            try:
                tube = self._sweep_with_node_frames(
                    pts,
                    path[0, :2],
                    path[-1, :2],
                    radius=(path[0, 3] / 2.0, path[-1, 3] / 2.0),
                    sections=circle_sections,
                ) 
            except Exception:
                if self._cancel_event.is_set():
                    return (i, None, False)
                pts = self._dedupe_points(pts)
                if len(pts) >= 2 and self._cancel_event.is_set():
                    tube = self._sweep_with_node_frames(
                        pts,
                        path[0, :2],
                        path[-1, :2],
                        radius=(path[0, 3] / 2.0, path[-1, 3] / 2.0),
                        sections=circle_sections,
                    )
                else:
                    tube = None
            if self._cancel_event.is_set():
                return (i, None, False)
            
            return (i, tube, False)
        except Exception:
            # worker 내부 에러는 상위에서 판단
            return (i, None, True)

    def _lighten_mesh_record(self,i, m, digits, meta):
        """
        메시 경량화(스레드에서 실행)
        반환: (i, (name, mesh)) 또는 (i, None)  # None이면 실패/스킵
        """
        try:
            if m is None or m.vertices is None or len(m.vertices) == 0:
                return (i, None)
            # 동일 정리 절차
            m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=digits)
            if self._cancel_event.is_set():
                return (i, None)
            m.remove_duplicate_faces()
            if self._cancel_event.is_set():
                return (i, None)
            m.remove_degenerate_faces()
            if self._cancel_event.is_set():
                return (i, None)
            m.remove_unreferenced_vertices()
            if self._cancel_event.is_set():
                return (i, None)
            m.fix_normals()
            if self._cancel_event.is_set():
                return (i, None)
            m.metadata = {"feature_index": i, **meta}
            name = f"feature_{i}"
            if self._cancel_event.is_set():
                return (i, None)
            return (i, (name, m))
        except Exception:
            return (i, None)

    # --------- Shapefile 읽기 ---------
    def _read_lines(self):
        geoms = []
        attrs = []
        with fiona.open(self.shapefile, 'r') as src:
            crs = src.crs_wkt or src.crs
            for feat in src:
                geom = shape(feat['geometry'])
                if geom is None:
                    continue
                if geom.geom_type == 'MultiLineString':
                    geom = linemerge(geom)
                if geom.geom_type != 'LineString':
                    continue
                props = feat['properties'] or {}
                # 규칙 1: 필드 읽기
                if self.depth_field not in props or self.pipe_size_field not in props or self.feature_id_field not in props:
                    raise RuntimeError(f"DBF에 필드가 없습니다: '{self.depth_field}', '{self.pipe_size_field}','{self.feature_id_field}'")
                feature_id = props[self.feature_id_field]   # 관로번호
                depth_val = float(props[self.depth_field])   # 평균심도
                pipe_mm = float(props[self.pipe_size_field])      # 관로구경(mm)
                pipe_m = pipe_mm / 1000.0                    # 관로구경(m)
                geoms.append(geom)
                attrs.append({'featureID': feature_id,'depth_m': depth_val, 'diam_m': pipe_m})
        if not geoms:
            raise RuntimeError("LineString 피처가 없습니다.")
        return geoms, attrs, crs

    # --------- 좌표계 처리 ---------
    def _to_local_metric(self, geoms, crs):
        try:
            crs_in = CRS.from_user_input(crs) if crs else None
        except Exception:
            crs_in = None

        # 좌표가 lon/lat처럼 보이는지 간단 판정
        def looks_like_lonlat(G):
            pts = np.array([p for g in G for p in g.coords], dtype=float)
            return np.all((np.abs(pts[:, 0]) <= 180.0) & (np.abs(pts[:, 1]) <= 90.0))
        
        wgs84 = CRS.from_epsg(4326)
        
        # 1) CRS가 아예 없고, lon/lat처럼도 안 보이면 -> 이미 미터계라고 가정하고 통과
        if crs_in is None and not looks_like_lonlat(geoms):
            return geoms, None  # 변환 없음(이미 로컬 미터)
        
        if crs_in and crs_in.to_epsg() == 4326:
            geoms_wgs = geoms
        else:
            to_wgs = Transformer.from_crs(crs_in, wgs84, always_xy=True) if crs_in else None
            geoms_wgs = []
            for g in geoms:
                if to_wgs is None:
                    geoms_wgs.append(g)
                else:
                    xs, ys = zip(*list(g.coords))
                    lon, lat = to_wgs.transform(xs, ys)
                    geoms_wgs.append(LineString(list(zip(lon, lat))))

        all_pts = np.array([p for g in geoms_wgs for p in g.coords])
        clon = float(np.mean(all_pts[:, 0]))
        clat = float(np.mean(all_pts[:, 1]))
        utm_zone = int((clon + 180) // 6) + 1
        is_northern = clat >= 0
        epsg = 32600 + utm_zone if is_northern else 32700 + utm_zone
        local_metric = CRS.from_epsg(epsg)
        to_metric = Transformer.from_crs(wgs84, local_metric, always_xy=True)

        geoms_metric = []
        for g in geoms_wgs:
            xs, ys = zip(*list(g.coords))
            X, Y = to_metric.transform(xs, ys)
            geoms_metric.append(LineString(list(zip(X, Y))))

        return geoms_metric, to_metric

    # --------- Z 계산(심도 적용) & 끝점 보정 ---------
    def _build_lines_with_z(self, geoms_metric, attrs):
        """
        처리 순서:
        1) 각 라인의 p0, p1 끝점에만 z 계산 (depth가 0/None이면 관종별 기본심도 적용)
        2) 허용오차(self.tolerance, 기본 0.2m) 내 연결되는 끝점들을 노드로 묶어
            - Z는 노드 평균값으로 정합
            - XY는 노드 평균 좌표로 스냅(연결점이 2개 이상일 때만 스냅)
        3) XY/Z가 정합된 2점짜리 라인 배열 반환 (리샘플링은 별도 함수에서 수행)
        """

        # -------- 관종 판별 / 기본심도 --------
        def _default_depth_by_kind(kind_str: str) -> float:
            # 상수 1.2 / 하수 2.0 / 전기 1.5 / 가스 1.7 / 통신 0.7
            s = (kind_str or '').lower()
            if ('하수' in s) or ('sewer' in s) or ('drain' in s):
                return 2.0
            if ('전기' in s) or ('electric' in s) or ('power' in s):
                return 1.5
            if ('가스' in s) or ('gas' in s):
                return 1.7
            if ('통신' in s) or ('telecom' in s) or ('comm' in s):
                return 0.7
            if ('상수' in s) or ('수도' in s) or ('water' in s):
                return 1.2
            return 1.2  # 기본값

        # -------- 1) 끝점 Z 계산 --------
        endpoints = []  # 각 라인의 p0/p1과 z, 직경, 원본 geom 보관
        for g, a in zip(geoms_metric, attrs):
            x0, y0 = g.coords[0]
            x1, y1 = g.coords[-1]
            diam = float(a.get('diam_m', 0.0))

            depth_raw = a.get('depth_m', None)
            try:
                depth_val = float(depth_raw) if depth_raw is not None else None
            except Exception:
                depth_val = None

            if (depth_val is None) or (depth_val == 0.0):
                depth_val = _default_depth_by_kind(self.facility_type)

            # 끝점 Z = -(심도 + 반지름)
            z_end = - (depth_val + diam / 2.0)

            endpoints.append({
                'p0': (x0, y0),
                'p1': (x1, y1),
                'z0': z_end,
                'z1': z_end,
                'diam': diam,
                'geom': g
            })

        # -------- 2) 연결 노드 정합( Z 평균 + XY 스냅 ) --------
        tol = self.tolerance
        tol = 0.2 if tol is None else max(float(tol), 1e-9)

        def _key(pt):
            return (round(pt[0] / tol), round(pt[1] / tol))

        # 노드 구성: 같은 버킷에 들어오면 '연결'로 간주
        node_map = {}  # key -> [(i, 'p0'|'p1'), ...]
        for i, e in enumerate(endpoints):
            node_map.setdefault(_key(e['p0']), []).append((i, 'p0'))
            node_map.setdefault(_key(e['p1']), []).append((i, 'p1'))

        # 노드별 평균 XY/Z 계산
        node_stat = {}  # key -> {'xy_mean': (x,y), 'z_mean': float, 'count': int}
        for k, lst in node_map.items():
            xs, ys, zs = [], [], []
            for (i, tag) in lst:
                e = endpoints[i]
                x, y = e[tag]
                z = e['z0'] if tag == 'p0' else e['z1']
                xs.append(x); ys.append(y); zs.append(z)
            node_stat[k] = {
                'xy_mean': (float(np.mean(xs)), float(np.mean(ys))),
                'z_mean':  float(np.mean(zs)),
                'count':   len(lst),
            }

        # 평균값 적용:
        # - count >= 2 인 노드: XY 스냅 + Z 평균값 적용
        # - count == 1 인 노드: 현재값 유지
        for k, lst in node_map.items():
            stat = node_stat[k]
            if stat['count'] >= 2:
                xy_mean = stat['xy_mean']
                z_mean = stat['z_mean']
                for (i, tag) in lst:
                    if tag == 'p0':
                        endpoints[i]['p0'] = xy_mean
                        endpoints[i]['z0'] = z_mean
                    else:
                        endpoints[i]['p1'] = xy_mean
                        endpoints[i]['z1'] = z_mean
            # count == 1: 아무 것도 하지 않음(유지)

        # -------- 3) 2점 라인 배열로 반환(리샘플링 단계에서 내부점 생성) --------
        lines3d = []
        for e in endpoints:
            x0, y0 = e['p0']; x1, y1 = e['p1']
            z0, z1 = e['z0'], e['z1']
            d = e['diam']
            line_arr = np.array([[x0, y0, z0, d], [x1, y1, z1, d]], dtype=float)
            lines3d.append(line_arr)

        return lines3d


    # --------- 리샘플링 ---------
    def _resample_line(self, line_arr, interval, preserve_vertices: bool = False):
        """
        line_arr: shape (N, 4) -> [x, y, z, d]
        - z: 시작/끝점은 이미 _build_lines_with_z에서 확정된 값이어야 함
        - d: 직경(끝점이 다르면 테이퍼링 선형보간)

        interval: 리샘플 간격(m). None/<=0이면 좌표는 유지하되 z는 보간해 갱신
        preserve_vertices:
        - False: 균일 간격 샘플로만 좌표 재구성(기본)
        - True : 기존 정점 + 균일 간격 지점 모두 포함 (기존 점 보존)

        공통 규칙:
        - 모든 내부점 z는 누적길이 비율로 시작~끝 z를 선형보간하여 할당(요구사항 4)
        - 직경 d는 원본 정점 기준으로 선형보간(끝점만 있으면 끝점 기준 보간)
        """
        # 기본 검사
        if line_arr is None or len(line_arr) < 2:
            return line_arr

        xy = line_arr[:, :2]
        z  = line_arr[:, 2]
        d  = line_arr[:, 3]

        # 누적 길이 s_orig (원본 정점 기준)
        diffs = np.diff(xy, axis=0)
        if len(diffs) == 0:
            # 한 점만 있는 경우
            return line_arr
        seg = np.linalg.norm(diffs, axis=1)
        s_orig = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s_orig[-1])
        if total <= 0.0:
            # 길이가 0이면 그대로 반환
            return line_arr

        # 균일 분할 위치(s_ins): 끝점(0,total)은 제외
        if interval is not None and interval > 0:
            k = int(math.floor(total / interval))
            s_ins = (np.arange(1, k) * interval) if k >= 2 else np.array([], dtype=float)
        else:
            s_ins = np.array([], dtype=float)


        if interval is not None and interval > 0:
            if preserve_vertices:
                # 기존 정점 + 균일 분할점 모두 포함
                s_all = np.unique(np.concatenate([s_orig, s_ins]))
            else:
                # 균일 간격만 사용(끝점 포함)
                n = max(2, int(math.ceil(total / interval)) + 1)
                s_all = np.linspace(0.0, total, n)
        else:
            # 간격 없음: 기존 정점만 사용
            s_all = s_orig
            
        # XY 보간 (s_orig 기준으로 좌표 선형보간)
        x_all = np.interp(s_all, s_orig, xy[:, 0])
        y_all = np.interp(s_all, s_orig, xy[:, 1])

        # z 보간 (요구사항 4: 내부점 z는 시작~끝 z로 길이 보간)
        z0 = float(z[0]); z1 = float(z[-1])
        z_all = np.interp(s_all, [0.0, total], [z0, z1])

        # 직경 d 보간 (원본 정점 d를 기준으로 선형보간 → 테이퍼링 지원)
        d_all = np.interp(s_all, s_orig, d)

        return np.column_stack([x_all, y_all, z_all, d_all])



    # --------- 튜브 스윕(Parallel Transport) 구현 ---------
    def _node_key(self, xy):
        t = max(self.tolerance, 1e-6)
        return (round(xy[0] / t) * t, round(xy[1] / t) * t)

    def _initial_tangent(self, pts):
        P = np.asarray(pts, float)
        if len(P) < 2:
            return np.array([1.0, 0.0, 0.0], float)
        seg = P[1] - P[0]
        n = np.linalg.norm(seg)
        if n < 1e-12:
            for i in range(1, len(P)-1):
                seg = P[i+1] - P[i]
                n = np.linalg.norm(seg)
                if n > 1e-12:
                    break
        return seg / max(n, 1e-12)

    # 관 연결단면간 맞붙을 경사각도 계산
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

    def _sweep_with_node_frames(self, pts, start_xy, end_xy, radius, sections=24, resample_step=None, seam_align=True):
        """
        - pts: Nx3 polyline 경로
        - start_xy, end_xy: 경로 양 끝 노드 XY (스냅 전 원본 XY여도 무방)
        - radius: float | (r0, r1) | len(P)의 배열 | callable(i, t01, s)->r
        - sections, resample_step, seam_align: _tube_sweep 전달
        """
        P = np.asarray(pts, float)
        P = self._dedupe_points(P)
        if len(P) < 2:
            print(f"[SKIP] 경로점 부족: {len(P)}개 ({start_xy} -> {end_xy})")
            return None

        k_start = self._node_key(start_xy)
        k_end   = self._node_key(end_xy)

        start_frame, which = self._pick_start_frame(P, start_xy, end_xy)

        # 뒤쪽 노드 프레임을 쓸 경우 경로/반지름을 뒤집어 시작-끝을 바꿈
        flip = (which == "tail")
        if flip:
            P = P[::-1].copy()
            if isinstance(radius, (list, tuple, np.ndarray)) and not callable(radius):
                if len(radius) == 2:
                    radius = (float(radius[1]), float(radius[0]))
                else:
                    radius = np.asarray(radius)[::-1].copy()

            # 시작/끝 키도 뒤집기
            k_start, k_end = k_end, k_start

        if len(P) < 2:
            print(f"{len(P)}개의 경로점이 부족하는 선형: {k_start} -> {k_end}")
            return None

        # 가변 반지름 + 시작 프레임으로 스윕
        mesh, (N_end, B_end, T_end) = self._tube_sweep(
            P, radius=radius, sections=sections,
            start_frame=start_frame,
            resample_step=resample_step,
            seam_align=seam_align,
            return_end_frame=True
        )
        # 끝 노드 프레임을 캐시에 저장 → 다음 관로가 이 노드에서 시작할 때 시임 정렬 보장
        self._frame_cache[k_end] = (N_end, B_end, T_end)
        return mesh

    def _dedupe_points(self, pts):
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - out[-1]) > 1e-9:
                out.append(p)
        return np.array(out)


    def _tube_sweep(self, pts, radius=1.0, sections=16, start_frame=None, resample_step=None, seam_align=True, return_end_frame=False):
        import math as _math
        if pts is None:
            raise ValueError("경로가 없습니다.")
        P = self._dedupe_points(np.asarray(pts, dtype=float))
        if len(P) < 2:
            raise ValueError("경로점이 부족합니다.")
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

        P = _resample_polyline(P, resample_step)

        seg = P[1:] - P[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        keep = seg_len > 1e-12
        if not np.all(keep):
            P = P[np.concatenate(([True], keep))]
            if len(P) < 2:
                raise ValueError("유효한 경로가 없습니다.")
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
            # N을 T[0]에 직교로 보정
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
                faces.append([a, b, c])
                faces.append([a, c, d])

        mesh = trimesh.Trimesh(vertices=V, faces=np.asarray(faces), process=False)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        if return_end_frame:
            return mesh, (N_cur, B_cur, T[-1])
        return mesh



    def _pick_start_frame(self, pts, start_xy, end_xy):
        """
        두 노드 중 프레임 캐시가 있는 쪽을 우선 사용하되,
        둘 다 있으면 '회전량이 작은' 프레임을 선택.
        """
        k0 = self._node_key(start_xy)
        k1 = self._node_key(end_xy)
        f0 = self._frame_cache.get(k0)  # (N, B) 또는 (N, B, T)
        f1 = self._frame_cache.get(k1)

        T0 = self._initial_tangent(pts)  # 경로 진행 방향(+)

        def score(frame):
            # 프레임에 T가 있으면 T와 T0의 정합도를 사용(각도 작을수록 좋음)
            if frame is None: 
                return None
            if len(frame) >= 3:
                T_cached = np.asarray(frame[2], float)
                c = float(np.clip(np.dot(T_cached, T0), -1.0, 1.0))
                # 각도 = arccos(c). 값이 작을수록 매칭이 잘 맞음
                return math.acos(c)
            # T가 없으면 우선순위만
            return math.pi  # 보통 뒤로 미룸

        s0 = score(f0)
        s1 = score(f1)

        start_frame = None
        which = None  # "head" or "tail" (경로의 앞/뒤에서 시작할지)
        if f0 is None and f1 is None:
            # 둘 다 없으면 None 반환 → _tube_sweep이 전역 up으로 시작
            return None, "head"
        if f0 is not None and (f1 is None or (s0 is not None and s1 is not None and s0 <= s1)):
            start_frame = (np.asarray(f0[0], float), np.asarray(f0[1], float))
            which = "head"
        else:
            # f1을 쓰는 경우: 경로를 뒤집어서 f1이 시작 프레임이 되도록 처리
            start_frame = (np.asarray(f1[0], float), np.asarray(f1[1], float))
            which = "tail"

        return start_frame, which

    # --------- 내보내기 ---------
    def _export_scene(self, geom, out_file):
        """
        geom: Trimesh 또는 Scene
        GLB/GLTF는 Scene으로 감싸서 내보냄(호환성/가시성 개선),
        OBJ는 Trimesh 그대로 내보내도 무방.
        """
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        ext = os.path.splitext(out_file)[1].lower()
        if ext not in ('.obj', '.glb', '.gltf'):
            raise RuntimeError(f"지원하지 않는 포맷: {ext}")

        # 항상 Scene로 감싸서 내보냄 (OBJ도 병합하지 않음)
        if isinstance(geom, trimesh.Trimesh):
            scene = trimesh.Scene()
            scene.add_geometry(geom, node_name="mesh_0", geom_name="mesh_0")
        else:
            scene = geom

        # 주: trimesh는 OBJ로도 Scene을 내보낼 수 있으며, geometry별로 o/g 그룹이 생성됩니다.
        scene.export(out_file)
    
    
