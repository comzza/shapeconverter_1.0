from PySide6.QtCore import QObject, Signal, Slot
from collision_analyzer import CollisionAnalyzer

# ---- Worker 정의 ----
class CollisionAnalyzerWorker(QObject):
    finished = Signal(str)   # out_path
    error = Signal(str)      # error message
    progress = Signal(int)   # 0~100
    status = Signal(str)     # 상태 메시지

    def __init__(self, facil_type, line_shp, depth_shp, save_dir,
                 avg_depth_field, pipe_size_field, point_depth_field,
                 feature_id_field, layer_name):
        super().__init__()
        self.facil_type = facil_type
        self.line_shp = line_shp
        self.depth_shp = depth_shp
        self.save_dir = save_dir
        self.avg_depth_field = avg_depth_field
        self.pipe_size_field = pipe_size_field
        self.point_depth_field = point_depth_field
        self.feature_id_field = feature_id_field
        self.layer_name = layer_name

    @Slot()
    def run(self):
        try:
            # 0%: 초기화
            self.status.emit("초기화 중...")
            self.progress.emit(0)

            # 10%: Fixer 생성
            CollisionAnalyzer = CollisionAnalyzer(
                facility_type=self.facil_type,
                line_shp_file_path=self.line_shp,
                depth_shp_file_path=self.depth_shp,
                save_path=self.save_dir,
                avg_depth_field=self.avg_depth_field,
                pipe_size_field=self.pipe_size_field,
                point_depth_field=self.point_depth_field,
                feature_id_field=self.feature_id_field
            )
            self.progress.emit(10)

            # 10% -> 70%: 연산(형상 결합/심도 처리)
            self.status.emit("관로 line Feature와 심도 Point Feature의 결합 및 심도(depth) 보정 처리 중...")
            pass
            self.progress.emit(70)

            # 80%: 출력 파일 구성
            self.status.emit("출력 파일 구성 중...")
            pass
            self.progress.emit(80)

            # 80% -> 100%: GPKG 내보내기
            self.status.emit("GeoDataFrame을 GeoPackage 내보내기 중...")
            pass
            self.progress.emit(100)

            # 완료
            pass
            #self.finished.emit(output_file)

        except Exception as e:
            self.error.emit(str(e))