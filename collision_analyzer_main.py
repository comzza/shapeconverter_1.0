# 1) 먼저 깊이 보정
fixer = shape_union_run(
    line_path="pipe_lines.shp",
    depth_path="depth_points.shp",
    avg_depth_field="AVG_DEPTH",
    pipe_size_field="PIPE_D",       # mm
    point_depth_field="P_DEPTH",    # m
    feature_id_field="FID",
    facility_type_field="FTYPE",
    snap_tol_m=0.3,
)
fixed_points = get_list_result(fixer)   # LineDepthFixer 결과

# 2) 충돌 분석 + 3D 출력
from collision_analyzer import run_collision_analysis

hits, files = run_collision_analysis(
    line_path="pipe_lines.shp",
    feature_id_field="FID",
    facility_type_field="FTYPE",
    pipe_size_field="PIPE_D",
    fixed_points=fixed_points,
    out_dir="./collisions_out",   # OBJ/CSV 출력 폴더
    export_models=True,
)

print(f"충돌 {len(hits)}건")
for f in files:
    print("생성:", f)
