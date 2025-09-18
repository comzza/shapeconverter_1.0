# 필요한 라이브러리 import
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt

# 1. GeoPackage 파일 경로 지정
# 이전에 저장했던 GeoPackage 파일의 실제 경로를 입력합니다.
gpkg_path = 'pipe_depth_data.gpkg'

# 2. 파일 불러오기
try:
    # read_file 함수로 GeoPackage 파일을 읽어 GeoDataFrame으로 로드합니다.
    gdf = gpd.read_file(gpkg_path)

    # 3. 불러온 데이터 확인
    print("✅ 데이터 불러오기 성공!")
    print("\n--- 데이터 샘플 (상위 5개) ---")
    print(gdf.head())

    print("\n\n--- 데이터 기본 정보 ---")
    gdf.info()

    print(f"\n\n--- 좌표계(CRS) 정보 ---")
    print(gdf.crs)

    # 4. 데이터 활용 (간단한 시각화)
    print("\n\n--- 데이터 시각화 중... ---")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(ax=ax, 
             markersize=5, 
             legend=True,
             cmap='viridis', # Z값에 따라 색상을 다르게 표현
             column='z'
            )
    ax.set_title('Loaded Pipe Depth Data')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
    print("✔️ 시각화 완료.")

except FileNotFoundError:
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {gpkg_path}")
except Exception as e:
    print(f"❌ 오류: 데이터를 불러오는 중 문제가 발생했습니다: {e}")

# 5. 추가 분석 예시
# 특정 조건에 맞는 데이터 필터링, 예를 들어
# 관경(diam)이 500mm 이상인 데이터만 필터링
large_pipes = gdf[gdf['diam'] >= 500]
print(large_pipes)

#공간 분석: 각 점 주변으로 버퍼(buffer)를 생성하는 등 간단한 공간 연산이 가능합니다.
# 각 포인트를 중심으로 10미터 반경의 버퍼 생성 (좌표계가 미터 단위일 경우)
gdf['buffer_geometry'] = gdf.geometry.buffer(10)
print(gdf.head())

# 버퍼를 시각화하여 확인
gdf['buffer_geometry'].plot()
plt.show()
