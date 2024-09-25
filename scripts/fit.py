from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import json
import numpy as np
import json
import os
import re

def cluster_and_fit_geojson(input_file, \
                            output_file, \
                            eps, \
                            min_samples):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = []
    for feature in data['features']:
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            coordinates = geometry['coordinates']
            for ring in coordinates:
                points.extend(ring)
        elif geometry['type'] == 'Point':
            points.append(geometry['coordinates'])

    points = np.array(points)
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    points = np.hstack((x, y.reshape(-1, 1)))

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # # 绘制聚类结果
    # plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='rainbow', label='Data Points')
    # plt.title("DBSCAN Clustering")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # 存储拟合结果
    fits = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            cluster_points = points[labels == label]
            x_cluster = cluster_points[:, 0]
            y_cluster = cluster_points[:, 1]

            # 三次拟合
            linear_model = np.polyfit(x_cluster, y_cluster, 2)
            linear_model_fn = np.poly1d(linear_model)

            # 生成拟合曲线的 x 值
            x_fit = np.linspace(min(x_cluster), max(x_cluster), 100)
            y_fit = linear_model_fn(x_fit)

            fits.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': list(zip(x_fit, y_fit))
                },
                'properties': {
                    'cluster_id': int(label)
                }
            })

    # 写入 GeoJSON 文件
    geojson_data = {
        'type': 'FeatureCollection',
        'features': fits
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f)
'''
temp_path = r'..\\reconstruction_output_0923\fe46c83c-fc96-41ab-a62c-1f07881abd5b.geojson'
cluster_and_fit_geojson(temp_path,
                        temp_path.replace('.geojson','_clustered.geojson'),
                        eps=2.5, 
                        min_samples=4)
'''

path_r = ".\\reconstruction_output_0923"
pattern = os.path.join(path_r, '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}.geojson')
pattern = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\.geojson$')

matching_files = []
for file in os.listdir(path_r):
    if pattern.match(file):  # 如果文件名匹配正则表达式
        _file_path = os.path.join(path_r,file)
        cluster_and_fit_geojson(_file_path, 
                            _file_path.replace('.geojson','_line_fit.geojson'), 
                            eps=2.5, 
                            min_samples=4)
        print(_file_path)
        # matching_files.append(os.path.join(path_r,file))
input()
# for item in os.listdir(path_r):
#     target_path = os.path.join(target_path,item)
#     cluster_and_fit_geojson(target_path, 
#                             target_path.replace('.geojson','_line_fit.geojson'), 
#                             eps=2.5, 
#                             min_samples=4)