import os
import geojson
import re
import json
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import shape,mapping,\
                            LineString,Polygon,Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from dataclasses import dataclass
import numpy as np

from fit import recons_cluster
# from ..src.io.output import write_recons_bias_info

def write_recons_bias_info(target_file_path,
                    bias_info,
                    bias_file_name = 'veh2recons_bias_hau_distance.json'):
    output_json_path = os.path.join(target_file_path, bias_file_name)
    with open(output_json_path, 'w') as json_file:
        json.dump(bias_info, json_file, indent=4)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
def transform_geometry(geom: BaseGeometry):
    """
    将Shapely几何对象的坐标从WGS84 (EPSG:4326) 转换为 EPSG:32648
    """
    if geom.is_empty:
        return geom

    # 如果是 Polygon
    if isinstance(geom, Polygon):
        # 转换外部环
        exterior = [(transformer.transform(x, y)) for x, y, z in geom.exterior.coords]
        # 转换内部环
        interiors = [
            [(transformer.transform(x, y)) for x, y, z in interior.coords]
            for interior in geom.interiors
        ]
        return Polygon(exterior, interiors)
        # 如果是 LineString
    elif isinstance(geom, LineString):
        transformed_coords = [(transformer.transform(x, y)) for x, y, z in geom.coords]
        return LineString(transformed_coords)
    # 如果是 Point
    elif isinstance(geom, Point):
        # 只转换单个点的坐标
        x, y, z = geom.coords[0]
        transformed_coord = transformer.transform(x, y)
        return Point(transformed_coord)
    else:
        raise TypeError(f"不支持的几何类型: {type(geom)}")
    
class SliceMeasure:
    def __init__(self, name):
        self.name = name

class GlobalMeasure:
    def __init__(self, name):
        self.name = name

@dataclass
class SliceDataPackage:
    def __init__(self, 
                 folder_path):
        """
        初始化，读取文件夹中的所有数据。
        
        Args:
            folder_path (str): 文件夹路径，对应一个uuid命名的文件夹。
        """
        self.folder_path = folder_path
        self.reconstruction_file = None
        self.line_fit_file = None
        self.original_data_file = None
        self._load_files()
        
    def _load_files(self):
        """
        使用正则表达式加载文件夹中的文件，匹配特定的文件模式。
        """
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            
            # 正则匹配重建结果文件（uuid形式的geojson文件）
            if re.match(r'[0-9a-fA-F\-]{36}\.geojson$', file_name):
                self.reconstruction_file = self._load_geojson(file_path)
            
            # 正则匹配拟合结果文件（包含 'line_fit' 的文件）
            elif re.match(r'[0-9a-fA-F\-]{36}_line_fit\.geojson$', file_name):
                self.line_fit_file = self._load_geojson(file_path)
            
            # 正则匹配原始数据文件（以时间戳开头的 PE 文件）
            elif re.match(r'\d+_\d+_PE\.geojson$', file_name):
                self.original_data_file = self._load_geojson(file_path)      
    def _load_geojson(self, 
                      file_path):
        """
        读取 geojson 文件并解析为 Python 数据类型。
        
        Args:
            file_path (str): geojson 文件的路径。
            
        Returns:
            dict: geojson 文件的解析内容。
        """
        with open(file_path, 'r') as f:
            return geojson.load(f)
        
    @staticmethod
    def _measure_one_patch_to_a_slice_recons(patch, 
                                             recons_polygons):
        '''
        度量一个patch到一个slice的重建结果，
        patch指的是vehicles_polygons中的一个feature，
        recons指的是reconstruction_polygons，即如果几何多边形构成的列表

        在度量patch到recons的距离时，绘制出距离值分布
        '''
        total_distances_cls = {}
        # distances_original = []
        # 遍历每个重建结果多边形
        for _key,recons in recons_polygons.items():
            # 计算patch到每个recons的最小距离
            # distance = patch.distance(recons)
            distances_sum = 0
            distance = patch.hausdorff_distance(recons)
            total_distances_cls[_key] = distance
            # print(distance)
            # distances[_key] = distance
            # distances.append(distance)

            # distances_original.append(patch.distance(recons))

        '''
        # 绘制距离分布图
        fig1, ax1 = plt.subplots()
        ax1.hist(distances, bins=50, edgecolor='black')
        ax1.set_title('haus Distance Distribution from Patch to Reconstruction')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Frequency')

        fig2, ax2 = plt.subplots()
        ax2.hist(distances_original, bins=50, edgecolor='black')
        ax2.set_title('Distance Distribution from Patch to Reconstruction')
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Frequency')

        plt.show()
        # 返回最小距离
        '''
        min_key = min(total_distances_cls, key=lambda k: total_distances_cls[k])
        min_value = total_distances_cls[min_key]
        return min_key,min_value
    def compare_reconstruction_with_original(self):
        """
        比较重建结果与原始数据之间的距离，假设比较每个多边形的最近距离。
        
        Returns:
            list: 每个重建多边形与最近原始数据多边形之间的最短距离。
        """
        if not self.reconstruction_file or not self.original_data_file:
            raise ValueError("缺少必要的文件进行比较。")
        
        reconstruction_polygons,_ = self._extract_polygons(self.reconstruction_file)
        recons_set = recons_cluster(shapes = reconstruction_polygons,\
                    eps = 1.5,\
                    min_samples = 4)       
        
        _,vehicle_dict_origin = self._extract_polygons(self.original_data_file)
        # vehicles_polygons = [transform_geometry(geom) for geom in original_polygons]
        
        vehicles_polygons = {key: transform_geometry(geom) for key, geom in vehicle_dict_origin.items()}
        res = {}
        # distances = []
        for _key,orig_poly in vehicles_polygons.items():
            # min_distance = float('inf')
            _,min_distance_for_orig_poly2ref = self._measure_one_patch_to_a_slice_recons(orig_poly,
                                                      recons_set)
            '''
            for rec_poly in recons_set:
                distance = rec_poly.distance(orig_poly)
                if distance < min_distance:
                    min_distance = distance
            '''
            # distances.append(min_distance_for_orig_poly2ref)
            res[_key] = min_distance_for_orig_poly2ref
        # print(distances,vehicle_dict_origin)
        return res
    
    def _extract_polygons(self, 
                          geojson_data):
        """
        提取 geojson 数据中的所有多边形或曲线并转换为 Shapely 对象。
        
        Args:
            geojson_data (dict): geojson 类型字典。
            
        Returns:
            list: Shapely 多边形对象列表。
        """
        polygons = []
        feature_to_polygon = {}
        index_feature_to_polygon = 0
        for feature in geojson_data['features']:
            if feature.geometry.type == 'Polygon' and \
            1<len(feature.geometry.coordinates[0])<4: 
                exter_coords = feature.geometry.coordinates[0]
                new_geom = LineString(exter_coords)
                feature['geometry'] = mapping(new_geom)
                feature['geometry']['type'] = 'LineString'

            elif feature.geometry.type == 'Polygon' and \
            len(feature.geometry.coordinates[0])<=1: 
                exter_coords = feature.geometry.coordinates[0]
                new_geom = Point(exter_coords)
                feature['geometry'] = mapping(new_geom)
                feature['geometry']['type'] = 'Point'

            geom = shape(feature['geometry'])  # 使用 Shapely 将 geojson 转为几何对象
            polygons.append(geom)
            try:
                # 尝试获取 'oid' 的值
                oid_value = feature.properties['oid']
                # 检查 'oid' 是否是字符串
                if isinstance(oid_value, str):
                    feature_to_polygon[oid_value] = geom
                else:
                    # 如果 'oid' 不是字符串，根据需要处理
                    feature_to_polygon[index_feature_to_polygon] = geom
            except KeyError:
                # 如果 'oid' 键不存在，执行 pass
                feature_to_polygon[index_feature_to_polygon] = geom
                
            index_feature_to_polygon+=1
        return polygons,feature_to_polygon
    
# target_batch_dir = r"D:\BaiduSyncdisk\workspace\py\CrowdQC\reconstruction_output_full"
# target_batch_dir = r'E:\fit0925'
target_batch_dir = r'.\\reconstruction_output_temp\\'
count = 0
for _uuid in os.listdir(target_batch_dir):
    print(_uuid,count)
    count+=1
    # target_path = r'D:\BaiduSyncdisk\workspace\py\CrowdQC\reconstruction_output_full\000eeb74-eb41-42b1-bed8-17c555595005'
    # 初始化类
    target_path = os.path.join(target_batch_dir,_uuid)
    slice_data_package = SliceDataPackage(target_path)
    r2v_bias_info = slice_data_package.compare_reconstruction_with_original()
    write_recons_bias_info(target_file_path=target_path,
                    bias_info = r2v_bias_info)
input()