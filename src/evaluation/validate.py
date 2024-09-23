import shapely
import math
import numpy as np
import os
import json
import configparser
import pickle

from shapely import LineString,Polygon,Point
from shapely import wkt
from shapely.geometry import shape,box
from shapely import from_geojson

from .match_utils import get_featurecollection_extent
from .hdmap_item import *
# from ..config import HighDefinitionMapTileNameList,HighDefinitionMapItemName

HighDefinitionMapItemName = {
    "HADLane":1,
    "HADLaneAdas":2,
    "HADLaneDivider":3,
    "HADLanegroup":4,
    "HADLaneLimit":5,
    "HADLaneNode":6,
    "HADLaneZlevel":7,
    "HADLink":8,
    "HADNode":9,
    "HADOdd":10,
    "HADRoadDivider":11,
    "HADToll":12,
    "LandMark":13,
    "LocObj":14,
    "LocSign":15,
    "LocTrafficLight":16
}

HighDefinitionMapTileNameList = [
    '556162799', 
    '556162808', 
    '556162810', 
    '556162811', 
    '556168272', 
    '556168274', 
    '556168275', 
    '556168281', 
    '556168283', 
    '556168286', 
    '556168287', 
    '556168454', 
    '556168457', 
    '556168458', 
    '556168459', 
    '556168460', 
    '556168475', 
    '556168478', 
    '556168480', 
    '556168481', 
    '556168484', 
    '556168485', 
    '556168486',
    '556168487', 
    '556168493', 
    '556168496', 
    '556168497', 
    '556168498', 
    '556168499', 
    '556168500', 
    '556168501', 
    '556168502', 
    '556168503', 
    '556168504', 
    '556168505', 
    '556168506', 
    '556168507', 
    '556168544', 
    '556168545', 
    '556168546', 
    '556168547', 
    '556168548', 
    '556168550', 
    '556168551', 
    '556168556', 
    '556168557', 
    '556168592', 
    '556168593'
]

#此处应当设置你的高精地图真值瓦片数据所存储的目录
#   groundtruh 目录下应当是若干个地理编码命名的文件夹
GROUND_TRUTH_FEATURE_TILED_PATH = 'F:\\caq\\root_bak\\data\\_no_bias_ground_truth\\groundtruth'

class positionAccuracyAssessor:
    def __init__(self,
                 vec_path="",
                 config_file="config.ini",
                 id = True,
                 target_folder = 'output'):
        print("Assessor : positionAccuracyAssessor - 位置准确度比较类")
        # self.case = case_name
    
    def test_compare(self,target_path,ground_truth_root_path = GROUND_TRUTH_FEATURE_TILED_PATH,target_feature_type_list = HighDefinitionMapItemName):
        # _slice_to_gt_tile = self.match_current_feature_to_ground_truth(target_path,ground_truth_root_path)
        # _slice_to_gt_tile = self.match_current_feature_to_ground_truth_all(target_path,ground_truth_root_path)
        with open('semantic_slice_to_ground_truth.pkl', 'rb') as f:
            _slice_to_gt_tile = pickle.load(f)

        for _item_key in _slice_to_gt_tile.keys():
            # reference_file_path_list = [
            #     r'I:\caq_data\dataset\features\ground_truth\all_truth_trans.geojson'
            # ]

            #遍历存储slice与对应真值瓦片的字典
            reference_file_path_list = []
            _item_value = _slice_to_gt_tile[_item_key]
            for _item_file in _item_value:
                for target_feature_type in target_feature_type_list:
                    reference_file_path_list.append(os.path.join(ground_truth_root_path,_item_file,target_feature_type+'_trans.geojson'))
            target_file_path = os.path.join(target_path,_item_key)
            target_item,reference_item_list = self.get_items_for_comparation(target_file_path,reference_file_path_list)
            if reference_item_list!=[]:
                _positional_error_in_current_slice  = self.compare_between_all_feautures(target_item,reference_item_list)
                #为输出的字典赋值，改字典将geojson逐个对应到绝对位置误差
                # _semantic_level_positional_error_per_slice[_item_key] = _positional_error_in_current_slice
            else:
                #如果reference_item_list为空，表示当前车端数据没有对应真值
                _positional_error_in_current_slice = 'No_reference_ground_truth'
            _target_geojson = os.path.join(target_path,_item_key)
            self.write_positional_error_to_all_sementic_file(_target_geojson,_positional_error_in_current_slice)
            
    def write_positional_error_to_all_sementic_file(self,target_path,positional_error_array) -> bool:
        '''
        此函数将每个slice的绝对位置误差赋值到semantic中
        '''
        # positional_error_list = _dict_to_PE_value.values()

        _index_ = 0
        with open(target_path, 'r') as geojson_file:
            geojson_data = json.load(geojson_file)
        for feature in geojson_data['features']:
            # 获取要素的几何信息
            geometry = feature['geometry']
            if geometry != None and positional_error_array != 'No_reference_ground_truth':
                feature["properties"]["positional_error"] = positional_error_array[_index_]
            else:
                feature["properties"]["positional_error"] = positional_error_array
            _index_ += 1

        output_path = target_path.replace('.geojson','_PE.geojson').replace('VecJsonData_noH','VecPE_new')
        _dir = os.path.split(output_path)[0]
        if os.path.exists(_dir) == False:
            os.makedirs(_dir)
        with open(output_path, 'w') as output_file:
            json.dump(geojson_data, output_file, indent=2)
        return True
        # 返回是否输出成功 
    def get_items_for_comparation(self,target_file_path,reference_file_path_list):
        '''
        Parameters
        -----------
        self: positionAccuracyAssessor
        位置准确度分析类
        target_file_path: path
        目标要素完整路径
        refer_file_path_list: list
        参考要素完整路径组成的列表

        Returns
        --------
        target_item : hd_item
        目标要素
        reference_item_list : list
        参考要素列表（hd_item组成的列表）

        Description
        --------                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        '''
        _time_slice = os.path.split(target_file_path)[1]
        _time_slice = _time_slice.split('.')[0]
        target_item = VehiclesData(target_file_path,_time_slice)

        reference_item_list = []
        for reference_file_path in reference_file_path_list:
            if HdmapData(reference_file_path)!=None:
                reference_item_list.append(HdmapData(reference_file_path))                                                                                                                                                                                                                                                                                                                                                                                                                                          

        return target_item,reference_item_list
    def compare_between_all_feautures(self, target_item, reference_item_list):
        ''' 
        本函数计算一个slice内部的所有向量之间的
        此函数输入为本工程定义的类 hd_item1，hd_item2中的data数据为gdal的DataSource对象

        Parameters
        -----------
        target_item: HdmapMade
            云端制图的数据
        reference_item_list: List 
            存储真值的对象(HdmapTruth)组成的列表，作为参考值

        Returns
        --------
        float:
        absolute_positional_error
            绝对位置误差,单位为米(m)
            The distance between the geometries or -1 if an error occurs

        '''
        absolute_positional_error = np.empty(0)

        # ds2_list = []
        # for ds_r in reference_item_list:
        #     ds2_list.append(ds_r.data)

        from src import coord_trans
        #定义transform对象

        # driver = ogr.GetDriverByName("GeoJSON")
        # target_ds = driver.CreateDataSource("tmp.geojson", options=["RFC7946=YES"])
        # target_layer = target_ds.CreateLayer("transformed_features", geom_type=ogr.wkbLineString25D)  # 注意这里使用了25D几何类型

        # for feature in target_layer:
        # ds1.Transform(transform)
        # ds2.Transform(transform)
        
        #读取原始目标数据集
        #转换target数据到utm坐标系
        _trans_reference_geometry_list =[]
        _trans_target_geometry = coord_trans.transform_coordinates(target_item.data, 32648)
        #转换target数据到utm坐标系
        for reference_item in reference_item_list:
            #将转换后的datasource对象存储到_trans_reference_geometry_list列表中
            _trans_reference_geometry = coord_trans.transform_coordinates(reference_item.data, 32648)
            _trans_reference_geometry_list.append(_trans_reference_geometry)
        for layer1 in _trans_target_geometry:
            for feature1 in layer1:
                geometry1 = feature1.GetGeometryRef()
                name1 = feature1.GetField("oid")
                # if name1 == '51':
                #     print('ok')
                ts = feature1.GetField("start_time")

                nearest_distance = float("inf")
                nearest_name = ""
                for _trans_reference_geometry in _trans_reference_geometry_list:
                    for layer2 in _trans_reference_geometry:
                        for feature2 in layer2:
                            geometry2 = feature2.GetGeometryRef()
                            if geometry2.GetArea()>20:
                            #此时为较大的道路停止线等要素，通常位于路口，此时应当跳过
                                continue
                            name2 = feature2.GetField("id")
                            # if name2 == '9547631416740050278' or name2 == '9583660961083323750':
                            #     print("target")
                            '''
                            此处计算两个要素的绝对位置误差
                            '''
                            # #1 最短距离
                            distance = geometry1.Distance(geometry2) 
                            # #2 hausdorff_distance距离
                            # distance = shapely.hausdorff_distance(self.ogr_geometry_to_shapely_linestring(geometry1), self.ogr_geometry_to_shapely_linestring(geometry2))
                            # distance = shapely.hausdorff_distance(self.ogr_geometry_to_shapely_new(geometry1), self.ogr_geometry_to_shapely_new(geometry2))
                            
                            # distance = shapely.hausdorff_distance(geometry1, geometry2)
                            #3 frechet_distance距离
                            # distance = shapely.frechet_distance(geometry1, geometry2)
                            if distance < nearest_distance:# and self.are_features_parallel(geometry1,geometry2):
                                nearest_distance = distance
                                nearest_name = name2
                                
                                # absolute_positional_error.append(nearest_distance)
                print(f"--Feature '{name1}' in {ts} is closest to '{nearest_name}' in ds2 with distance {nearest_distance}",end = "\r")
                absolute_positional_error = np.append(absolute_positional_error,nearest_distance)
        return absolute_positional_error
    def match_current_feature_to_ground_truth_all(self,target_geojson_path,original_path = GROUND_TRUTH_FEATURE_TILED_PATH)->dict:
        _dict_semantic_slice_to_ground_truth = {}
        for vec_item in os.listdir(target_geojson_path):
            _dict_semantic_slice_to_ground_truth[vec_item] = HighDefinitionMapTileNameList
        return _dict_semantic_slice_to_ground_truth
    def match_current_feature_to_ground_truth(self,target_geojson_path,original_path = GROUND_TRUTH_FEATURE_TILED_PATH)->dict:
        '''
        本函数实现的功能为：输入任何一个车端矢量要素，判断其类型，首先获取其最近的tile_id,
        Parameters
        -----------
        ground_truth_path: geojson_path
        真值数据

        Returns
        --------
        _tile_to_extend_dict：dict
        返回存储距离当前目标要素最近的要素类型（id）
        '''
        
        # 首先读取真值元数据，每个真值瓦片的id与extend的对应关系
        path_parts = os.path.split(original_path)
        meta_data_name = 'ground_truth_tiles.geojson'
        
        ground_truth_root = path_parts[0]
        meta_data_full_path = os.path.join(ground_truth_root,meta_data_name)
        with open(meta_data_full_path,'r') as geojson_item:
            geojson_data = json.load(geojson_item)

        _dict_semantic_slice_to_ground_truth = {}
        #遍历文件夹内文件
        slice_num = len(os.listdir(target_geojson_path))
        count = 0
        for vec_item in os.listdir(target_geojson_path):
            print(f'{count} / {slice_num}')
            count += 1
            #返回一个列表，存储相关联的瓦片，首先置为空值
            _refered_tiles = []
            vec_item_full_path = os.path.join(target_geojson_path,vec_item)
            
            #获取当前帧的extent
            _current_fc_extent = get_featurecollection_extent(vec_item_full_path)
            
            #将_current_fc_extent转换为shapely的box类型
            _single_extent_geometry = box(_current_fc_extent[0], _current_fc_extent[1], _current_fc_extent[2], _current_fc_extent[3])

            #使用
            _keys_name_list = list(geojson_data.keys())
            if "type" in _keys_name_list and geojson_data['type'] == "FeatureCollection":
                #遍历真值瓦片
                for item_feature in geojson_data["features"]:
                    if item_feature["geometry"]!={} and item_feature["geometry"]!= None:
                        _current_ground_truth_geometry = shape(item_feature["geometry"])

                        #判断当前fc的extent与瓦片是否相交
                        if _single_extent_geometry.intersects(_current_ground_truth_geometry):
                            _refered_tiles.append(item_feature['properties']['name'])
                # print('Features of \'',vec_item,'\'are spatially related to :',_refered_tiles,'              ',end = '\r')
                _dict_semantic_slice_to_ground_truth[vec_item] = _refered_tiles

        return _dict_semantic_slice_to_ground_truth
    def are_features_parallel(self,line1, line2, tolerance=5):#
        # 获取线1的起点和终点坐标
        x1_start, y1_start = line1.GetPoints()[0][0],line1.GetPoints()[0][1]
        x1_end, y1_end = line1.GetPoints()[-1][0],line1.GetPoints()[-1][1]
        
        # 获取线2的起点和终点坐标
        x2_start, y2_start = line2.GetPoints()[0][0],line2.GetPoints()[0][1]
        x2_end, y2_end = line2.GetPoints()[-1][0],line2.GetPoints()[-1][1]
        
        # 计算线1的方向向量
        vector1 = (x1_end - x1_start, y1_end - y1_start)
        # 计算线2的方向向量
        vector2 = (x2_end - x2_start, y2_end - y2_start)
        
        # 计算方向向量的差异
        # a*b = |a||b|cosθ
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        vector1_length = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
        vector2_length = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5
        try:
            cos_angle = dot_product / (vector1_length * vector2_length)
        except:
            return False
        angle_diff = abs(180.0 * math.acos(cos_angle) / math.pi)

        # 判断是否基本平行
        if angle_diff < tolerance or abs(angle_diff - 180.0) < tolerance:
            return True
        else:
            return False

    def ogr_geometry_to_shapely_new(self,ogr_geometry):
        # Get geometry type
        global rings
        geom_type = ogr_geometry.GetGeometryName()
        # print("Geometry Type:", geom_type)

        # Get coordinates and convert to shapely
        if geom_type == "POINT":
            # print("geom_type:POINT", "Point Coordinates: X:", x, "Y:", y)
            x = ogr_geometry.GetX()
            y = ogr_geometry.GetY()
            coords = (x, y)
            return Point(coords)

        elif geom_type == "LINESTRING":
            num_points = ogr_geometry.GetPointCount()
            coords = []
            for i in range(num_points):
                x, y, z = ogr_geometry.GetPoint(i)
                # print("geom_type:LINESTRING", "Point", i + 1, "Coordinates: X:", x, "Y:", y)
                coords.append((x, y))
            return LineString(coords)

        elif geom_type == "POLYGON":
            rings = []
            num_rings = ogr_geometry.GetGeometryCount()
            for ring in range(num_rings):
                ring_geom = ogr_geometry.GetGeometryRef(ring)
                # print(f"Ring {ring + 1} Points:", ring_geom.GetPointCount())
                for i in range(ring_geom.GetPointCount()):
                    x, y = ring_geom.GetX(i), ring_geom.GetY(i)
                    # print(f"Point {i + 1}: ({x}, {y})")
                    rings.append((x, y))
            return Polygon(rings)
        
    def ogr_geometry_to_shapely_linestring(self,ogr_geometry):
        coords = []
        for i in range(ogr_geometry.GetPointCount()):
            x, y, _ = ogr_geometry.GetPoint(i)
            coords.append((x, y))
        return LineString(coords)
    def compare_between_lines_batch(self, hd_item1, hd_item2):
        ''' 
        此函数输入为本工程定义的类 hd_item1，hd_item2中的data数据为gdal的DataSource对象

        Parameters
        -----------
        hd_item1: HdmapMade
            云端制图的数据
        hd_item2: HdmapTruth
            存储真值的对象，用于参考
            

        Returns
        --------
        float:
        absolute_positional_error
            绝对位置误差，单位为米（m）
            The distance between the geometries or -1 if an error occurs

        '''
        
        absolute_positional_error = np.empty(0)

        ds1 = hd_item1.data
        ds2 = hd_item2.data

        from src import coord_trans
        #定义transform对象

        # driver = ogr.GetDriverByName("GeoJSON")
        # target_ds = driver.CreateDataSource("tmp.geojson", options=["RFC7946=YES"])
        # target_layer = target_ds.CreateLayer("transformed_features", geom_type=ogr.wkbLineString25D)  # 注意这里使用了25D几何类型

        # for feature in target_layer:
        # ds1.Transform(transform)
        # ds2.Transform(transform)

        _trans_geometry_ds1 = coord_trans.transform_coordinates(ds1, 32648)
        _trans_geometry_ds2 = coord_trans.transform_coordinates(ds2, 32648)

        for layer1 in _trans_geometry_ds1:
            for feature1 in layer1:
                geometry1 = feature1.GetGeometryRef()
                name1 = feature1.GetField("id")

                nearest_distance = float("inf")
                nearest_name = ""

                for layer2 in _trans_geometry_ds2:
                    for feature2 in layer2:
                        geometry2 = feature2.GetGeometryRef()
                        name2 = feature2.GetField("id")
                        '''
                        此处计算两个要素的绝对位置误差
                        '''

                        # #1 最短距离
                        # distance = geometry1.Distance(geometry2) 
                        # #2 hausdorff_distance距离
                        # distance = shapely.hausdorff_distance(self.ogr_geometry_to_shapely_linestring(geometry1), self.ogr_geometry_to_shapely_linestring(geometry2))
                        distance = shapely.hausdorff_distance(self.ogr_geometry_to_shapely_new(geometry1), self.ogr_geometry_to_shapely_new(geometry2))
                        
                        # distance = shapely.hausdorff_distance(geometry1, geometry2)
                        #3 frechet_distance距离
                        # distance = shapely.frechet_distance(geometry1, geometry2)
                        if distance < nearest_distance and self.are_features_parallel(geometry1,geometry2):
                            nearest_distance = distance
                            # print(geometry2)
                            nearest_name = name2
                            
                            # absolute_positional_error.append(nearest_distance)
                print(f"Feature '{name1}' in ds1 is closest to '{nearest_name}' in ds2 with distance {nearest_distance}",end = "\r")
                absolute_positional_error = np.append(absolute_positional_error,nearest_distance)
        return absolute_positional_error

    def compare_between_lines(self, hd_item1, hd_item2):
        ''' 
        此函数输入为本工程定义的类 hd_item，hd_item中的data数据为gdal的DataSource对象
        '''
        ds1 = hd_item1.data
        ds2 = hd_item2.data

        # 获取第一个DataSource的第一个图层
        layer1 = ds1.GetLayerByIndex(0)
        
        # 获取第二个DataSource的第一个图层
        layer2 = ds2.GetLayerByIndex(0)
        
        # 获取第一个图层的第一个要素
        feature1 = layer1.GetFeature(0)
        
        # 获取第一个要素的几何对象
        geometry1 = feature1.GetGeometryRef()
        
        # 获取第二个图层的第一个要素
        feature2 = layer2.GetFeature(0)
        
        # 获取第二个要素的几何对象
        geometry2 = feature2.GetGeometryRef()
        
        # 计算两个几何对象的距离
        distance = geometry1.Distance(geometry2)
        
        return distance
        #验证要素类型是否正确

if __name__ == 'src.evaluation.validate':
    _file_path = 'examples\\config.ini'
    config = configparser.ConfigParser()
    config.read(_file_path,encoding = 'UTF8')

    # read paths in config file
    dataset_path_root = config.get('Paths','dataset_path_root')
    ground_truth_tile_folder = config.get('Paths','ground_truth_tile_folder')
    vehicle_feature_slice_folder_l35 = config.get('Paths','vehicle_feature_slice_folder_l35')
    location_folder = config.get('Paths','location_folder')
    vision_folder = config.get('Paths','vision_folder')
    target_path = config.get('Paths','target_path')
    
    hdmap_truth_path = os.path.join(dataset_path_root,ground_truth_tile_folder) 

    print(target_path)
    # print(_)
    # _temp_hdmap_truth_path = os.path.join(sample_data_path,"samples\\ground_truth\\had_lane_divider_trans.geojson")
    # _path = os.path.join(sample_data_path,"samples\\ground_truth\\had_lane_divider_trans.geojson")
    # emp_hdmap_truth = hdmap_truth.HdmapTruth(_temp_hdmap_truth_path,0)
    # root = os.getcwd()
    # target_path = r'H:\l35\2023-07-25\LS6A2E161NA505442_L35\VecJsonData_noH'
    PA_Assessor = positionAccuracyAssessor()
    PA_Assessor.test_compare(target_path,hdmap_truth_path)