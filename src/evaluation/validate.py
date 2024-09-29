import shapely
import math
import numpy as np
import os
import json
import re
import copy

from shapely import LineString,Polygon,Point
from shapely import wkt
from shapely.geometry import shape,box
from shapely import from_geojson

from .match_utils import get_featurecollection_extent,\
    get_recons_feature_extent,\
    transform_coordinates

from .hdmap_item import *
# from ..config import HighDefinitionMapTileNameList,HighDefinitionMapItemName
Target_HighDefinitionMapItemName = {
    "HADLaneDivider":3,
    "HADRoadDivider":11,
    "LandMark":13,
}
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

class positionAccuracyAssessor:
    def __init__(self,
                 config_file="config.ini"):
        self.read_config(config_file)

    def read_config(self,_path):
        self.gt_tile_path,self.gt_feature_path = input.read_config_file_valid(_path)

    @staticmethod
    def update_index_project(index_project:dict,
                            reference_pack:dict):
        # 遍历 reference_pack 的 key
        for key, value in reference_pack.items():
            # 如果 key 已经在 index_project 中，跳过
            if key in index_project:
                continue
            index_project[key] = value
            # 否则，需要添加新的键值对，但首先对第三级进行处理
            # 使用 deepcopy 保证不修改 reference_pack 的原始数据
            '''
            new_value = copy.copy(value)

            # 遍历第三级，删除 hd_item 键值对
            for sub_key in new_value:
                if 'hd_item' in new_value[sub_key]:
                    del new_value[sub_key]['hd_item']
                    # 将修改后的键值对添加到 index_project 中
            index_project[key] = new_value
            '''
        # 返回更新后的 index_project
        return index_project

    def batch_compare(self,
                      target_path=".",
                      target_feature_type_list = Target_HighDefinitionMapItemName,
                      if_index = False,
                      target_type = 'slice'):
        '''
        # 此处的batch_compare函数用于批量比较目标路径下的所有slice或三角化重建结果与真值tile之间的位置误差，并将结果写入到semantic_slice_to_ground_truth.pkl文件
        # 支持两种输入，即target_path为slice的路径，或者target_path为三角化重建结果的路径
        '''
        
        # 获取slice和真值tile的映射关系，建立初步的粗索引
        if if_index == False:
            if target_type == 'slice':
                _slice_to_gt_tile = self.__match_current_feature_to_ground_truth(target_path)
            elif target_type == 'recons':
                _slice_to_gt_tile = self.__match_current_feature_to_ground_truth(target_path,
                                                                             mode='recons')
        else:
            _slice_to_gt_tile = input.read_semantic_slice_to_ground_truth_dict('semantic_slice_to_ground_truth.pkl')

        index_project = {}
        count = 0
        for _item_key in _slice_to_gt_tile.keys():
            print(f'{count} / {len(_slice_to_gt_tile)}')
            count += 1
            # reference_file_path_list = [
            #     r'I:\caq_data\dataset\features\ground_truth\all_truth_trans.geojson'
            # ]
            #遍历存储slice与对应真值瓦片的字典
            reference_file_path_list = []
            _item_value = _slice_to_gt_tile[_item_key]

            if _item_value == []:
                continue
            for _item_file in _item_value:
                #遍历所有对应的真值tile 
                # （一般的情况下是1个，极端情况会有2个，即当前的要素位于两个tile交界处）
                for target_feature_type in target_feature_type_list:
                    reference_file_path_list.append(os.path.join(self.gt_feature_path,
                                                                 _item_file,
                                                                 target_feature_type+'_trans.geojson'))
            # target_file_path = os.path.join(target_path,_item_key)
            target_file_path = _item_key
            
            def __are_all_elements_in_a_contained_in_b(a, b):
                return all(item in b for item in a)
            
            if_index_build = __are_all_elements_in_a_contained_in_b(_item_value, index_project.keys())
            # 返回当前瓦片中对应的全部地图要素
            if reference_file_path_list!=[]:
                target_item,reference_pack = self.__get_items_for_comparison(target_file_path,
                                                                            reference_file_path_list,
                                                                            index_project,
                                                                            if_index_build)
                
                if reference_pack!={} and target_item.data!=None:
                    target_tiles = _slice_to_gt_tile[target_file_path]
                    _positional_error_in_current_slice  = self.compare_vec2gt(target_item,
                                                                            reference_pack,
                                                                            index_project,
                                                                            mode=target_type,
                                                                            target_tiles=target_tiles)
                    
                    output.write_bias_info(target_file_path,
                                    bias_info = _positional_error_in_current_slice,
                                    mode = target_type)
                    

                    # if target_type == 'slice':
                    #     _positional_error_in_current_slice  = self.compare_vec2gt(target_item,
                    #                                                         reference_pack,
                    #                                                         index_project)
                    # elif target_type == 'recons':
                    #     _positional_error_in_current_slice  = self.compare_vec2gt(target_item,
                    #                                                         reference_pack,
                    #                                                         index_project,
                    #                                                         mode=target_type)
                    
                else:
                    #如果reference_item_list为空，表示当前车端数据没有对应真值
                    _positional_error_in_current_slice = 'No_reference_ground_truth'
                # for _k,_v in reference_pack.items():
                print(_positional_error_in_current_slice)
                if reference_pack!='index_build':
                    index_project = positionAccuracyAssessor.update_index_project(index_project,
                                                                                reference_pack,                                                                               )
        return

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
    def __get_items_for_comparison(self,
                                   target_file_path,
                                   reference_file_path_list,
                                   index_project,
                                   if_index_build):
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
        if if_index_build == False:
            ref_items = {}
            # ref_item_index = 0
            for reference_file_path in reference_file_path_list:
                if HdmapData(reference_file_path)!=None:
                    def __extract_tile_id_from_path(path):
                        # 使用os.path.split来分割路径
                        head, tail = os.path.split(path)
                        # 继续分割直到找到包含数字的目录
                        while not re.match(r'\d+', tail):
                            head, tail = os.path.split(head)
                        # 返回匹配的数字部分，即瓦片ID
                        return tail
                    
                    def __extract_hd_item_id_from_path(path):
                        # 使用os.path.split来分割路径
                        base_name = os.path.basename(path)
                        hd_item_name = base_name.split('.')[0]
                        # 返回匹配的数字部分，即瓦片ID
                        return hd_item_name
                    tile_id = __extract_tile_id_from_path(reference_file_path)
                    if tile_id not in ref_items.keys():
                        ref_items[tile_id] = {}

                    hd_item_name = __extract_hd_item_id_from_path(reference_file_path)
                    if hd_item_name not in ref_items[tile_id].keys():
                        # print("New High_Definition Item")
                        temp_ref_item = {}

                        # 某一tile内指定类型的空间索引
                        temp_hd_item = HdmapData(reference_file_path)
                        if temp_hd_item.data.GetLayerCount()==0:
                            continue
                        temp_spatial_index = temp_hd_item.build_si()

                        temp_ref_item['hd_item'] = temp_hd_item
                        temp_ref_item['spatial_index'] = temp_spatial_index

                        ref_items[tile_id][hd_item_name] = temp_ref_item
                        # ref_items[tile_id] = temp_ref_item
                    # ref_item_index += 1

                    # reference_item_list.append(HdmapData(reference_file_path))                                                                                                                                                                                                                                                                                                                                                                                                                                          
            return target_item,ref_items
        else:
            return target_item,"index_build"
    
    def __query_intersecting_features(geojson_feature,layer):
        # 将GeoJSON转换为ogr.Geometry对象
        geom = ogr.CreateGeometryFromJson(geojson_feature)
        envelope = geom.GetEnvelope()
        possible_matches = list(index.intersection(envelope))
        actual_matches = []
        for fid in possible_matches:
            feature = layer.GetFeature(fid)
            feature_geom = feature.GetGeometryRef()
            if geom.Intersects(feature_geom):
                actual_matches.append((fid, feature_geom))
        return actual_matches

# 函数：查找最近的要素
    def __query_nearest_feature(geojson_feature,layer,
                                buffer = 10):
        geom = ogr.CreateGeometryFromJson(geojson_feature)
        envelope = geom.GetEnvelope()

        # 扩展边界框以增加搜索范围
        search_envelope = (envelope[0] - buffer, \
                           envelope[2] + buffer, \
                           envelope[1] - buffer, \
                           envelope[3] + buffer)
        
        possible_matches = list(index.intersection(search_envelope))
        nearest_feature = None
        min_distance = float('inf')
        for fid in possible_matches:
            feature = layer.GetFeature(fid)
            feature_geom = feature.GetGeometryRef()
            distance = geom.Distance(feature_geom)
            if distance < min_distance:
                min_distance = distance
                nearest_feature = (fid, feature_geom)
        return nearest_feature
    @staticmethod
    def find_nearest_feature(_trans_target_slice, 
                             trans_reference_geometry_list, 
                             min_area=20,
                             index_project = {},
                             mode="slice",
                             target_tiles=[]):
        """
        寻找给定要素最近的要素。

        :param layer1: 第一个图层的要素集合
        :param trans_reference_geometry_list: 包含其他图层的要素集合的列表
        :param field_name1: 第一个要素集合中要素的字段名称
        :param field_time1: 第一个要素集合中要素的时间字段名称
        :param min_area: 忽略的要素的最小面积
        :return: None
        """
        ans_dict = {}
        for layer1 in _trans_target_slice:
            for feature1 in layer1:
                # 遍历第一个图层的各个要素
                geometry1 = feature1.GetGeometryRef()
                if mode == 'recons':
                    name1 = feature1.GetField("cluster_id")
                    ts = 'temp'
                elif mode=='slice':
                    name1 = feature1.GetField("oid")
                    ts = feature1.GetField("start_time")

                nearest_distance = float("inf")
                nearest_name = ""
                if index_project == {}:
                    #此时从候选的trans_reference_geometry_list遍历搜索
                    for layer2_list in trans_reference_geometry_list:
                        for layer2 in layer2_list:
                            for feature2 in layer2:
                                geometry2 = feature2.GetGeometryRef()
                                if geometry2.GetArea() > min_area:
                                    # 跳过面积大于min_area的要素
                                    continue
                                
                                distance = geometry1.Distance(geometry2)
                                if distance < nearest_distance:
                                    name2 = feature2.GetField("id")
                                    nearest_distance = distance
                                    nearest_name = name2
                else:
                    envelope = geometry1.GetEnvelope()  # 获取边界框
                    selected_index_project = {}
                    for tile_id in target_tiles:
                        # index_list.append(index_project[tile_id][])
                        selected_index_project[tile_id] = index_project[tile_id]

                    
                    #此时直接根据索引进行搜索
                    target_refs = []
                    for _idx_key,_idx_value in selected_index_project.items():
                    # for _idx_value in index_list:
                        for _hd_item_type_key,_hd_value in _idx_value.items():
                            for _key,_value_geomnindex in _hd_value.items():
                                if _key == 'spatial_index':
                                    #该图层为空间索引
                                    intersection_geom_ids = list(_value_geomnindex.intersection(envelope))
                                    # print(intersection_geom_ids)
                                    if len(intersection_geom_ids)>0:
                                        _trans_target_slice = transform_coordinates(_hd_value['hd_item'].data, 32648)
                                        target_refs = _hd_value['hd_item'].get_items_by_id(intersection_geom_ids)
                                        
                                        #此时遍历列表intersection_geom_ids，读取每个要素中的
                                        # print(target_refs)

                                        for _key,feature2 in target_refs.items():
                                            geometry2 = feature2.GetGeometryRef()
                                            if geometry2.GetArea() > min_area:
                                                # 跳过面积大于min_area的要素
                                                continue
                                            
                                            distance = geometry1.Distance(geometry2)
                                            if distance < nearest_distance:
                                                name2 = feature2.GetField("id")
                                                nearest_distance = distance
                                                nearest_name = name2

                print(f"--Feature '{name1}' in {ts} is closest to '{nearest_name}' in gt with distance {nearest_distance}", end = "\r")
                ans_dict[name1] = nearest_distance
                # absolute_positional_error = np.append(absolute_positional_error, nearest_distance)
        return ans_dict
    def compare_vec2gt(self, 
                    target_item, 
                    reference_items,
                    index_project,
                    mode='slice',
                    target_tiles = []):
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
        _trans_reference_geometry_list =[]
        if mode=='slice':
            _trans_target_slice = transform_coordinates(target_item.data, 32648)
        elif mode=='recons':
            _trans_target_slice = target_item.data

        #转换target数据到utm坐标系
        if reference_items !='index_build':
            for reference_tile_id,reference_item in reference_items.items():
                #将转换后的datasource对象存储到_trans_reference_geometry_list列表中
                for hd_item_name,ref_pack in reference_item.items():
                    _trans_reference_geometry = transform_coordinates(ref_pack['hd_item'].data, 32648)
                    _trans_reference_geometry_list.append(_trans_reference_geometry)
                    absolute_positional_error = positionAccuracyAssessor.find_nearest_feature(_trans_target_slice, 
                                                                                _trans_reference_geometry_list, 
                                                                                min_area=20,
                                                                                mode=mode) #面积特别大的要素直接跳过
        else:
            # 此时已经建立了索引
            absolute_positional_error = positionAccuracyAssessor.find_nearest_feature(_trans_target_slice, 
                                                                                _trans_reference_geometry_list, 
                                                                                min_area=20,
                                                                                index_project=index_project,
                                                                                mode=mode,
                                                                                target_tiles=target_tiles) 


        return absolute_positional_error
    
    def __match_current_feature_to_ground_truth_all(self,
                                                    target_geojson_path)->dict:
        original_path = self.gt_feature_path
        _dict_semantic_slice_to_ground_truth = {}
        for vec_item in os.listdir(target_geojson_path):
            _dict_semantic_slice_to_ground_truth[vec_item] = HighDefinitionMapTileNameList
        return _dict_semantic_slice_to_ground_truth
    


    def __match_current_feature_to_ground_truth(self,
                                                target_geojson_path,
                                                mode = 'vehicle_slice')->dict:
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
        original_path = self.gt_feature_path
        path_parts = os.path.split(original_path)
        meta_data_name = 'ground_truth_tiles.geojson'
        
        ground_truth_root = path_parts[0]
        meta_data_full_path = os.path.join(ground_truth_root,meta_data_name)
        with open(meta_data_full_path,'r') as geojson_item:
            geojson_data = json.load(geojson_item)

        _dict_semantic_slice_to_ground_truth = {}

        if mode == 'vehicle_slice':
            pattern = re.compile(r'^\d+_\d+(_PE)?\.geojson$')
        elif mode == 'recons':
            # pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.geojson$')
            pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_line_fit.geojson$')
        else:
            raise ValueError("Invalid mode specified.")
        def find_files_with_pattern(directory):
            matching_files = []
            
            # 遍历目录及子目录
            for root, dirs, files in os.walk(directory):
                for file in files:
                    # 如果文件名匹配正则表达式，则添加到列表
                    if pattern.match(file):
                        matching_files.append(os.path.join(root, file))
            
            return matching_files
        
        target_geojson_path = find_files_with_pattern(target_geojson_path)

        #遍历文件夹内文件，迭代地找到该目录下所有的车端矢量数据
        slice_num = len(target_geojson_path)
        count = 0
        print("building index...")
        for vec_item in target_geojson_path:
            print(f'{count} / {slice_num}',end='\r')
            count += 1
            #返回一个列表，存储相关联的瓦片，首先置为空值
            _refered_tiles = []
            # vec_item_full_path = os.path.join(target_geojson_path,vec_item)
            
            # #获取当前帧的extent
            # _current_fc_extent = get_featurecollection_extent(vec_item_full_path)
           
            if mode == 'vehicle_slice':
                 _current_fc_extent = get_featurecollection_extent(vec_item)
            elif mode == 'recons':
                 _current_fc_extent = get_recons_feature_extent(vec_item)
            else:
                raise ValueError("Invalid mode specified.")
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
    _file_path = 'config.ini'
    PA_Assessor = positionAccuracyAssessor(_file_path)
    # PA_Assessor.batch_compare(
    #     # target_path = r".\\output\\"
    #     target_path = r'.\\reconstruction_output_temp\\'
    #     # if_index=True,
    #     # target_type ='recons'
    #     )
    # input()
    PA_Assessor.batch_compare(
        target_path = r'.\\reconstruction_output_temp\\',
        # if_index=True,
        target_type ='recons'
        )
    input()