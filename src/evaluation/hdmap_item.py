import os
import json
import sys
from ..io import input
from ..utils import *
from .match_utils import transform_coordinates
from shapely.geometry import shape,LineString,Point
from rtree import index
try:
    from osgeo import ogr
except:
    sys.exit('ERROR: 未找到 GDAL/OGR modules')

from .match_utils \
import get_featurecollection_extent,\
    get_geojson_item_from_ogr_datasource

'''
此函数是使用gdal读取长安提供的真值地图要素
'''
class HdData:
    data = None
    def build_si(self):
        rt_index = index.Index(interleaved=False)
        if self.data.GetLayerCount()<1:
            return rt_index
        utm_trans_data = transform_coordinates(self.data, 32648)
        for _layer in utm_trans_data:
            for _feature in _layer:
                geom = _feature.GetGeometryRef()
                envelope = geom.GetEnvelope()  # 获取边界框
                fid = _feature.GetFID()  # 获取feature ID
                rt_index.insert(fid, envelope)
        return rt_index
    
    def get_items_by_id(self,
                        feature_ids):
        # self.data = transform_coordinates(self.data, 32648)
        items = {}
        if self.data!=None:
            for _layer in self.data:
                for _id in feature_ids:
                    items[_id] = _layer.GetFeature(_id)
        return items

class VehiclesData(HdData):
    data = None
    time_slice = 'no-time-slice'
    _rtree_index = None
    id2timeslice_dict = {}
    source_data_path = ''
    ''' 
    该部分实际填充的是车端数据，读取车端捕捉的地图要素
    '''
    def __init__(self,data,
                 time_slice = 'no-time-slice',
                 dataForm = '车端数据'):
        #此处实现一个类似于重载的逻辑，通过参数类型构造不同的车端数据类型
        if os.path.isfile(data) and type(time_slice) == str:
            self.data = input._m_read_data_to_ogr_datasource(data)
            self.time_slice = time_slice
        elif os.path.isdir(data):
            if time_slice == 'no-time-slice':
                self.data = input._m_read_data_to_ogr_datasource_batch(data)
                self.time_slice = time_slice
            else:
                #此时该文件不存在
                return
        elif type(data) == ogr.DataSource:
            self.data = data
            self.time_slice = time_slice
    # def read_data(self,data):
    #     self.data = dataload.read_data_with_gdal(data)
    def __build_Rtree_sdb(self,hd_vehicles_data):
        '''
        此函数对于一个车端数据构建一个Rtree数据库
        '''
        if type(hd_vehicles_data.data) == dict:
            _id2timeslice_dict = {}
            _id_index = 0
            idx = index.Index()
            for _key in list(hd_vehicles_data.data.keys()):
                _m_current_bound = tuple(get_featurecollection_extent(hd_vehicles_data.data[_key]))
                idx.insert(_id_index,_m_current_bound)
                # print(_count)
                _id2timeslice_dict[_id_index] = _key
                _id_index += 1
            #计算获得一个边界框
            self._rtree_index = idx
            self.id2timeslice_dict = _id2timeslice_dict
        elif isinstance(hd_vehicles_data.data,ogr.DataSource) and hd_vehicles_data.data.name!='':
            _id2timeslice_dict = {}
            _id_index = 0
            idx = index.Index()
            _geojson_data = get_geojson_item_from_ogr_datasource(hd_vehicles_data)
            #构建临时的R树
            for feature_item in _geojson_data['features']: 
                _geom_shapely = shape(feature_item['geometry'])
                _m_current_bound = tuple(_geom_shapely.bounds)
                idx.insert(_id_index,_m_current_bound)
                # print(feature_item)
                _id_index += 1
            self._rtree_index = idx
        elif type(hd_vehicles_data.data) == ogr.DataSource and hd_vehicles_data.data.name=='':
            _id2timeslice_dict = {}
            _id_index = 0
            idx = index.Index()
            for layer in hd_vehicles_data.data:
                for feature in layer:
                    geometry = feature.GetGeometryRef()
                    _m_current_bound = tuple(geometry.GetEnvelope())
                    idx.insert(_id_index,_m_current_bound)
                    _id_index += 1
            self._rtree_index = idx
            # return idx,_id2timeslice_dict
    def get_items_from_raw_data_by_id(self,hd_item_id,reference_hd_item):
        '''
        此函数采用高精地图要素id作为输入，在车端数据集中搜索位置存在重叠的车端矢量数据，当前数据集中所有的车端矢量
        '''
        _geom_shapely = None

        _current_index = self._rtree_index
        reference_data = get_geojson_item_from_ogr_datasource(reference_hd_item)
        for feature_item in reference_data['features']:
            if feature_item['properties']['id'] == hd_item_id and feature_item["geometry"]!={} and feature_item["geometry"]!= None:
                _geom_shapely = shape(feature_item['geometry'])

        if _geom_shapely!=None:
            _bound = tuple(_geom_shapely.bounds)
            out = list(_current_index.intersection(_bound))
            return [self.id2timeslice_dict[_id] for _id in out]
        else:
            print("未找到对应id的高精地图要素")
            return -1

    def get_items_from_raw_data_by_extent(self,extent):
        '''
        此函数采用要素的范围作为输入，在车端数据集中搜索位置存在重叠的车端矢量数据，当前数据集中所有的车端矢量
        '''
        _current_index = self._rtree_index
        out = list(_current_index.intersection(extent))
        return [self.id2timeslice_dict[_id] for _id in out]
    def set_source_data_path(self,str_path):
        if type(str_path) == str:
            self.source_data_path = str_path
            return 1
        else:
            return -1
        
    @print_run_time('数据匹配字典构建')
    def get_relavant_trajectory_data(self,trajectory_folder_name):
        '''通过时间关联至对应轨迹数据
        '''
        dict_vec_to_traj = {}

        target_traj_path = os.path.join(self.source_data_path,trajectory_folder_name)
        target_traj_name_list = os.listdir(target_traj_path)
        set_target = set(target_traj_name_list)

        if type(self.data) == dict:
            for _key in self.data.keys():
                start_time = _key.split('_')[0]
                # for _traj_file_name in target_traj_name_list:
                for _traj_file_name in set_target:
                    if start_time in _traj_file_name:
                        _traj_file_path = os.path.join(self.source_data_path,trajectory_folder_name,_traj_file_name)
                        dict_vec_to_traj[_key] = _traj_file_path
                        # print(_key,_traj_file_path)
                # print(start_time)
        print("数据匹配字典构建完毕")
        return dict_vec_to_traj

    def get_relavant_vision_data(self):
        '''通过时间关联至对应视觉数据
        '''
        return 1
    
    def get_sensor_angle_batch(self,dict_vec_to_traj):
        '''获取车端所有slice中，每个矢量化要素被观测到时，相机位置所在的角度    
        '''
        #初始化局部变量，记录要素获取时间戳对应的id
        _id_start_time_filed = -1
        _id_end_time_filed = -1

        segment_length = 50

        for _key in dict_vec_to_traj.keys():
            target_slice = self.data[_key]
            for _layer_index in range(target_slice.GetLayerCount()):
                _layer = target_slice.GetLayerByIndex(_layer_index)
                # _lyr_name = _layer.GetName()
                _layer_defn = _layer.GetLayerDefn()
                _field_count = _layer_defn.GetFieldCount()

                for i in range(_field_count):
                    _field_name = _layer_defn.GetFieldDefn(i).GetName()

                    #记录关键字段顺序id
                    if _field_name == 'start_time':
                        _id_start_time_filed = i
                    elif _field_name == 'end_time':
                        _id_end_time_filed = i
                    # print(_field_name)
                with open(dict_vec_to_traj[_key], 'r') as trajectory_file:
                    trajectory_data = json.load(trajectory_file)
                    
                # 计算矢量化要素与轨迹线的最近点
                trajectory_line = LineString([(point['geometry']['coordinates'][0], 
                                            point['geometry']['coordinates'][1]) 
                                            for point in trajectory_data['features']])
                trajectory_points = [(point['geometry']['coordinates'][0], 
                                      point['geometry']['coordinates'][1], 
                                      point['properties']['timestamp']) 
                                      for point in trajectory_data['features']]
                num_segments = int(trajectory_line.length / segment_length)
                if num_segments == 0:
                    num_segments = 1
                result_dict = {}
                _feature_index = 0
                for _feature in _layer:
                    start_time_of_cur_feature = _feature.GetField('start_time')
                    end_time_of_cur_feature = _feature.GetField('end_time')
                    # print(start_time_of_cur_feature,end_time_of_cur_feature)
                    shapely_feature = pretreatment.ogr_to_shapely(_feature)

                    vector_feature_centroid = shapely_feature.centroid

                    # 存储匹配结果的列表
                    matching_points = []

                    for i in range(num_segments):
                        start = i * segment_length
                        end = (i + 1) * segment_length

                        segment_line = LineString(list(trajectory_line.coords)[start:end])
                        
                        nearest_point = segment_line.interpolate(segment_line.project(vector_feature_centroid))

                        projected_points = sorted([
                            (x, y, timestamp, segment_line.project(Point(x, y))) 
                            for x, y, timestamp in trajectory_points
                        ], key=lambda point_info: point_info[3])

                        for x, y, timestamp, _ in projected_points:
                            matching_points.append({
                                'time': timestamp,
                                'trajectory_point': (x, y),
                                'vector_feature_centroid': vector_feature_centroid,
                                'nearest_point_on_trajectory': nearest_point
                            })
                    _feature_index += 1
                    # 保存结果到字典
                    result_dict[_feature_index] = matching_points
                    self._vis_project(_feature_index,shapely_feature,trajectory_line,nearest_point)
                
                print(result_dict)
                '''
                    # 计算矢量化要素质心与轨迹线的最近点
                    nearest_point = nearest_points(vector_feature_centroid, trajectory_line)[0]
       
                    # 计算轨迹点在轨迹线上的投影位置
                    projection_position = trajectory_line.project(nearest_point,normalized=True)

                    self._vis_project('feature_name',shapely_feature,trajectory_line,nearest_point)
                    print(projection_position)
                    '''
                    # overall_direction = measurement.calculate_direction(shapely_feature,trajectory_data)
    def _vis_project(self,feature_name, vector_feature, trajectory_line, nearest_point):

        import matplotlib.pyplot as plt
        import geopandas as gpd

        # 创建 GeoDataFrame 用于可视化
        gdf_vector = gpd.GeoDataFrame(geometry=[vector_feature])
        gdf_trajectory = gpd.GeoDataFrame(geometry=[trajectory_line])
        gdf_nearest_point = gpd.GeoDataFrame(geometry=[Point(nearest_point.x, nearest_point.y)])

        # 绘制地图
        fig, ax = plt.subplots(figsize=(8, 8))

        # 可视化矢量化要素、轨迹线、最近点
        gdf_vector.plot(ax=ax, color='blue', alpha=0.5, label='Vector Feature')
        gdf_trajectory.plot(ax=ax, color='green', alpha=0.5, label='Trajectory Line')
        gdf_nearest_point.plot(ax=ax, color='red', marker='o', label='Nearest Point')

        # 设置标题和标签
        ax.set_title(f"Matching Result for {feature_name}")
        ax.legend()

        # 显示图形
        plt.show()

class HdmapData(HdData):
    ''' 
    待填充 
    '''
    def __init__(self,data,
                 dataForm = '高精地图真值',
                 crf = 'wgs84',
                 target_crf = 'utm111'):
        self.data = input._m_read_data_to_ogr_datasource(data)
        if crf == 'wgs84' and target_crf == 'utm111':
            self.data = transform_coordinates(self.data, 32648)

class MapLearnData(HdData):
    ''' 
    此处存储众包地图学习数据
    '''
    def __init__(self,data,
                 dataForm = '众包地图学习数据',
                 crf = 'wgs84',
                 target_crf = 'utm11'):
        self.data = input._m_read_data_to_ogr_datasource(data)
        if crf == 'wgs84' and target_crf == 'utm11':
            self.data = transform_coordinates(self.data, 32648)
        # 在进行初始时，此处的数据并未经过gcj02转换，因此需要进行转换
        
class CartographicData(HdData):
    ''' 
    此处存储众包地图学习数据
    '''
    def __init__(self,data,
                 dataForm = '众包地图学习数据',
                 crf = 'wgs84',
                 target_crf = 'utm11'):
        self.data = input._m_read_data_to_ogr_datasource(data)
        if crf == 'wgs84' and target_crf == 'utm11':
            self.data = transform_coordinates(self.data, 32648)
    # 在进行初始时，此处的数据并未经过gcj02转换，因此需要进行转换
