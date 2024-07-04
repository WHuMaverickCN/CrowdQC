import re
import os
import math
import datetime
from dataclasses import dataclass
from datetime import datetime,timedelta
import pandas as pd
from .io import *

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率

TIME_SPAN_PER_DATA_PACKAGE = 60 #每个数据包（.dat）文件包含的原始数据所涵盖的时间范围通常为60秒
TILE_NAME = "505442"
VEHICLE_TYPE = "C385"

@dataclass
class Period:
    start_time:str
    end_time:str

class CoordProcessor:
    @staticmethod
    def gcj02towgs84_point_level(lng, lat):
        """
        GCJ02(火星坐标系)转WGS84
        :param lng:火星坐标系的经度
        :param lat:火星坐标系纬度
        :return:
        """
        def transformlat(lng, lat):
            ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
                0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
            ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                    math.sin(2.0 * lng * pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(lat * pi) + 40.0 *
                    math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
            ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
                    math.sin(lat * pi / 30.0)) * 2.0 / 3.0
            return ret
        def transformlng(lng, lat):
            ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
                0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
            ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
                    math.sin(2.0 * lng * pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(lng * pi) + 40.0 *
                    math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
            ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
                    math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
            return ret
        def out_of_china(lng, lat):
            """
            判断是否在国内，不在国内不做偏移
            :param lng:
            :param lat:
            :return:d
            """
            if lng < 72.004 or lng > 137.8347:
                return True
            if lat < 0.8293 or lat > 55.8271:
                return True
            return False
        if out_of_china(lng, lat):
            return lng, lat
        dlat = transformlat(lng - 105.0, lat - 35.0)
        dlng = transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return lng * 2 - mglng, lat * 2 - mglat
    
class TimeStampProcessor:
    @staticmethod
    def get_extra_suffix_dataframe(_dataframe):
        _dataframe['sec_of_week_last_three'] = _dataframe['sec_of_week'].astype(str).str[-3:].astype(float) / 1000
        _dataframe['utc'] = _dataframe['utc'] + _dataframe['sec_of_week_last_three']

        return _dataframe
    @staticmethod
    def convert_timestamp(timestamp):
        # 转换为10位float时间戳
        if len(str(timestamp)) == 16:
            _suffix = (timestamp % 10**6)/10**6
            timestamp = timestamp // 1000000 + _suffix
        else:
            timestamp_main = str(timestamp).split('.')[0]
            _scale = pow(10,len(str(timestamp_main))-10)
            timestamp = int(timestamp_main) // _scale
        return timestamp
    def __init__(self):
        pass
    @staticmethod
    def unify_timestamp(input):
        if input is str:
            print("str")
        else:
            print(type(input))
        return 1
    
    @staticmethod
    def check_period(period):
        period.start_time,precison_start_time = TimeStampProcessor.__check_timestamp_format(period.start_time)
        period.end_time,precison_end_time = TimeStampProcessor.__check_timestamp_format(period.end_time)
        precision = precison_start_time if precison_start_time==precison_end_time else max(precison_start_time,precison_end_time)
        return period,precision
    
    @staticmethod
    def __check_timestamp_format(temporal_timestamp):
        if type(temporal_timestamp) != str:
            temporal_timestamp = temporal_timestamp.__str__()
        l = len(temporal_timestamp)
        timestamp_pattern = r'^\d{10}$'
        timestamp_pattern_16 = r'^\d{16}$'

        precision = 0
        if l == 10 and re.match(timestamp_pattern, temporal_timestamp):
            precision = 10
            return float(temporal_timestamp),precision
        elif l==16 and re.match(timestamp_pattern_16, temporal_timestamp):
            precision = 16
            time_stamp_int = int(temporal_timestamp)
            time_stamp_float = time_stamp_int / 10**6
            return time_stamp_float,precision
        
    @staticmethod
    def trans_timestamp_to_general_format(temporal_timestamp):
        if len(temporal_timestamp) == 16:
            temporal_timestamp = int(int(temporal_timestamp)/1000000)
        data_obj = datetime.fromtimestamp(temporal_timestamp)
        return data_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def calculate_time_interval(period):
        # 将时间戳转换为 datetime 对象
        if period.start_time=='' or period.end_time=='':
            return ''
        dt1 = datetime.fromtimestamp(float(period.start_time))
        dt2 = datetime.fromtimestamp(float(period.end_time))
        
        # 计算时间间隔
        delta = dt2 - dt1
        
        # 提取时分秒
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        hours += delta.days*24
        # 格式化输出
        time_interval = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return time_interval
    
    @staticmethod
    def get_vec_start_end_timestamp(file_name,mode='vec'):
        #读取从数据包中解析出的各种数据类型。
        # return file_name
        if mode == 'vec':
            return file_name.split('_')[0],file_name.split('_')[1].split('.')[0]
        if mode == 'traj':
            return file_name.split('_')[1].split('.')[0]

    def get_raw_data_package_timestamp(self,file_name):
        # extract timestamp from the file name of loc \ vision data
        name,suffix = file_name.split('.')
        items = name.split('_')
        try:
            date = items[2]
            time = items[3]
            time_str = date + ' ' + time
            time_format = "%Y-%m-%d %H-%M-%S"
            local_time = datetime.strptime(time_str, time_format)
            beijing_timestamp = local_time.timestamp()
            print("utc+8:", beijing_timestamp)
        except IndexError:
            return None
        
class DataSearcher:
    @staticmethod
    def get_target_shape_data_path(directory,search_pattern):
        def find_files(directory, pattern,mode="file"):
            if mode=="file":#查找文件模式
                matches = []
                import fnmatch
                for root, dirs, files in os.walk(directory):
                    for filename in fnmatch.filter(files, pattern):
                        matches.append(os.path.join(root, filename))
                return matches
            elif mode=="dir":#查找文件夹模式
                matches = []
                import fnmatch
                for root, dirs, files in os.walk(directory):
                    for filename in fnmatch.filter(dirs, pattern):
                        matches.append(os.path.join(root, filename))
                return matches
        if "vision" not in directory:
            found_files = find_files(directory,search_pattern)
        else:
            found_files = find_files(directory,search_pattern,mode="dir")
        return found_files

    @staticmethod
    def get_raw_data_package(directory_path,input_timestamp):
        def __convert_timestamp_format(input_timestamp):
            # Parse the input timestamp string into a datetime object
            dt = datetime.strptime(input_timestamp, '%Y-%m-%d %H:%M:%S')
            
            # Format the datetime object into the desired output format
            output_timestamp = dt.strftime('%Y-%m-%d_%H-%M-%S')
            
            return output_timestamp

        def __find_matching_data_package(directory, target_timestamp):
            # Convert the target timestamp to a datetime object
            target_datetime = datetime.strptime(target_timestamp, '%Y-%m-%d_%H-%M-%S')
            
            matching_files = []
            
            # Walk through the directory and all its subdirectories
            for root, _, files in os.walk(directory):
                for filename in files:
                    # Check if the filename matches the expected format
                    if len(filename.split('_')) == 4:
                        # Extract the timestamp part of the filename
                        file_timestamp = filename.split('_')[2] + "_" + filename.split('_')[3].split('.')[0]
                        
                        try:
                            # Convert the extracted timestamp to a datetime object
                            file_datetime = datetime.strptime(file_timestamp, '%Y-%m-%d_%H-%M-%S')
                            # file_datetime_last = file_datetime + timedelta(seconds=TIME_SPAN_PER_DATA_PACKAGE)
                            # Compare the file timestamp with the target timestamp
                            if file_datetime < target_datetime and target_datetime < file_datetime + timedelta(seconds=TIME_SPAN_PER_DATA_PACKAGE):
                                matching_files.append(os.path.join(root, filename))
                        except ValueError:
                            # Skip files with incorrect timestamp format
                            continue
            data_package_name_list = []
            for _mf in matching_files:
                _,data_package_name = os.path.split(_mf)
                data_package_name = data_package_name.split('.')[0]
                data_package_name_list.append(data_package_name)

            return matching_files,data_package_name_list
        target_timestamp = __convert_timestamp_format(input_timestamp)
        matching_files = __find_matching_data_package(directory_path, target_timestamp)
        if matching_files:
            return matching_files
        else:
            print("No matching files found.")

@dataclass
class DataSamplePathWrapper:
    vec_path:str
    traj_path:str
    loc_path:str
    vis_path:str
    def __private():
        pass
    def __post_init__(self):
        #该读取过程为
        self.vec_data = input.read_sample_geojson_file(self.vec_path)
        self.traj_data = input.read_sample_geojson_file(self.traj_path)
        self.loc_data = self.__match_vec_to_loc()
        self.vis_data = self.__match_vec_to_vis()
        # self.vec_path = os.path.join(self.data_package_path,os.path.split(self.vec_path)[1])
        # self.traj_path = os.path.join(self.data_package_path,os.path.split(self.traj_path)[1])
    def write_sample_to_target_folder(self,target_folder="."):
        # 写入文件到目标文件夹
        output.write_to_foler(
                            self.vec_path,
                            self.traj_path,
                            self.loc_data,
                            self.vis_data
                        )
    def __match_vec_to_vis(self):
        _all_pic = []
        for i in range(len(self.vis_path)):
            all_pic_files = os.listdir(self.vis_path[i])
            _all_pic = _all_pic + all_pic_files
        
        def filter_and_organize_images(vis_path,image_list, start_time, end_time):
            # 将时间字符串转换为时间戳
            def timestamp_to_datetime(ts):
                return datetime.fromtimestamp(float(ts))
            
            # 筛选出在时间范围内的图片列表
            filtered_images = [
                img for img in image_list
                if start_time <= timestamp_to_datetime(img.split('_')[0]) <= end_time
            ]
            
            # 根据相机编号分别组织为两个列表
            camera_0_images = [img for img in filtered_images if img.endswith('_0.jpg')]
            camera_1_images = [img for img in filtered_images if img.endswith('_1.jpg')]

            camera_0_images = [os.path.join(vis_path[0],item) for item in camera_0_images]
            camera_1_images = [os.path.join(vis_path[0],item) for item in camera_1_images]
            return camera_0_images, camera_1_images

        temp_gdf_vec = self.vec_data.copy()
        temp_gdf_vec['start_time'] = temp_gdf_vec['start_time'].apply(TimeStampProcessor.convert_timestamp)
        temp_gdf_vec['end_time'] = temp_gdf_vec['end_time'].apply(TimeStampProcessor.convert_timestamp)
        
        _str_start_time = temp_gdf_vec['start_time'].iloc[0]
        _str_end_time = temp_gdf_vec['end_time'].iloc[0]
        start_time = datetime.fromtimestamp(_str_start_time)  # 开始时间
        end_time = datetime.fromtimestamp(_str_end_time)    # 结束时间

        camera_0_images, camera_1_images = filter_and_organize_images(self.vis_path,_all_pic, start_time, end_time)

        return {'cam_0':camera_0_images,'cam_1':camera_1_images}
    def __match_vec_to_loc(self):
        # 匹配轨迹和定位数据
        target_df_loc = input.read_sample_location_file(self.loc_path) #此处的self.loc_path为文件路径组成的列表，可能包含多个
        # temp = TimeStampProcessor.convert_timestamp("1234567891")
        # target_df_loc['utc'] = target_df_loc['utc'].apply(TimeStampProcessor.convert_timestamp)  # 如果时间戳是以秒为单位的
        target_df_loc = TimeStampProcessor.get_extra_suffix_dataframe(target_df_loc)
        # 将 GeoDataFrame 的时间戳转换为10位
        temp_gdf_vec = self.vec_data.copy()
        temp_gdf_vec['start_time'] = temp_gdf_vec['start_time'].apply(TimeStampProcessor.convert_timestamp)
        temp_gdf_vec['end_time'] = temp_gdf_vec['end_time'].apply(TimeStampProcessor.convert_timestamp)

        # 遍历 GeoDataFrame
        for index, geo_obj in temp_gdf_vec.iterrows():
            print(index)
            # 获取当前 GeoDataFrame 的时间戳
            start_time = geo_obj['start_time']
            end_time = geo_obj['end_time']

            # 通过时间戳筛选轨迹点
            filtered_points = target_df_loc[
                (target_df_loc['utc'] >= start_time) &
                (target_df_loc['utc'] <= end_time)
            ]

            if filtered_points.empty==True:
                #未筛选到符合时间要求的轨迹点
                continue
            def convert_row(row):
                new_lng, new_lat = CoordProcessor.gcj02towgs84_point_level(row['longitude'], row['latitude'])
                return pd.Series([new_lng, new_lat], index=['new_longitude', 'new_latitude'])
            filtered_points[['new_longitude', 'new_latitude']] = filtered_points.apply(convert_row, axis=1)
            # filtered_points = TimeStampProcessor.get_extra_suffix_dataframe(filtered_points)
            return filtered_points
            '''
            uniq_time_stamp = []
            if filtered_points.empty==True:
                #未筛选到符合时间要求的轨迹点
                continue
            else:
                if start_time not in uniq_time_stamp:
                    uniq_time_stamp.append(start_time)
                print(filtered_points.shape[0])
                _pe = geo_obj['positional_error']
                label = get_label_from_positional_error(_pe)
                _pt = filtered_points['position_type'].mode()
                cluster_pt = get_cluster_from_position_type(_pt[0])

                result_df = result_df.append(pd.Series({'clus':cluster_pt,'label':label}), ignore_index=True)
            '''
        pass
class VehicleDataset:
    # 配置矢量数据与轨迹数据的根目录
    # vec_root:str
    # traj_root:str
    def __init__(self,vec_full_path,traj_full_path):
        self.vec_root = os.path.split(vec_full_path[0])[0]
        self.traj_root = os.path.split(traj_full_path[0])[0]
