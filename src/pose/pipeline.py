import re
import os
import logging
from ..io import input
from ..utils import *

logger = logging.getLogger(__name__)

class PositionErrorPredictPipeline():
    def __init__(self,
                 vec_path="",
                 config_file="config.ini",
                 id = True,
                 target_folder = 'output'):
        self.vec_path = vec_path
        self.id_flag = id
        self.target_folder = target_folder
        self.read_config(config_file)
        self.get_target_traj_path()

    def read_config(self,_path):
        ''''''
        feature_file_path,location_file_path,vision_file_path = input.read_config_file(_path)
        self.feature_file_path = feature_file_path
        self.location_file_path = location_file_path
        self.vision_file_path = vision_file_path
        self.vision_file_path = "I:\\ca data\\11_解析数据\\07-25_route35\\03_视觉数据"
        # feature_file_path,location_file_path,vision_file_path = input.read_config_file_for_batch(_path)

        # set the combox item of routes
        routes = os.listdir(feature_file_path)
        route_alias_pattern = r'^[a-zA-Z]+\d+$'  # 匹配以字母开头，后跟数字的命名格式 
        
        routes_list = []
        for route in routes:
            if re.match(route_alias_pattern,route):
                routes_list.append(route)
    
    def get_target_traj_path(self):
        # 获得目标时间戳
        start_time, end_time = TimeStampProcessor.get_vec_start_end_timestamp(self.vec_path)
        _period = Period(start_time, end_time)
        target_period, precision = TimeStampProcessor.check_period(_period)
        time_interval = TimeStampProcessor.calculate_time_interval(target_period)

        # 构造轨迹文件路径
        traj_path = "trajectory_" + start_time + ".geojson"

        # 获取完整的轨迹文件路径和向量文件路径
        full_vec_path_list = DataSearcher.get_target_shape_data_path(self.feature_file_path, self.vec_path)
        full_traj_path_list = DataSearcher.get_target_shape_data_path(self.feature_file_path, traj_path)

        # 创建VehicleDataset对象
        _current_vehicle_dataset = VehicleDataset(full_vec_path_list, full_traj_path_list)

        _gf_start_time = TimeStampProcessor.trans_timestamp_to_general_format(start_time)
        _gf_end_time = TimeStampProcessor.trans_timestamp_to_general_format(end_time)

        target_loc_data_package,package_names = DataSearcher.get_raw_data_package(self.location_file_path,_gf_start_time,_gf_end_time)
        if target_loc_data_package == -1 and package_names == -1:
            print(f"skip:{self.vec_path}")
            logger.info(f"skip:{self.vec_path}")
            return
        for package_name in package_names:
            target_vis_data_package = DataSearcher.get_target_shape_data_path(self.vision_file_path, package_name)
        
        if len(target_loc_data_package)==0 or len(target_vis_data_package)==0:
            print(f"skip:{self.vec_path}")
            logger.info(f"skip:{self.vec_path}")
            return
        # 打包文件路径为DataSamplePathWrapper对象
        if len(full_traj_path_list) >= 1 and len(full_vec_path_list) >= 1:
            full_traj_path = full_traj_path_list[0]
            full_vec_path = full_vec_path_list[0]
            _current_sample = DataSamplePathWrapper(full_vec_path,
                                                    full_traj_path, 
                                                    target_loc_data_package,
                                                    target_vis_data_package)
            _current_sample.write_sample_to_target_folder(self.target_folder,self.id_flag)
            logger.info(f"finish:{self.vec_path}")