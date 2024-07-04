import pandas as pd
import shutil
import os
def wirte_cluster_to_csv(path):
    """
    Write a dataframe to csv file.
    :param df: Dataframe
    :type df: pandas.DataFrame
    :param path: Path of the output file
    :type path: str
    """
    result_df = pd.DataFrame(columns=['clus','label'])
    
    result_df.to_csv(path)

def write_to_foler(full_vec_path,#  矢量全路径
                   full_traj_path,# 轨迹全路径
                   fetched_loc_data_frame,#筛选到的定位数据dataframe
                   vis_data_dict,
                   target_folder='output',
                   index='0'):# 目标路径
    # 检查目标文件夹是否存在，如果不存在则创建它
    target_sample_folder = os.path.join(target_folder,index)
    if not os.path.exists(target_sample_folder):
        os.makedirs(target_sample_folder)

    shutil.copy(full_vec_path,target_sample_folder)
    shutil.copy(full_traj_path,target_sample_folder)
    fetched_loc_data_frame.to_csv(target_sample_folder+'/loc_data.csv')
    
    for _key in vis_data_dict.keys():
        _target_root = os.path.join(target_sample_folder, _key)
        if not os.path.exists(_target_root):
            os.makedirs(_target_root)

        for _path in vis_data_dict[_key]:
            _target_path = os.path.join(target_sample_folder,_key)
            shutil.copy(_path,_target_path)

    