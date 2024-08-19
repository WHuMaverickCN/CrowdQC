import os
import json
import configparser
import geopandas as gpd
import pandas as pd
import pickle

def read_loc_data(path):
    df = pd.read_csv(path)    
    return df
    
#此类别中需要编写关于读取
def read_mask_data(path,mask_type="mask"): 
    with open(path, 'rb') as f:
    # 使用pickle.load加载数据
        loaded_data = pickle.load(f)
    return loaded_data

def read_vec_data(path):
    with open(path,"r") as fp:
        _content = fp.read()
        print(_content)
        return _content

def read_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path,encoding = 'UTF8')

    data_root = config.get('Paths','data_root')

    feature_file_path_rela = config.get('Paths','feature_file_path')
    location_file_path_rela = config.get('Paths','location_file_path')
    vision_file_path_rela = config.get('Paths','vision_file_path')

    feature_file_path = os.path.join(data_root,feature_file_path_rela)
    location_file_path = os.path.join(data_root,location_file_path_rela)
    vision_file_path = os.path.join(data_root,vision_file_path_rela)

    return [feature_file_path,location_file_path,vision_file_path]

def read_sample_geojson_file(geojson_path):
    gdf_vec = gpd.read_file(geojson_path)
    return gdf_vec

def read_sample_location_file(location_paths):
    df_loc_set = {}
    for _index in range(len(location_paths)):
        df_loc = pd.read_csv(location_paths[_index])
        df_loc_set[_index] = df_loc
    
    merged_df = pd.DataFrame()

    # 迭代地合并 DataFrame
    for df in df_loc_set.values():
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    return merged_df

def raed_feather(feather_path):
    #此函数用用户读取相机的内部参数与外部参数
    with open(feather_path,"rb") as fp:
        _content = fp.read()
    return _content