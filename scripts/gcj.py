import os
from src.utils import print_run_time,\
                    MapLearnCoordProcessor

mlcp = MapLearnCoordProcessor()
# coord trans for 
target_path = "/home/gyx/data/cqc/cloud/temp"
project_names = []
for root, dirs, files in os.walk(target_path):
# root 是当前目录的路径
# dirs 是当前目录中的子文件夹列表
# files 是当前目录中的文件列表
    # 遍历文件
    for f in files:
        geojson_item = os.path.join(root, f)
        project_names.append(geojson_item)
    # # 遍历所有的文件夹,在out目录下创造同名文件夹
    # for d in dirs:
    #     twin_dir = os.path.join(root, d).replace('JsonData','JsonData_noH')
    #     # if 'JsonData' in twin_dir:
    #     #     # twin_dir = os.path.join(root, d).replace('VecJsonData','VecJsonData_noH')        
    #     #     os.makedirs(twin_dir)
    #     print(twin_dir)
    # # project_names.extend(files)

    for i_target_path in project_names:
        mlcp.trans_gcj02towgs84(i_target_path)
    print("ok")