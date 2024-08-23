# 本脚本用于快速构建LSTM实验，相关路径采用配置文件读取
# 注意不用搞核心算法以外的冗余逻辑


from src.pose.pipeline import PositionErrorPredictPipeline
from src.reconstruction.egoview import EgoviewReconstruction
if __name__ == "__main__":
    # 使用示例
    '''
    # 使用1: 直接对图像进行逆透视变换读取参数
    reconstruction = EgoviewReconstruction(path_to_data='some_path')
    undistorted_img = reconstruction.get_undistort_img()
    # 批量进行车道线重建
    reconstruction.inverse_perspective_mapping(undistorted_img)
    '''

    # 使用2: 批量进行车道线重建
    # reconstruction = EgoviewReconstruction()
    # reconstruction.batch_ego_reconstruction()

    #根据目标车端矢量数据查询当前数据的配到多源数据并打包
    import os
    target_path = r"E:\caq_data\dataset\features\l35\2023-07-27\LS6A2E161NA505442_L35\VecPE"
    file_list = os.listdir(target_path)
    # target_vec_slice = "1690181536394800_1690181541927900_PE.geojson"

    # target_vec_slice = "1690353428424700_1690353430258800_PE.geojson"
    # target_vec_slice = "1690353426889700_1690353428890700_PE.geojson"
    # target_vec_slice = "1690353425624699_1690353426890700_PE.geojson"
    # target_vec_slice = "1690354775327700_1690354780995200_PE.geojson"
    # pipeline = PositionErrorPredictPipeline(vec_path = target_vec_slice,
    #                                         config_file = "config.ini",
    #                                         id = True,
    #                                         target_folder = 'output')
    _count = 0
    for target_vec_slice in file_list:
        print(f"scan:{_count} / {len(file_list)}")
        if target_vec_slice.endswith(".geojson"):
            pipeline = PositionErrorPredictPipeline(vec_path = target_vec_slice,
                                            config_file = "config.ini",
                                            id = True,
                                            target_folder = 'output')
        _count = _count + 1
