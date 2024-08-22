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
    reconstruction = EgoviewReconstruction()
    reconstruction.batch_ego_reconstruction()

    #根据目标车端矢量数据查询当前数据的配到多源数据并打包
    target_vec_slice = "1690353428424700_1690353430258800_PE.geojson"
    # target_vec_slice = "1690353426889700_1690353428890700_PE.geojson"
    # target_vec_slice = "1690353425624699_1690353426890700_PE.geojson"
    # target_vec_slice = "1690354775327700_1690354780995200_PE.geojson"
    pipeline = PositionErrorPredictPipeline(target_vec_slice,"config.ini")
    print("this is the gate")
