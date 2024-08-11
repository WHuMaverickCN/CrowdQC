# 本脚本用于快速构建LSTM实验，相关路径采用配置文件读取
# 注意不用搞核心算法以外的冗余逻辑
from src.pose.pipeline import PositionErrorPredictPipeline

if __name__ == "__main__":
    #根据目标车端矢量数据查询当前数据的配到多源数据并打包
    target_vec_slice = "1690354775327700_1690354780995200_PE.geojson"
    pipeline = PositionErrorPredictPipeline(target_vec_slice,"config.ini")
    print("this is the gate")