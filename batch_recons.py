from src.pose.pipeline import PositionErrorPredictPipeline
import logging
from src.reconstruction.egoview import EgoviewReconstruction

from config import inference

if __name__ == "__main__":
    # 配置日志记录器

    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(message)s',  # 日志格式：时间戳 - 消息内容
        handlers=[
            logging.FileHandler("data_processing.log"),  # 将日志保存到文件
            logging.StreamHandler()  # 同时在控制台打印
        ]
    )
    print("批量重建")
    # 使用2: 批量进行车道线重建
    # target_recons_dir = "G:\\Detect"
    target_recons_dir = "/home/gyx/data/cqc/processed/clips_d1106_l35_v727"
    reconstruction = EgoviewReconstruction()
    reconstruction.batch_ego_reconstruction(target_recons_dir,
                                            mask_mode="jpg")
