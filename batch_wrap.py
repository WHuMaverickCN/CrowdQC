import os
import logging
from concurrent.futures import ThreadPoolExecutor
from src.pose.pipeline import PositionErrorPredictPipeline
from src.reconstruction.egoview import EgoviewReconstruction

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(message)s',  # 日志格式：时间戳 - 消息内容
    handlers=[
        logging.FileHandler("data_processing_d1106_l35_v727.log"),  # 将日志保存到文件
        logging.StreamHandler()  # 同时在控制台打印
    ]
)
global count
global all_num
def process_file(target_vec_slice):
    """处理单个文件的函数."""
    if target_vec_slice.endswith(".geojson"):
        global count
        count+=1
        print(f"-----------deal {count}/{all_num} slices---------------")
        logging.info(f"Processing: {target_vec_slice}")
        pipeline = PositionErrorPredictPipeline(
            vec_path=target_vec_slice,
            config_file="config.ini",
            id=True,
            target_folder="/home/gyx/data/cqc/processed/clips_d1114_l17_v703/"
        )
        logging.info(f"Completed: {target_vec_slice}")

if __name__ == "__main__":
    # target_path = r'D:\ca_data_3rd\datasets\features\l17\2023-06-19\LS6A2E161NA505442_L17_2s\VecJsonData'
    # target_path = r"D:\ca_data_3rd\datasets\features\l17\2023-06-26\xxxx_L17_5s\VecJsonData"
    # target_path = r'D:\ca_data_3rd\datasets\features\l17\2023-07-03\xxxx_L17_5s\VecJsonData'

    # target_path = r"D:\ca_data_3rd\datasets\features\l17\2023-07-21\LS6A2E161NA505442_L17\VecJsonData"
    # target_path = r"D:\ca_data_3rd\datasets\features\l17\2023-07-24\LS6A2E161NA505442_L17\VecJsonData"
    # target_path = "/home/gyx/data/cqc/datasets_L17/features/l17/2023-06-19/LS6A2E161NA505442_L17_2s/VecJsonData"
    target_path = "/home/gyx/data/cqc/datasets_L17/features/l17/2023-07-03/xxxx_L17_5s/VecJsonData"
    

    # 获取目标目录下的所有文件
    file_list = os.listdir(target_path)
    # global all_num
    all_num = len(file_list)
    # 线程池，设定并行线程数（可根据 CPU 数量调整）
    max_workers = 4  # 你可以根据需求调整线程数量
    count = 0
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for target_vec_slice in file_list:
            # 提交任务给线程池
            futures.append(executor.submit(process_file, target_vec_slice))

        # 可选：等待所有线程完成并打印结果
        for future in futures:
            future.result()  # 如果有需要，可以检查任务的返回结果

    logging.info("All tasks completed.")
