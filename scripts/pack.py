import os
import shutil
import re


# 正则表达式匹配UUID
uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
created_folders = set()
target_dir = r'E:\caq_data\reconstruction_output_0923\reconstruction_output_0923'
source_dir = r'E:\caq_data\reconstruction_output_0923\reconstruction_output_0923'
file_list = os.listdir(target_dir)
for file_name in file_list:
    # 使用正则表达式查找UUID
    match = uuid_pattern.search(file_name)
    if match:
        # 提取UUID
        uuid = match.group(0)
        
        # 创建以UUID命名的新文件夹路径
        new_folder_path = os.path.join(target_dir, uuid)
        
        # 如果文件夹不存在，则创建它
        if uuid not in created_folders:
            os.makedirs(new_folder_path, exist_ok=True)
            created_folders.add(uuid)
        
        # 构建源文件和目标文件的完整路径
        source_file_path = os.path.join(source_dir, file_name)
        target_file_path = os.path.join(new_folder_path, file_name)
        
        # 移动文件
        shutil.move(source_file_path, target_file_path)
        print(f"Moved {file_name} to {new_folder_path}")

print("All files have been moved to their respective folders.")
input()
'''
# 定义目录路径
dir_a = r'F:\temp\CrowdQC\reconstruction_output_0923'  # 替换为目录A的路径
dir_b = r'F:\temp\CrowdQC\output'  # 替换为目录B的路径
dir_c = r'G:\fit0925'  # 替换为目录C的路径，将要创建的目录

# 创建目录C
if not os.path.exists(dir_c):
    os.makedirs(dir_c)

# 复制目录B到目录C
for item in os.listdir(dir_b):
    print(item)
    src_path = os.path.join(dir_b, item)
    dest_path = os.path.join(dir_c, item)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest_path)

# 正则表达式匹配 uuid 和其后可能跟随的 _recons_log, _line_fit, 以及 .geojson, .json
uuid_pattern = re.compile(r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(_recons_log|_line_fit)?(\.geojson|\.json)?$')
# 在目录A中搜索文件，并复制到目录C的对应子目录
for root, dirs, files in os.walk(dir_a):
    for file in files:
        # 使用正则表达式搜索匹配的文件
        match = uuid_pattern.match(file)
        if match:
            # 提取文件名中的uuid部分
            uuid_part = match.group(1)
            if uuid_part:  # 如果成功提取到uuid部分
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                # 找到对应的目录C中的子目录
                target_dir = os.path.join(dir_c, uuid_part)
                if os.path.exists(target_dir):  # 如果目录存在
                    shutil.copy(file_path, target_dir)  # 复制文件到目标目录
                else:
                    print(f"目录 {target_dir} 不存在，无法复制文件 {file_path}")
        else:
            print(f"文件 {file} 不匹配指定的模式")
print("文件组织完成。")
'''