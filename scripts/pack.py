import os
import shutil
import re
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