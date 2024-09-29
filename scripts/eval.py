# 读取每个uuid目录下的target_batch_dir = r'E:\fit0925'
import os
import json 
import matplotlib.pyplot as plt

def remove_if_list2_greater_than_three(list1, list2):
    """
    遍历两个列表，删除list2中值大于3的元素以及list1中对应位置的元素
    :param list1: 第一个列表
    :param list2: 第二个列表
    :return: 处理后的两个列表
    """
    filtered_list1 = []
    filtered_list2 = []
    
    for val1, val2 in zip(list1, list2):
        if val2 <= 3:  # 只要list2中的值不大于3
            filtered_list1.append(val1)
            filtered_list2.append(val2)
    
    return filtered_list1, filtered_list2

def normalize(numbers):
    if not numbers:
        return []

    # 找到列表中的最小值和最大值
    min_val = min(numbers)
    max_val = max(numbers)

    # 如果所有值都相同，返回全为0的列表（避免除以零）
    if min_val == max_val:
        return [0 for _ in numbers]

    # 归一化列表
    normalized_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
    return normalized_numbers
def plot_two_lists(list1, list2):
    """
    在同一个图中绘制两个列表的折线图
    :param list1: 第一个列表，包含若干数值
    :param list2: 第二个列表，包含若干数值
    """
    plt.figure(figsize=(10, 6))  # 设置图形大小
    
    # 绘制第一个列表的折线图
    plt.plot(list1, label='List 1', color='#ffcc00')
    # plt.plot(list1, label='List 1', color='red')
    # 绘制第二个列表的折线图
    plt.plot(list2, label='List 2', color='#3cb371')
    # plt.plot(list2, label='List 2', color='blue')
    # 设置标题和坐标轴标签
    plt.title('Line Plot of Two Lists')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # 显示图例
    plt.legend()
    
    # 显示图形
    plt.show()

# # 示例调用
# list1 = [i for i in range(1500)]
# list2 = [i * 1.5 for i in range(1500)]

def average_of_dict_values(input_dict):
    # 确保字典不为空
    if not input_dict:
        return 0

    # 计算所有值的总和
    total_sum = sum(input_dict.values())
    # 获取值的数量
    count = len(input_dict)
    # 计算平均值
    average = total_sum / count
    return average
def read_result_json(file_path):
    with open(file_path,'r') as fp:
        return json.load(fp)
if __name__ == '__main__':
    sum1 = 0
    sum2 = 0

    jueduiweizhiwucha = []
    chongjian_jueduiweizhiwucha = []

    target_batch_dir = r'D:\BaiduSyncdisk\workspace\py\CrowdQC\reconstruction_output_temp'
    count = 0

    file1 = 'veh2recons_bias_hau_distance.json'
    # file1 = 'recons_bias.json'
    file2 = 'slice_bias.json'

    qlf_slice_num=0
    feat_num = 0
    slice_num = 0
    for _uuid in os.listdir(target_batch_dir):
        sub_folder = os.path.join(target_batch_dir,
                                    _uuid)
        # 检查当前目录下是否存在这两个文件

        if not (os.path.isfile(
                    os.path.join(target_batch_dir,
                        _uuid,
                        file1)) and 
                os.path.isfile(
                    os.path.join(target_batch_dir,
                        _uuid,
                        file2))):
            continue
            # print(f"文件 {file1} 或 {file2} 不存在，将跳过执行。")
        else:
        # num = os.listdir(sub_folder).__len__() 
        # if num<13:
            slice_num += 1
            target_veh2recons_bias_path = os.path.join(target_batch_dir,
                                                    _uuid,
                                                    file1)
            veh2recons_bias = read_result_json(target_veh2recons_bias_path)

            target_slice_bias = os.path.join(target_batch_dir,
                                                    _uuid,
                                                    file2)
            slice_bias = read_result_json(target_slice_bias)
            
            # sb = average_of_dict_values(slice_bias)
            # jueduiweizhiwucha.append(sb)

            # rb = average_of_dict_values(veh2recons_bias)
            # chongjian_jueduiweizhiwucha.append(rb)
            for ksb,vsb in slice_bias.items():
                jueduiweizhiwucha.append(vsb)
            for krb,vrb in veh2recons_bias.items():
                chongjian_jueduiweizhiwucha.append(vrb)
            # if sb<1:
            #     qlf_slice_num+=1
            # for _key,_v in veh2recons_bias.items():
            #     feat_num += 1
            #     if slice_bias[_key]<1:
            #         count+=1
            #     print(slice_bias[_key]/_v)

    print(qlf_slice_num,"/",slice_num)  
    def remove_values_greater_than_five(list1, list2):
        """
        遍历两个列表，删除列表中值大于5的元素以及另一个列表中对应位置的元素
        :param list1: 第一个列表
        :param list2: 第二个列表
        :return: 处理后的两个列表
        """
        # 使用 zip 将两个列表对应值组合在一起，并通过列表推导式过滤值
        filtered_list1 = []
        filtered_list2 = []
        
        for val1, val2 in zip(list1, list2):
            if val1 <= 5 and val2 <= 5:  # 如果两个列表中的值都不大于5
                filtered_list1.append(val1)
                filtered_list2.append(val2)
        
        return filtered_list1, filtered_list2
    # new_list1, new_list2 = remove_values_greater_than_five(jueduiweizhiwucha, chongjian_jueduiweizhiwucha)
    # plot_two_lists(new_list1[400:500], new_list2[400:500])

    # new_list1, new_list2 = remove_if_list2_greater_than_three(jueduiweizhiwucha, chongjian_jueduiweizhiwucha)
    v2recons, v2gt = remove_if_list2_greater_than_three(chongjian_jueduiweizhiwucha,jueduiweizhiwucha)



    new_list1 = normalize(v2recons)
    new_list2 = normalize(v2gt)
    # normalize
    plot_two_lists(new_list1[100:900], new_list2[100:900])

    def average_of_smaller_90_percent(numbers):
        if not numbers:  # 检查列表是否为空
            return None
        
        # 对列表进行排序
        sorted_numbers = sorted(numbers)
        
        # 计算90%的长度
        percentage_length = int(len(sorted_numbers) * 0.9)
        
        # 获取90%的值
        ninety_percent_values = sorted_numbers[:percentage_length]
        
        # 计算平均值
        if ninety_percent_values:  # 确保列表不为空
            average = sum(ninety_percent_values) / len(ninety_percent_values)
            return average
        else:
            return None

    # 示例
    # numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result1 = average_of_smaller_90_percent(jueduiweizhiwucha)
    if result1:
        print("车端相对更小的90%的值的平均值是:", result1)
    else:
        print("列表为空或长度不足以计算90%的值")
    result2 = average_of_smaller_90_percent(chongjian_jueduiweizhiwucha)
    if result2:
        print("重建相对更小的90%的值的平均值是:", result2)
    else:
        print("列表为空或长度不足以计算90%的值")
    input()
