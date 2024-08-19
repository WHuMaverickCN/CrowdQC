import os
import cv2 
import numpy as np
from shapely.geometry import Polygon
from .transformation_utils import pixel_to_world_coords

def extract_edges_from_mask(mask):
    """
    提取给定二值化mask的边缘。
    
    参数:
    mask -- 二值化图像，numpy数组，包含0和255值。
    
    返回:
    edges_mask -- 边缘检测结果的mask，numpy数组。
    """
    # 确保mask是单通道的
    if len(mask.shape) == 3:
        raise ValueError("输入的mask必须是单通道图像。")

    # 应用Canny算法检测边缘
    edges_mask = cv2.Canny(mask, threshold1=50, threshold2=150)

    return edges_mask

def trans_1darray_to_grayimage(instance_mask,output_path = 'temp.jpg'):
    # 将一维数组扩展为三维数组，第三维度为1
    img_reshaped_array = instance_mask[:,:,np.newaxis]*64
    
    # 将三维数组在第三维度上重复3次，生成一个三维的彩色图像数组
    img_3d = img_reshaped_array.repeat(3,axis=2)
    
    # 将数组数据类型转换为无符号8位整数
    img_3d = img_3d.astype(np.uint8)

    # 将图像从RGB颜色空间转换为BGR颜色空间
    img_bgr = cv2.cvtColor(img_3d,cv2.COLOR_RGB2BGR)
    
    # 将图像保存到指定路径
    cv2.imwrite(output_path, img_bgr)

def show_pixel_value_count(img):
    import matplotlib.pyplot as plt

    # 假设 image 是一个二维 numpy ndarray
    # image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # 绘制直方图
    plt.hist(img.flatten(), bins=256, range=(0, 255), alpha=0.75)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def semantic_to_instance_segmentation(semantic_mask):
    """
    将语义分割的mask图层转化为实例分割的mask图层。
    
    参数:
        semantic_mask (numpy.ndarray): 语义分割的结果，大小为 (H, W)，
                                       每个像素值表示类别标签。
    
    返回:
        instance_mask (numpy.ndarray): 实例分割的结果，大小为 (H, W)，
                                       每个像素值表示唯一的实例ID。
    """
    # 初始化实例化的mask图层，大小与semantic_mask相同
    instance_mask = np.zeros_like(semantic_mask)

    # 用来跟踪每个类别的实例id
    instance_id_counter = {}

    # 获取所有类别的标签值
    classes = np.unique(semantic_mask)

    # 对每个类别分别处理
    for cls in classes:
        if cls == 0:
            # 0 通常表示背景，可以跳过
            continue

        # 提取当前类别的mask
        class_mask = (semantic_mask == cls).astype(np.uint8)

        # 找出连通组件
        num_labels, labels = cv2.connectedComponents(class_mask)

        # 初始化该类别的实例id计数器
        if cls not in instance_id_counter:
            instance_id_counter[cls] = 1

        # 为每个连通组件分配一个唯一的实例ID
        for i in range(1, num_labels):
            instance_mask[labels == i] = instance_id_counter[cls]
            instance_id_counter[cls] += 1

    # 返回实例分割的结果
    return instance_mask

def if_in_util_feild(pixel_value):
    pixel_to_world_coords(1444,1444)
    return 

def mask_upper_60_percent(image, output_path='mask_image_6.png'):
    height = image.shape[0]
    cutoff = int(height * 0.6)

    # 将上面60%的区域置为黑色
    image[:cutoff, :] = 0
    return image
    # 保存处理后的图像
    cv2.imwrite(output_path, image)
    print(f"处理后的图像已保存为: {output_path}")

def points_set_to_shape(points_in_file):
    for item_in_file in points_in_file:
        print(item_in_file)
    return 

def instance_segmentation_with_cv2(mask, output_path):
    # 使用 connectedComponents 函数找到连通区域
    # connectivity=4表示考虑4邻域（上下左右），默认是8邻域（包括对角）
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    
    # 将 labels 结果缩放到0-255范围，便于保存和可视化
    # 实例标签的范围是 [0, num_labels-1]，将其映射到 [0, 255]
    if num_labels > 1:  # 如果有前景对象
        labels = (labels * (255 / (num_labels - 1))).astype(np.uint8)
    else:
        labels = labels.astype(np.uint8)
    
    # 保存实例化结果
    cv2.imwrite(output_path, labels)
    
    return labels, num_labels
def segment_mask_to_utilized_field_mask(semantic_mask, fixed_param):
    """
    将实例分割的mask图层转化为利用过的区域掩码
    instance_mask是0,1分布的ndarray，表示实例分割的结果
    返回:
        utilized_field_mask (numpy.ndarray): 利用过的区域掩码，大小为 (H, W)，
                                            每个像素值表示是否被利用。
    """
    # 初始化利用过的区域掩码，大小与instance_mask相同（2160,3840）
    scaled_mask = mask_array_to_image(semantic_mask)
    # scaled_mask是resize之后的图片，大小为(2160,3840)，值为1的mask改为255

    # 根据视域范围截断mask
    umask = mask_upper_60_percent(scaled_mask)

    # 对scaled_mask进行实例分割
    instance_mask = semantic_to_instance_segmentation(umask)

    # 使用canny算子进行对截断结果进行边缘检测
    edges_mask = extract_edges_from_mask(umask)
    
    # instance_mask是实例分割之后的结果，像素值为0、1、2、3、4、5、6、7、8、9... 具体数值取决于实例数
    # simplify_edges_mask = simplify_lane_edges(umask)
    # simplify_edges_mask_from_instance = simplify_lane_edges(instance_mask)
    simplify_edges_points = simplify_lane_edges(edges_mask)
    # save_simplified_points_image(simplify_edges_points, edges_mask.shape, 'simpled_mask.png')
    _,instance_edge_points_list = match_simplified_points_to_instances(instance_mask,simplify_edges_points)

    # save_simplified_points_image(simplify_edges_points, edges_mask.shape, 'simpled_mask.png')    
    return instance_edge_points_list
    # instance_umask = semantic_to_instance_segmentation(umask)
    # trans_1darray_to_grayimage(instance_umask,'0815.jpg')
    # 提取实例分割结果的边缘
    edge_mask = extract_edges_from_mask(umask)
    trans_1darray_to_grayimage(edge_mask,'0818.jpg')
    cv2.imwrite('mask_image_.png', instance_mask)
    # if_in_util_feild(scaled_mask)
    # umask = mask_upper_60_percent(scaled_mask)
    # instance_segmentation_with_cv2(umask, 'instance_mask_image.png')
    instance_mask = semantic_to_instance_segmentation(umask)

    return
    utilized_field_mask = np.zeros_like(instance_mask)
    cv2.resize(utilized_field_mask, instance_mask.shape)
    # 获取所有类别的标签值
    classes = np.unique(instance_mask)
    # 对每个类别分别处理
    for cls in classes:
        if cls == 0:
            # 0 通常表示背景，可以跳过
            continue

        # 提取当前类别的mask
        class_mask = (instance_mask == cls).astype(np.uint8)
        # 找出连通组件
        num_labels, labels = cv2.connectedComponents(class_mask)
        # 为每个连通组件分配一个唯一的实例ID
        for i in range(1, num_labels):
            utilized_field_mask[labels == i]

def mask_array_to_image(mask):
    mask_image = (mask * 255).astype('uint8')
    scaled_mask = cv2.resize(mask_image, (3840, 2160), interpolation=cv2.INTER_NEAREST)
    # _, mask_image_inv = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY_INV)
    return scaled_mask
    
    # cv2.imwrite('mask_image.png', scaled_mask)
    # # 显示图像
    # cv2.imshow('Mask Image', scaled_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def simplify_lane_edges(mask, epsilon=5.0):
    """
    简化车道线边缘点集。

    参数:
    mask (ndarray): 输入的二值化车道线边缘mask图像。
    epsilon (float): 近似精度参数，值越大，点集简化得越多。

    返回:
    simplified_lane_points (list): 每个车道线实例的简化边缘点集，格式为[(x1, y1), (x2, y2), ...]。
    """
    # 寻找车道线边缘轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 存储所有车道线实例的简化点集
    simplified_lane_points = []

    for contour in contours:
        # 使用Ramer-Douglas-Peucker算法对轮廓点进行简化
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)
        
        # 将简化后的点集转换为(x, y)格式
        points = [(point[0][0], point[0][1]) for point in approx_curve]
        simplified_lane_points.append(points)
    
    return simplified_lane_points

def save_simplified_points_image(simplified_points, image_shape, save_path):
    """
    将简化后的点集保存为图像，点像素值设置为255，其他点为0。

    参数:
    simplified_points (list): 简化后的点集。
    image_shape (tuple): 输出图像的尺寸，例如(2160, 3840)。
    save_path (str): 保存图像的路径。
    """
    # 创建一个黑色背景图像
    output_image = np.zeros(image_shape, dtype=np.uint8)

    # 遍历每个车道线实例的点集，并在图像上标记这些点
    for points in simplified_points:
        for (x, y) in points:
            output_image[y, x] = 255  # 设置点像素值为255
    
    # 保存图像
    cv2.imwrite(save_path, output_image)


def match_simplified_points_to_instances(maskA, simplified_edge_points):
    """
    将简化的边缘点集对应到maskA中的各个实例。

    参数:
    maskA (ndarray): 原始车道分割mask，包含不同实例。
    simplified_edge_points (list): 包含简化边缘点的列表，每个子列表为(x, y)的点集。

    返回:
    labeled_instance_mask (ndarray): 标注好不同实例的mask图像。
    instance_edge_points_list (list): 区分好实例的边缘点集列表。
    """
    # 获取maskA中的唯一实例标签
    instance_labels = np.unique(maskA)
    instance_labels = instance_labels[instance_labels != 0]  # 排除背景（假设背景为0）

    # 创建一个与maskA同尺寸的全零背景，用于标记不同实例的边缘
    labeled_instance_mask = np.zeros_like(maskA, dtype=np.uint8)

    # 存储分类后的实例边缘点集
    instance_edge_points_list = []

    for instance_label in instance_labels:
        # 提取maskA中当前实例的区域
        instance_mask = (maskA == instance_label).astype(np.uint8)

        # 存储属于该实例的边缘点
        instance_edge_points = []

        # 遍历所有简化后的边缘点集
        for points in simplified_edge_points:
            for (x, y) in points:
                # 检查点(x, y)是否在当前实例区域内
                if instance_mask[y, x] == 1:
                    instance_edge_points.append((x, y))

        # 如果该实例有对应的边缘点
        if instance_edge_points:
            instance_edge_points_list.append(instance_edge_points)
            # 在输出mask中将这些点标记为当前实例的标签值
            for (x, y) in instance_edge_points:
                labeled_instance_mask[y, x] = instance_label
    save_colored_instances(labeled_instance_mask, 'colored_instances.png')
    return labeled_instance_mask, instance_edge_points_list

def save_colored_instances(mask, save_path, thickness=3):
    """
    保存带有不同颜色实例的图像。

    参数:
    mask (ndarray): 标注好不同实例的mask图像。
    save_path (str): 保存图像的路径。
    thickness (int): 描边粗细，默认为3。
    """
    colored_mask = draw_colored_instances(mask, thickness)
    cv2.imwrite(save_path, colored_mask)

def draw_colored_instances(mask, thickness=3):
    """
    将不同实例的边缘用不同颜色加粗显示。

    参数:
    mask (ndarray): 标注好不同实例的mask图像。
    thickness (int): 描边粗细，默认为3。

    返回:
    colored_mask (ndarray): 带有彩色边缘的图像。
    """
    # 获取mask中的唯一实例标签
    instance_labels = np.unique(mask)
    instance_labels = instance_labels[instance_labels != 0]  # 排除背景（假设背景为0）

    # 创建一个彩色图像来显示不同实例
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # 为每个实例分配一种颜色
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in instance_labels]

    # 遍历每个实例，并加粗其边缘
    for i, label in enumerate(instance_labels):
        # 获取当前实例的区域
        instance_mask = (mask == label).astype(np.uint8)

        # 查找边缘轮廓
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在彩色图像上绘制轮廓，使用分配的颜色
        cv2.drawContours(colored_mask, contours, -1, colors[i], thickness)

    return colored_mask
    
def get_quaternion_and_coordinates(df, pic_value, comparison_field="pic_0"):
    # 读取CSV文件
    # 筛选与 pic_value 相同的行
    file_name = os.path.basename(pic_value).replace("_seg_mask.pkl", ".jpg")
    matched_rows = df[df[comparison_field] == file_name]



    if not matched_rows.empty:
        # 获取对应行的 q_w, q_x, q_y, q_z 列
        quaternion_list = matched_rows[['q_w', 'q_x', 'q_y', 'q_z']].values.tolist()

        # 获取 longitude 和 latitude 列
        longitude = matched_rows['longitude'].values.tolist()
        latitude = matched_rows['latitude'].values.tolist()

        return quaternion_list, longitude, latitude
    else:
        return None, (None, None)