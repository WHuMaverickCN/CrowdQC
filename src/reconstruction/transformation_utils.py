import cv2
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as sciR
np.set_printoptions(precision=6, suppress=True)
from math import pi

CHANGAN_RAW_RPH = {
    "tx":1.77,
    "ty":0.07,
    "tz":1.34,
    "rx":-0.03037,
    "ry":0.028274+pi/2,
    "rz":-0.006632-pi/2
}

ARGOVERSE_RAW_QUAT = {
    "qw":0.502809,
    "qx":-0.499689,
    "qy":0.500147,
    "qz":-0.497340
}

ARGOVERSE_RAW_TRANS = {
    "tx":1.631216,
    "ty":-0.000779,
    "tz":1.432780
}

SAMPLE_POINTS_IN_PIXEL = {
    "p1_left_a":(1281, 1581),
    "p2_left_b":(1442, 1463),
    "p3_left_c":(1551, 1260),
    "p4_left_d":(1479, 1172),
    "p5_right_e":(1734, 1163),
    "p6_right_f":(2035, 1246),
    "p7_right_g":(2292, 1375),
    "p8_right_h":(2608, 1555)
    }

def from_rotation_vector_to_rotation_matrix(rvec):
    R = cv2.Rodrigues(rvec)[0]

def from_quanternion_to_euler(quanternion):
    rot = sciR.from_quat(quanternion)
    euler = rot.as_euler('xyz', degrees=True)
    print(euler)

def pose_to_extrinsic(pose_matrix):
    """
    将相机位姿矩阵转化为外参矩阵。

    参数：
    pose_matrix (numpy.ndarray): 3x4 的相机位姿矩阵 [R | t]

    返回：
    extrinsic_matrix (numpy.ndarray): 4x4 的外参矩阵
    """
    # 检查输入矩阵的形状
    if pose_matrix.shape != (3, 4):
        raise ValueError("位姿矩阵的形状应为 3x4")

    # 构建 4x4 外参矩阵
    extrinsic_matrix = np.eye(4)  # 创建一个单位矩阵
    extrinsic_matrix[:3, :4] = pose_matrix  # 将 3x4 位姿矩阵赋值给外参矩阵的前 3 行

    return extrinsic_matrix

def triangulation():
    pass

from_rotation_vector_to_rotation_matrix(np.array(
    [CHANGAN_RAW_RPH["rx"],\
    CHANGAN_RAW_RPH["ry"],\
    CHANGAN_RAW_RPH["rz"]])
    )

def extrinsic_to_pose(R, t):
    """
    将外参矩阵转换为位姿矩阵。

    参数：
    R (numpy.ndarray): 3x3 的旋转矩阵
    t (numpy.ndarray): 3x1 的平移向量

    返回：
    pose_matrix (numpy.ndarray): 4x4 的位姿矩阵
    """
    # 计算旋转矩阵的转置
    R_inv = R.T
    
    # 计算新的平移向量
    t_inv = -R_inv @ t

    # 构建 4x4 位姿矩阵
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R_inv
    pose_matrix[:3, 3] = t_inv.flatten()

    return pose_matrix

def camera_pose_to_extrinsic(_rotation_matrix:np.array, _translation_vector:np.array):
    """
    将相机位姿矩阵转换为外参矩阵。
    """

    # 计算旋转矩阵的逆（转置）
    rotation_matrix_inv = _rotation_matrix.T

    # 计算平移向量的逆：-R^T * t
    translation_vector_inv = -np.dot(rotation_matrix_inv, _translation_vector)

    extrinsic_matrix = np.hstack((rotation_matrix_inv, translation_vector_inv.reshape(3, 1)))
    return extrinsic_matrix

def pose_to_extrinsic(pose_matrix):
    """
    将位姿矩阵转换为外参矩阵。

    参数：
    pose_matrix (numpy.ndarray): 4x4 的位姿矩阵

    返回：
    R_extrinsic (numpy.ndarray): 3x3 的旋转矩阵
    t_extrinsic (numpy.ndarray): 3x1 的平移向量
    """
    # 提取位姿矩阵的旋转部分和平移部分
    R_pose = pose_matrix[:3, :3]
    t_pose = pose_matrix[:3, 3]

    # 计算旋转矩阵的转置（逆矩阵）
    R_extrinsic = R_pose.T

    # 计算外参平移向量
    t_extrinsic = -R_extrinsic @ t_pose

    return R_extrinsic, t_extrinsic

if __name__ == "__main__":
    # 通过四元数，构建相机位姿矩阵
    quanternion = [
        ARGOVERSE_RAW_QUAT["qx"],
        ARGOVERSE_RAW_QUAT["qy"],
        ARGOVERSE_RAW_QUAT["qz"],
        ARGOVERSE_RAW_QUAT["qw"],
        ]
    
    trans = [
        ARGOVERSE_RAW_TRANS["tx"],
        ARGOVERSE_RAW_TRANS["ty"],
        ARGOVERSE_RAW_TRANS["tz"]
    ]
    print("**********Argo 相机参数计算结果***********")
    rot = sciR.from_quat(quanternion)
    print(rot.as_euler('xyz', degrees=True))
    print(rot.as_euler('zxy', degrees=True))
    print(rot.as_euler('zyx', degrees=True))
    rot_mat = rot.as_matrix()

    rot_vec = rot.as_rotvec(degrees=True)
    print(f"根据四元数计算的旋转向量结果为：\n{rot_vec}")
    trans_vec = np.array(trans)
    extrinsic_matrix_rt = camera_pose_to_extrinsic(rot_mat,trans_vec)
    test_pose = extrinsic_matrix_rt @ (np.array([[0,0,0,1]]).T)

    inv_r = sciR.from_matrix(extrinsic_matrix_rt[:3,:3])
    print(f"根据位姿矩阵求逆的逆向旋转矩阵结果为：\n{inv_r.as_rotvec(degrees=True)}")
    print(test_pose)
    
    print("**********Changan 相机参数计算结果***********")

    # 通过长安提供的相机位姿参数构建旋转，声明的旋转顺序为 zyx
    rot_ca = sciR.from_euler('zyx',[
                                CHANGAN_RAW_RPH["rz"],\
                                CHANGAN_RAW_RPH["ry"],\
                                CHANGAN_RAW_RPH["rx"]
                                ])

    rot_vec_ca = rot_ca.as_rotvec(degrees=False)
    rot_mat_ca = rot_ca.as_matrix()
    print(f"根据四元数计算的旋转向量结果为：\n{rot_vec_ca}")
    print(rot_ca.as_euler('xyz', degrees=False))
    print(rot_ca.as_euler('zyx', degrees=False))
    print(rot_ca.as_euler('zxy', degrees=False))

    print(rot_ca.as_quat())
    trans_ca = [
        CHANGAN_RAW_RPH["tx"],
        CHANGAN_RAW_RPH["ty"],
        CHANGAN_RAW_RPH["tz"]
    ]
    trans_vec_ca = np.array(trans_ca)
    extrinsic_matrix_rt_ca = camera_pose_to_extrinsic(rot_mat_ca,trans_vec_ca)
    rot_mat = rot.as_matrix()
