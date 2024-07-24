import numpy as np
import cv2

def get_vanishing_point(reconstruction):
    def get_vanishing_point(camera_matrix, rotation_matrix, yaw, pitch):
        # 计算视平面方向向量
        d = np.array([
            [np.sin(np.radians(yaw)) / np.cos(np.radians(pitch))],
            [np.cos(np.radians(yaw)) / np.cos(np.radians(pitch))],
            [0]
        ])

        # 构建航向角旋转矩阵
        R_yaw = np.array([
            [np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
            [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
            [0, 0, 1]
        ])

        # 构建俯仰角旋转矩阵
        R_pitch = np.array([
            [1, 0, 0],
            [0, -np.sin(np.radians(pitch)), -np.cos(np.radians(pitch))],
            [0, np.cos(np.radians(pitch)), -np.sin(np.radians(pitch))]
        ])

        # 计算总旋转矩阵
        transform = np.dot(R_yaw, R_pitch)
        transform = np.dot(camera_matrix, transform)

        # 计算灭点
        vp = np.dot(transform, d)

        # 归一化灭点
        vp = vp / vp[2]

        return vp[:2]

    # # 相机内参矩阵
    # camera_matrix = np.array([
    #     [1907.819946, 0, 1065.380005],
    #     [0, 1903.890015, 639.999986],
    #     [0, 0, 1]
    # ])

    # 相机内参矩阵
    camera_matrix = np.array([
        [reconstruction.cameraInfo.focalLengthX, 0, reconstruction.cameraInfo.opticalCenterX],
        [0, reconstruction.cameraInfo.focalLengthY, reconstruction.cameraInfo.opticalCenterY],
        [0, 0, 1]
    ])

    R_vec = np.array(reconstruction.six_dof_data[3:])
    T_vec = np.array(reconstruction.six_dof_data[:3])
    R_matrix, _ = cv2.Rodrigues(R_vec)
    # 假设的相机旋转矩阵 (从外部传入)
    R_matrix
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # 相机的yaw和pitch角度 (假设值)
    yaw = reconstruction.cameraInfo.yaw
    pitch =  reconstruction.cameraInfo.pitch

    # 计算灭点
    vanishing_point = get_vanishing_point(camera_matrix, rotation_matrix, yaw, pitch)
    # print("Vanishing Point:", vanishing_point)
    return vanishing_point

