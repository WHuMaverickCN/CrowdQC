import os
import cv2
import numpy as np

class EgoviewReconstruction():
    def __init__(self, path_to_data):
        self.path = os.path.join(os.getcwd(), 'src', 'reconstruction')
        self.path_to_data = path_to_data

    @staticmethod
    def ipm(temp=''):
        class Info(object):
            def __init__(self, dct):
                self.dct = dct

            def __getattr__(self, name):
                return self.dct[name]
        
        # 畸变系数
        dist_coeffs_cam0 = np.array([
        0.639999986,
        -0.0069,
        0.00065445,
        0.000117648,
        -0.0057,
        1.003900051,
        0.131899998,
        -0.020199999
        ])

        K_cam0 = np.array([[1907.819946, 0, 1903.890015],
                    [0, 1907.670044, 1065.380005],
                    [0, 0, 1]])

        
        # 读取图像
        temp = 'ca_cam0_sample'
        image = cv2.imread(temp+'.jpg')
        height = int(image.shape[0]) # row y
        width = int(image.shape[1]) # col x
        ipmInfo = Info({
        "inputWidth": width,
        "inputHeight": height,
        "left": 50,
        "right": width-50,
        "top": height*0.6,
        "bottom": height-50
        })
        oi = image
        koi = K_cam0
        # # 校正图像
        # undistorted_image = cv2.undistort(image, oi,K_cam0, dist_coeffs_cam0,koi)
        # cv2.imshow(temp+'_undist.jpg', undistorted_image)
        # # 保存校正后的图像
        # cv2.imwrite(temp+'_undist.jpg', undistorted_image)
        tx = 1.77
        ty = 0.07
        tz = 1.34
        rx = -0.03037
        ry = 0.028274
        rz = -0.006632

        six_dof_data = np.array([tx, ty, tz, rx, ry, rz])

        def from_6DoF_to_Rvec(six_dof_data):
            R_vec = np.array(six_dof_data[:3])
            T_vec = np.array(six_dof_data[3:])
            R_matrix, _ = cv2.Rodrigues(R_vec)
            print("旋转矩阵：")
            print(R_matrix)
            extrinsic_matrix = np.hstack((R_matrix, T_vec.reshape(3, 1)))
            print("外参矩阵：")
            print(extrinsic_matrix)

            return extrinsic_matrix

        extrinsic_matrix = from_6DoF_to_Rvec(six_dof_data)

        points = [[100, 100], [200, 200]]
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # 进行逆透视变换，此处根据点获取变换关系
        undistorted_points = cv2.undistortPoints(points, K_cam0, None, R=extrinsic_matrix[:3, :3], P=K_cam0)

        h, w = image.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K_cam0, dist_coeffs_cam0, (w, h), 1, (w, h))

        mapx, mapy = cv2.initUndistortRectifyMap(K_cam0, dist_coeffs_cam0, None, new_camera_matrix, (w, h), 5)
        undistorted_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imshow(temp+'_undist.jpg', undistorted_img)
        cv2.imwrite(temp+'_undist.jpg', undistorted_img)

EgoviewReconstruction.ipm()