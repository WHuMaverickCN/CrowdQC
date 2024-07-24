import os
import cv2
import numpy as np

from src.reconstruction.vanishing_point_utils import get_vanishing_point
from .TransformGround2Image import TransformGround2Image
from .TransformImage2Ground import TransformImage2Ground
from math import pi
class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
    
class EgoviewReconstruction:
    def __init__(self, path_to_data):
        
        self.path_to_data = path_to_data
        # 假设这些参数是从配置或文件中读取的
        self.dist_coeffs_cam0 = np.array([
            0.639999986,
            -0.0069,
            0.00065445,
            0.000117648,
            -0.0057,
            1.003900051,
            0.131899998,
            -0.020199999
        ])

        self.K_cam0 = np.array([[1907.819946, 0, 1903.890015],
                                [0, 1907.670044, 1065.380005],
                                [0, 0, 1]])
        
        self.cameraInfo = Info({
            "focalLengthX": int(1907.819946),   # focal length x
            "focalLengthY": int(1907.670044),   # focal length y
            "opticalCenterX": int(1903.890015), # optical center x
            "opticalCenterY": int(1065.380005), # optical center y
            "cameraHeight": 640,    # camera height in `mm`
            "pitch": -0.030369*(180/pi),    # rotation degree around x
            "yaw": 0.028274*(180/pi),   # rotation degree around y
            "roll": -0.006632*(180/pi),  # rotation degree around z
            "k1":0.639999986,
            "k2":-0.0069,
            "p1":0.00065445,
            "p2":0.000117648,
            "k3":-0.0057,
            "k4":1.003900051,
            "k5":0.131899998,
            "k6":-0.020199999,
            "tx":1.77,
            "ty":0.07,
            "tz":1.34,
            "rx":-0.03037,
            "ry":0.028274,
            "rz":-0.006632
        })

        # tx = 1.77
        # ty = 0.07
        # tz = 1.34
        # rx = -0.03037
        # ry = 0.028274
        # rz = -0.006632

        self.six_dof_data = np.array([self.cameraInfo.tx, 
                                 self.cameraInfo.ty, 
                                 self.cameraInfo.tz, 
                                 self.cameraInfo.rx, 
                                 self.cameraInfo.ry, 
                                 self.cameraInfo.rz])
        self.extrinsic_matrix = self.from_6DoF_to_Rvec(self.six_dof_data)
    def from_6DoF_to_Rvec(self, six_dof_data):
        R_vec = np.array(six_dof_data[3:])
        T_vec = np.array(six_dof_data[:3])
        R_matrix, _ = cv2.Rodrigues(R_vec)
        print("旋转矩阵：")
        print(R_matrix)
        extrinsic_matrix = np.hstack((R_matrix, T_vec.reshape(3, 1)))
        print("外参矩阵：")
        print(extrinsic_matrix)
        return extrinsic_matrix

    def get_undistort_img(self, temp='ca_cam0_sample'):
        image = cv2.imread(f"{temp}.jpg")
        if image is None:
            raise FileNotFoundError(f"Image {temp}.jpg not found.")
        height, width = image.shape[:2]

        # 根据需求选择是否执行逆透视变换（当前未使用）
        # points = [[100, 100], [200, 200]]
        # points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        # undistorted_points = cv2.undistortPoints(points, self.K_cam0, None, R=extrinsic_matrix[:3, :3], P=self.K_cam0)

        # 这里执行图像校正
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.K_cam0, self.dist_coeffs_cam0, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(self.K_cam0, self.dist_coeffs_cam0, None, new_camera_matrix, (w, h), 5)
        undistorted_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        
        # 可以根据需要决定是否展示或保存图像
        # cv2.imshow(temp+'_undist.jpg', undistorted_img)
        # cv2.waitKey()
        cv2.imwrite(temp+'_undist.jpg', undistorted_img)

        return undistorted_img
    
    def inverse_perspective_mapping(self, undistorted_img):
        height = int(undistorted_img.shape[0]) # row y
        width = int(undistorted_img.shape[1]) # col x
        
        ipm_factor = 0.54
        
        self.ipmInfo = Info({
            "inputWidth": width,
            "inputHeight": height,
            "left": 50,
            "right": width-50,
            "top": height*ipm_factor, #选取需要进行逆透视变化的图像范围
            "bottom": height-50
        })
        ipmInfo = self.ipmInfo
        cameraInfo = self.cameraInfo
        R = undistorted_img
        # vpp = GetVanishingPoint(self.cameraInfo)
        vpp = get_vanishing_point(self)
        vp_x = vpp[0][0]
        vp_y = vpp[1][0]
        ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
        uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
                [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)
        xyLimits = TransformImage2Ground(uvLimitsp, self.cameraInfo)
        row1 = xyLimits[0, :]
        row2 = xyLimits[1, :]
        xfMin = min(row1)
        xfMax = max(row1)
        yfMin = min(row2)
        yfMax = max(row2)
        xyRatio = (xfMax - xfMin)/(yfMax - yfMin)
        # target_height = 960
        # target_width = 960
        # outImage = np.zeros((target_height,target_width,4), np.float32)
        outImage = np.zeros((640,960,4), np.float32)
        outImage[:,:,3] = 255
        # 输出图片的大小
        outRow = int(outImage.shape[0])
        outCol = int(outImage.shape[1])
        stepRow = (yfMax - yfMin)/outRow
        stepCol = (xfMax - xfMin)/outCol
        xyGrid = np.zeros((2, outRow*outCol), np.float32)
        y = yfMax-0.5*stepRow
        #构建一个地面网格（天然地平行）
        for i in range(0, outRow):
            x = xfMin+0.5*stepCol
            for j in range(0, outCol):
                xyGrid[0, (i-1)*outCol+j] = x
                xyGrid[1, (i-1)*outCol+j] = y
                x = x + stepCol
            y = y - stepRow

        #将地面格网转回图像
        # TransformGround2Image
        uvGrid = TransformGround2Image(xyGrid, cameraInfo)
        # mean value of the image
        means = np.mean(R)/255
        RR = R.astype(float)/255
        for i in range(0, outRow):
            # print(i,outRow)
            for j in range(0, outCol):
                #得到了每个点在图像中的u,v坐标
                ui = uvGrid[0, i*outCol+j]
                vi = uvGrid[1, i*outCol+j]
                #print(ui, vi)
                if ui < ipmInfo.left or ui > ipmInfo.right or vi < ipmInfo.top or vi > ipmInfo.bottom:
                    outImage[i, j] = 0.0
                else:
                    x1 = np.int32(ui)
                    x2 = np.int32(ui+0.5)
                    y1 = np.int32(vi)
                    y2 = np.int32(vi+0.5)
                    x = ui-float(x1)
                    y = vi-float(y1)
                    # print(ui, vi)
                    #双线性插值
                    outImage[i, j, 0] = float(RR[y1, x1, 0])*(1-x)*(1-y)+float(RR[y1, x2, 0])*x*(1-y)+float(RR[y2, x1, 0])*(1-x)*y+float(RR[y2, x2, 0])*x*y
                    outImage[i, j, 1] = float(RR[y1, x1, 1])*(1-x)*(1-y)+float(RR[y1, x2, 1])*x*(1-y)+float(RR[y2, x1, 1])*(1-x)*y+float(RR[y2, x2, 1])*x*y
                    outImage[i, j, 2] = float(RR[y1, x1, 2])*(1-x)*(1-y)+float(RR[y1, x2, 2])*x*(1-y)+float(RR[y2, x1, 2])*(1-x)*y+float(RR[y2, x2, 2])*x*y
        outImage[-1,:] = 0.0 
        # show the result

        outImage = outImage * 255
        print("finished")
        cv2.imwrite("dist_"+ipm_factor.__str__()+"_inverse_perspective_mapping.jpg", outImage)



