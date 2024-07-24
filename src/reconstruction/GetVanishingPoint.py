import numpy as np
from math import cos, sin, pi
# 得到灭点在图像上的位置
def GetVanishingPoint(cameraInfo):
    vpp = [[sin(cameraInfo.yaw*pi/180)/cos(cameraInfo.pitch*pi/180)],
        [cos(cameraInfo.yaw*pi/180)/cos(cameraInfo.pitch*pi/180)],
        [0]]
    # 此处均为角度制转弧度制
    # yaw角旋转矩阵    
    tyawp = [[cos(cameraInfo.yaw*pi/180), -sin(cameraInfo.yaw*pi/180), 0],
            [sin(cameraInfo.yaw*pi/180), cos(cameraInfo.yaw*pi/180), 0],
            [0, 0, 1]]
    # pitch角旋转矩阵  			
    tpitchp = [[1, 0, 0],
            [0, -sin(cameraInfo.pitch*pi/180), -cos(cameraInfo.pitch*pi/180)],
            [0, cos(cameraInfo.pitch*pi/180), -sin(cameraInfo.pitch*pi/180)]]
    # 相机内参变换矩阵
    t1p = [[cameraInfo.focalLengthX, 0, cameraInfo.opticalCenterX],
		  [0, cameraInfo.focalLengthY, cameraInfo.opticalCenterY],
		  [0, 0, 1]]

    transform = np.array(tyawp).dot(np.array(tpitchp))
    transform = np.array(t1p).dot(transform)
    vp = transform.dot(np.array(vpp))
    
    return vp