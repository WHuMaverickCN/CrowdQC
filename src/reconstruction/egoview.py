import os
import cv2
import geojson
import numpy as np
from pathlib import Path
from math import pi
from scipy.spatial.distance import euclidean
import glob

from src.reconstruction.vanishing_point_utils import get_vanishing_point
from .transformation_utils import *
from .landmark_utils import *
from .TransformGround2Image import TransformGround2Image
from .TransformImage2Ground import TransformImage2Ground

from ..io import input

np.set_printoptions(precision=6, suppress=True)

DEFAULT_MASK_FILE_PATH = "output/4/array_mask"
DEFAULT_PROCESSED_DAT_PATH = "output/4"
DEFAULT_PROCESSED_DAT_LOC_PATH = "output/4/loc2vis.csv"

class Info(object):
    def __init__(self, dct):
        self.dct = dct

    def __getattr__(self, name):
        return self.dct[name]
    
class EgoviewReconstruction:
    def __init__(self, path_to_data=""):
        
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
        # self.R = 
        self.cameraInfo = Info({
            "focalLengthX": int(1907.819946),   # focal length x
            "focalLengthY": int(1907.670044),   # focal length y
            "opticalCenterX": int(1903.890015), # optical center x
            "opticalCenterY": int(1065.380005), # optical center y
            "cameraHeight": 1340,    # camera height in `mm`
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
            # "rx":-0.03037,
            # "ry":0.028274,
            # "rz":-0.006632
            # "rx":-0.03037-pi/2,
            # "ry":0.028274,
            # "rz":-0.006632-pi/2
            "rx":-0.03037,
            "ry":0.028274+pi/2,
            "rz":-0.006632-pi/2
        })


        self.six_dof_data = np.array([self.cameraInfo.tx, 
                                 self.cameraInfo.ty, 
                                 self.cameraInfo.tz, 
                                 self.cameraInfo.rx, 
                                 self.cameraInfo.ry, 
                                 self.cameraInfo.rz])
        
        rot_mat_ca,trans_vec_ca = self.get_camera_pose()
        self.extrinsic_matrix = camera_pose_to_extrinsic(rot_mat_ca,trans_vec_ca)
        print("extrinsic_matrix:\n",self.extrinsic_matrix)
        self.extrinsic_rotation_matrix = self.extrinsic_matrix[:3,:3]
        self.extrinsic_transaction_vector = self.extrinsic_matrix[:3,3]
        # self.extrinsic_rotation_matrix = 
        # self.R_vec,self.T_vec,self.extrinsic_matrix = self.from_6DoF_to_Rvec(self.six_dof_data)
    def get_camera_pose(self):
        # rot_ca = sciR.from_euler('zyx',[
        #                         self.cameraInfo["rz"],\
        #                         self.cameraInfo["ry"],\
        #                         self.cameraInfo["rx"]
        #                         ])
        # trans_ca = [
        #     self.cameraInfo["tx"],
        #     self.cameraInfo["ty"],
        #     self.cameraInfo["tz"]
        # ]
        rot_ca = sciR.from_euler('zyx',[
                                self.cameraInfo.rz,\
                                self.cameraInfo.ry,\
                                self.cameraInfo.rx
                                ])
        trans_ca = [
            self.cameraInfo.tx,
            self.cameraInfo.ty,
            self.cameraInfo.tz
        ]
        trans_vec_ca = np.array(trans_ca)
        rot_mat_ca = rot_ca.as_matrix()
        self.pose_rotation_matrix = rot_mat_ca
        self.pose_transaction_vector = trans_vec_ca
        pose_matrix = np.hstack((rot_mat_ca, trans_vec_ca.reshape(3, 1)))
        return rot_mat_ca,trans_vec_ca
    
    def from_6DoF_to_Rvec(self, six_dof_data):
        R_vec = np.array(six_dof_data[3:])
        T_vec = np.array(six_dof_data[:3])

        # Converts a rotation matrix to a rotation vector or vice versa.
        print(R_vec)
        R_matrix, _ = cv2.Rodrigues(R_vec)
        print("相机坐标系相对于世界坐标系的旋转矩阵：")
        print(R_matrix)
        pose_matrix = np.hstack((R_matrix, T_vec.reshape(3, 1)))
        print("相机位姿矩阵：")
        print(pose_matrix)

        # 由计算外参矩阵
        # extrinsic_matrix = pose_to_extrinsic(pose_matrix)
        extrinsic_matrix_rt = camera_pose_to_extrinsic(R_matrix,T_vec)
        print(f"自车坐标系原点在相机坐标系的x，y，z向量坐标：\n{extrinsic_matrix_rt @ (np.array([[0,0,0,1]]).T)}")
        return R_matrix,T_vec,extrinsic_matrix_rt#pose_matrix
    
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

    @staticmethod
    def pixel_to_world(u, v, K, dist_coeffs, R, T, vehicle_height):
        # 步骤1：使用内参矩阵和畸变系数对像素坐标进行去畸变，得到校正后的像素坐标。
        undistorted_points = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float32), K, dist_coeffs)

        # 步骤2：将校正后的像素坐标转换为归一化相机坐标。
        # 归一化坐标即是从像素坐标除以焦距并减去主点偏移。
        normalized_camera_coords = np.array([undistorted_points[0][0][0], undistorted_points[0][0][1], 1.0])

        # 步骤3：将归一化相机坐标通过逆旋转矩阵转换为相机坐标。
        # 使用相机旋转矩阵的逆（转置）将方向从相机坐标变换到世界坐标。
        cam_to_world_rotation = np.linalg.inv(R)
        cam_coords = cam_to_world_rotation.dot(normalized_camera_coords)

        # 步骤4：假设地面水平（Z=0），计算相机坐标系中某一点与地面的交点。
        # 这里我们假设相机高度已知，并使用此来找到尺度因子。
        scale_factor = vehicle_height / cam_coords[1]  # 取Y轴高度除以相机高度

        # 计算世界坐标，使用尺度因子。
        world_coords = scale_factor * cam_coords

        # 加上平移向量，得到最终的世界坐标。
        world_coords += T

        # 返回世界坐标，Z坐标在此为0。
        return world_coords
        
    @staticmethod
    def pixel_to_world_new(u, v, camera_matrix, dist_coeffs, R, T, vehicle_height):
        # 将像素坐标转化为图像坐标
        uv = np.array([[u, v]], dtype=np.float32)

        # 去畸变并归一化
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs, P=camera_matrix)
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs)
        # tips此处P参数如果不赋值，该该函数返回的结果

        # 归一化相机坐标系
        u_n, v_n = uv_undistorted[0][0]

        # 形成归一化的相机坐标
        normalized_camera_coords = np.array([u_n, v_n, 1.0])

        # 计算比例因子，假设平面高度Z=0,即每个像素换算为世界坐标系对应的距离，以米为单位
        # 这里，直接利用外参进行变换之前计算比例因子是关键步骤
        scale_factor = vehicle_height / (R[2, 0] * normalized_camera_coords[0] + 
                                        R[2, 1] * normalized_camera_coords[1] + 
                                        R[2, 2])
        # scale_factor = vehicle_height / np.dot(R[2], normalized_camera_coords)

        # 乘以比例因子得到相机坐标系中的点
        camera_coords_scaled = normalized_camera_coords * scale_factor

        # 应用外参变换，将相机坐标系坐标转换到世界坐标系
        world_coords = np.dot(R, camera_coords_scaled) + T

        # 返回世界坐标
        return camera_coords_scaled,world_coords
        # return camera_coords_scaled,world_coords[:2]  # 通常假设z=0，返回x和y坐标
    
    @staticmethod
    def image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height):
        # 去畸变像素点（如果使用的是畸变图像点）
        uv = np.array([[[u, v]]], dtype=np.float32)
        uv_undistorted = cv2.undistortPoints(uv, camera_matrix, dist_coeffs, P=camera_matrix)
        undistorted_points = cv2.undistortPoints(uv, camera_matrix, dist_coeffs,P=camera_matrix)

        # 将像素坐标转换为归一化相机坐标
        u_n = uv_undistorted[0, 0, 0]
        v_n = uv_undistorted[0, 0, 1]

        # 转换为齐次坐标
        normalized_camera_coords = np.array([u_n, v_n, 1.0])

        # 假设地面为平面，求比例因子使点落在 Z = 0 上
        # 如果知道 vehicle_height（相机到地面的高度），用它来求比例因子
        scale_factor = vehicle_height / np.dot(R[1], normalized_camera_coords)

        # 计算在相机坐标系中的3D位置
        point_camera = scale_factor * normalized_camera_coords

        # 将相机坐标转换为车辆坐标
        point_vehicle = np.dot(R, point_camera) + t

        return point_camera,point_vehicle
    
    def transation_instance(self,
                            DEFAULT_MASK_FILE_PATH,
                            DEFAULT_PROCESSED_DAT_LOC_PATH,
                            traj_correction_dict,
                            output_file_name,
                            default_output_path = "reconstruction_output_0919/"):
        # Example usage
        # camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        camera_matrix = self.K_cam0
        # dist_coeffs = np.zeros(5)  # assuming no distortion
        dist_coeffs = self.dist_coeffs_cam0
        # R = np.eye(3)  # replace with actual rotation matrix from camera to vehicle
        # t = np.array([0, 0, vehicle_height])  # replace with actual translation vector
        R = self.extrinsic_rotation_matrix  # replace with actual rotation matrix from camera to vehicle
        t = self.extrinsic_transaction_vector  # replace with actual translation vector
        vehicle_height = t[-1]  # example camera height from ground

        format_str = "Camera matrix:\n{}\nDistortion coefficients: {}"
        print(format_str.format(camera_matrix, dist_coeffs))

        mask_path = str(Path(DEFAULT_MASK_FILE_PATH).absolute())
        if os.path.isdir(mask_path):
            files = sorted(glob.glob(os.path.join(mask_path, '*.*'))) 
        else:
            return
        fixed_param = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "R": self.pose_rotation_matrix,
            "t": self.pose_transaction_vector,
            "vehicle_height": vehicle_height
        }
        features = []

        #读取包含定位数据
        _loc_path = str(Path(DEFAULT_PROCESSED_DAT_LOC_PATH).absolute())
        loc_data_df = input.read_loc_data(_loc_path)
        
        for file in files:
            # 从定位数据中获取四元数、世界坐标、欧拉角
            quat,world_coords_from_ins,rph = get_quaternion_and_coordinates(loc_data_df,
                                                                            file,
                                                                            "pic_0")
            
            rot_param = [quat,rph]
            print(rph)
            if quat == None and world_coords_from_ins==None:
                continue
            print(file)
            def find_closest_point(target_point, dataframe):
                """
                找到与目标点距离最近的DataFrame中的点。

                参数:
                target_point : tuple
                    目标点的经纬度 (latitude, longitude)。
                dataframe : pd.DataFrame
                    包含经纬度数据的DataFrame。

                返回:
                pd.Series
                    距离目标点最近的DataFrame中的行。
                """
                # 计算目标点与DataFrame中每个点的欧几里得距离
                distances = dataframe.apply(lambda row: euclidean((row['new_longitude'], row['new_latitude']), target_point), axis=1)
                
                # 找到距离最近的点的索引
                closest_index = distances.idxmin()
                
                # 返回距离最近的那行数据
                return dataframe.loc[closest_index]
            
            closest_point = find_closest_point(world_coords_from_ins, traj_correction_dict)
            # print(closest_point)
            # print(world_coords_from_ins)
            world_coords_from_ins=(closest_point.iloc[5],closest_point.iloc[4])
            # print(world_coords_from_ins)
            sem_seg = read_segmentation_mask_from_pickle(file)
            instance_edge_points_list = segment_mask_to_utilized_field_mask(sem_seg,
                                                                            fixed_param)
            # ins_seg = semantic_to_instance_segmentation(sem_seg)
            # print(instance_edge_points_list)
            
            for _edge_points_for_one_instance in instance_edge_points_list:
                # print(len(instance_edge_points_list))
                coordinates = []
                for pixel in _edge_points_for_one_instance:
                    point_camera, point_vehicle = self.pixel_to_world_new(pixel[0],
                                                                        pixel[1], 
                                                                        camera_matrix, 
                                                                        dist_coeffs, 
                                                                        self.pose_rotation_matrix, 
                                                                        self.pose_transaction_vector, 
                                                                        vehicle_height)
                    # 将自车坐标转化为世界坐标
                    point_world = trans_ego_to_world_coord(point_vehicle = point_vehicle, 
                                                           quanternion = rot_param, 
                                                           geographical_coords=world_coords_from_ins)
                    # print(point_world)
                    # q,world_coords_from_ins
                    # coordinates.append((point_vehicle[0], point_vehicle[1],0))  # 添加点坐标到坐标列表中
                    # coordinates.append((point_world[0], point_world[1],0.0))  # 添加点坐标到坐标列表中
                    coordinates.append((point_world[0], point_world[1]))  # 添加点坐标到坐标列表中


                    # print(
                    #     "point_camera:",\
                    #     point_camera,\
                    #     "\npoint_vehicle:",\
                    #     point_vehicle,"\n",
                    #     "point_world:",\
                    #     point_world)
                # coordinates = simplify_polygon(coordinates)
                feature = geojson.Feature(
                    geometry=geojson.Polygon([coordinates]),  # 使用Polygon表示该实例的边界
                    properties={"file_name": os.path.basename(file)}  # 将文件名作为属性
                )
                features.append(feature)
        feature_collection = geojson.FeatureCollection(features)
        if os.path.exists(default_output_path)==False:
            os.mkdir(default_output_path)
        with open(os.path.join(default_output_path, output_file_name+".geojson"), 'w') as f:
            geojson.dump(feature_collection, f)
            # with open('intent_output_world_065.geojson', 'w') as f:
            #     geojson.dump(feature_collection, f)   
        return
        # 示例使用，输入参数需根据实际摄像机参数调整
        from .transformation_utils import SAMPLE_POINTS_IN_PIXEL as samples
        for item in samples.items():
            u, v = item[1]
            print(item[0],u, v, "")
            point_camera, point_vehicle = self.pixel_to_world_new(u, 
                                                                  v,
                                                                  camera_matrix, 
                                                                  dist_coeffs, 
                                                                  self.pose_rotation_matrix, 
                                                                  self.pose_transaction_vector, 
                                                                  vehicle_height)
            # point_camera, point_vehicle = self.image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
            print("point_camera:",point_camera,"\npoint_vehicle:", point_vehicle,"\n")
        u, v = 1442, 1463  # example pixel coordinates
        vehicle_height = self.cameraInfo.tz # 车辆高度（单位：米）
        point_vehicle = self.image_to_vehicle(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        print("Vehicle coordinates:", point_vehicle)

        # 示例使用，输入参数需根据实际摄像机参数调整
       

        # 计算世界坐标
        world_point = self.pixel_to_world(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        world_point_new = self.pixel_to_world_new(u, v, camera_matrix, dist_coeffs, R, t, vehicle_height)
        print("世界坐标:", world_point)

    def batch_ego_reconstruction(self,target_dir="output"):
        """
        批量进行ego数据重建
        Args:
            target_dir (str, optional): 目标文件夹路径，默认为"output"。

        Returns:
            None
        """
        files = os.listdir(target_dir)
        for file in files:
            print(file)
            temp_data_root = os.path.join(target_dir, file)
            temp_mask_path = os.path.join(temp_data_root, "array_mask")
            temp_loc_file = os.path.join(temp_data_root, "loc2vis.csv")
            
            temp_traj_file = glob.glob(os.path.join(temp_data_root, "trajectory*"))
            if len(temp_traj_file) > 0:
                temp_traj_file = temp_traj_file[0]
            else:
                temp_traj_file = ''

            traj_correction_dict = match_trajectory_to_insdata(temp_traj_file,temp_loc_file)
            self.transation_instance(
                DEFAULT_MASK_FILE_PATH = temp_mask_path,
                DEFAULT_PROCESSED_DAT_LOC_PATH = temp_loc_file,
                traj_correction_dict = traj_correction_dict,
                output_file_name=file
            )
    def inverse_perspective_mapping(self, undistorted_img):
        height = int(undistorted_img.shape[0]) # row y
        width = int(undistorted_img.shape[1]) # col x
        
        # 定义IPM参数，定义逆透视变换需要选取的图像范围
        ipm_factor = 0.6
        
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
        # 1.获取逆透视变换之后的特征点
        p_set = []
        self.transation_instance()

        # 2.获取逆透视变换之后的图像
        vpp = get_vanishing_point(self)
        vp_x = vpp[0][0]
        vp_y = vpp[1][0]
        print(ipmInfo.top)
        ipmInfo.top = float(max(int(vp_y), ipmInfo.top))
        print(ipmInfo.top)
        # uvLimitsp = np.array([[vp_x, ipmInfo.right, ipmInfo.left, vp_x],
        #         [ipmInfo.top, ipmInfo.top, ipmInfo.top, ipmInfo.bottom]], np.float32)
        uvLimitsp = np.array([[ipmInfo.left, ipmInfo.right, ipmInfo.right, ipmInfo.left],
                [ipmInfo.top, ipmInfo.top, ipmInfo.bottom, ipmInfo.bottom]], np.float32)
        xyLimits = TransformImage2Ground(uvLimitsp, self.cameraInfo)

        print(xyLimits)
        row1 = xyLimits[0, :]
        row2 = xyLimits[1, :]
        xfMin = min(row1)
        xfMax = max(row1)
        yfMin = min(row2)
        yfMax = max(row2)
        xyRatio = (xfMax - xfMin)/(yfMax - yfMin)
        target_height = 640
        target_width = 960
        outImage = np.zeros((target_height,target_width,4), np.float32)
        # outImage = np.zeros((640,960,4), np.float32)
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
        
        cv2.imwrite("dist_"+ipm_factor.__str__()+"_inverse_perspective_mapping_1340.jpg", outImage)

def read_segmentation_mask_from_pickle(mask_path):
    """
    从pickle文件中读取语义分割的mask图层。
    
    参数:
        mask_path (str): 语义分割的mask文件路径。
    
    返回:
        semantic_mask (numpy.ndarray): 语义分割的结果，大小为 (H, W)，
                                       每个像素值表示类别标签。
    """
    semantic_mask = input.read_mask_data(mask_path)
    return semantic_mask

