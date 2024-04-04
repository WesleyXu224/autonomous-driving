import open3d as o3d 
from open3d import geometry
import torch
import numpy as np

def _draw_points(points, 
                 vis, 
                 points_size=2, 
                 point_color=(0.5, 0.5, 0.5), 
                 mode='xyz'): 
    # 获取 Open3D Visualizer 的渲染设置，更改点云尺寸 
    vis.get_render_option().point_size = points_size  
    if isinstance(points, torch.Tensor): 
        points = points.cpu().numpy() 
 
    points = points.copy() 
    # 构建 Open3D 中提供的 gemoetry 点云类 
    pcd = geometry.PointCloud() 
    # 如果传入的点云 points 只包含位置信息 xyz 
    # 根据指定的 point_color 为每个点云赋予相同的颜色 
    if mode == 'xyz': 
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) 
        points_colors = np.tile(np.array(point_color), (points.shape[0], 1)) 
    # 如果传入的点云 points 本身还包含颜色信息（通常是分类预测结果或者标签） 
    # 直接从 points 获取点云的颜色信息 
    elif mode == 'xyzrgb': 
        pcd.points = o3d.utility.Vector3dVector(points[:, :3]) 
        points_colors = points[:, 3:6] 
        # 将颜色归一化到 [0, 1] 用于 Open3D 绘制 
        if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all(): 
            points_colors /= 255.0 
    else: 
        raise NotImplementedError 
 
    # 为点云着色 
    pcd.colors = o3d.utility.Vector3dVector(points_colors) 
    # 将点云加入到 Visualizer 中 
    vis.add_geometry(pcd) 
 
    return pcd, points_colors 

def _draw_bboxes(bbox3d, 
                 vis, 
                 points_colors, 
                 pcd=None, 
                 bbox_color=(0, 1, 0), 
                 points_in_box_color=(1, 0, 0), 
                 rot_axis=2, 
                 center_mode='lidar_bottom', 
                 mode='xyz'):
    in_box_color = np.array(points_in_box_color) 
    for i in range(len(bbox3d)): 
        center = bbox3d[i, 0:3] 
        dim = bbox3d[i, 3:6] 
        yaw = np.zeros(3) 
        # fix a bug that it not need become negative
        yaw[rot_axis] = bbox3d[i, 6]
        # 在 Open3D 中需要将 yaw 朝向角转为旋转矩阵 
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw) 
        # 底部中心点转为几何中心店 
        # if center_mode == 'lidar_bottom': 
        # center[rot_axis] += dim[ 
        #         rot_axis] / 2   
        # elif center_mode == 'camera_bottom': 
        # center[rot_axis] -= dim[ 
        #     rot_axis] / 2   
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim) 
 
        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color) 
        # 在 Visualizer 中绘制 Box 
        vis.add_geometry(line_set) 
 
        # 更改 3D Box 中的点云颜色 
        if pcd is not None and mode == 'xyz': 
            # import pdb
            # pdb.set_trace()
            indices = box3d.get_point_indices_within_bounding_box(pcd.points) 
            points_colors[indices] = in_box_color 
            print(len(indices))
 
    # 更新点云颜色 
    if pcd is not None: 
        pcd.colors = o3d.utility.Vector3dVector(points_colors) 
        vis.update_geometry(pcd) 

 
class Visualizer(object): 
    r"""Online visualizer implemented with Open3d.""" 
    def __init__(self, 
                 points, 
                 bbox3d=None, 
                 save_path=None, 
                 points_size=2, 
                 point_color=(0.5, 0.5, 0.5), 
                 bbox_color=(0, 1, 0), 
                 points_in_box_color=(1, 0, 0), 
                 rot_axis=2, 
                 center_mode='lidar_bottom', 
                 mode='xyz'): 
        super(Visualizer, self).__init__() 
        assert 0 <= rot_axis <= 2 
 
        # 调用 Open3D 的 API 来初始化 visualizer 
        self.o3d_visualizer = o3d.visualization.Visualizer() 
        self.o3d_visualizer.create_window() 
        # 创建坐标系帧 
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame( 
            size=1, origin=[0, 0, 0]) 
        self.o3d_visualizer.add_geometry(mesh_frame) 
 
        # 设定点云尺寸 
        self.points_size = points_size 
        # 设定点云颜色 
        self.point_color = point_color 
        # 设定 3D 框颜色 
        self.bbox_color = bbox_color 
        # 设定 3D 框内点云的颜色 
        self.points_in_box_color = points_in_box_color 
        self.rot_axis = rot_axis 
        self.center_mode = center_mode 
        self.mode = mode 
        self.seg_num = 0 

        # import pdb
        # pdb.set_trace()
        # 绘制点云 
        if points is not None: 
            self.pcd, self.points_colors = _draw_points( 
                points, self.o3d_visualizer, points_size, point_color, mode) 
 
    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None): 
        """Add bounding box to visualizer.""" 
        if bbox_color is None: 
            bbox_color = self.bbox_color 
        if points_in_box_color is None: 
            points_in_box_color = self.points_in_box_color 
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd, 
                     bbox_color, points_in_box_color, self.rot_axis, 
                     self.center_mode, self.mode)  
 
    def show(self, save_path=None): 
        """Visualize the points cloud.""" 
        self.o3d_visualizer.run() 
        if save_path is not None: 
            self.o3d_visualizer.capture_screen_image(save_path) 
        self.o3d_visualizer.destroy_window() 
        return 