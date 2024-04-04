import os
import numpy as np
import copy
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from ..sotdataset import SOTDatasetTemplate

from ...utils import box_utils, calibration_kitti, common_utils, tracklet3d_kitti
import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from . import points_utils, vis
import open3d as o3d
from . import open3d_vis_utils as V
general_to_tracking_class = {"animal": "void / ignore",
                             "human.pedestrian.personal_mobility": "void / ignore",
                             "human.pedestrian.stroller": "void / ignore",
                             "human.pedestrian.wheelchair": "void / ignore",
                             "movable_object.barrier": "void / ignore",
                             "movable_object.debris": "void / ignore",
                             "movable_object.pushable_pullable": "void / ignore",
                             "movable_object.trafficcone": "void / ignore",
                             "static_object.bicycle_rack": "void / ignore",
                             "vehicle.emergency.ambulance": "void / ignore",
                             "vehicle.emergency.police": "void / ignore",
                             "vehicle.construction": "void / ignore",
                             "vehicle.bicycle": "bicycle",
                             "vehicle.bus.bendy": "bus",
                             "vehicle.bus.rigid": "bus",
                             "vehicle.car": "car",
                             "vehicle.motorcycle": "motorcycle",
                             "human.pedestrian.adult": "pedestrian",
                             "human.pedestrian.child": "pedestrian",
                             "human.pedestrian.construction_worker": "pedestrian",
                             "human.pedestrian.police_officer": "pedestrian",
                             "vehicle.trailer": "trailer",
                             "vehicle.truck": "truck", }

tracking_to_general_class = {
    'void / ignore': ['animal', 'human.pedestrian.personal_mobility', 'human.pedestrian.stroller',
                      'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
                      'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack',
                      'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.construction'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'car': ['vehicle.car'],
    'motorcycle': ['vehicle.motorcycle'],
    'pedestrian': ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
                   'human.pedestrian.police_officer'],
    'trailer': ['vehicle.trailer'],
    'truck': ['vehicle.truck']}
class NuscenesSOTDataset(SOTDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, eval_flag=False, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, eval_flag=eval_flag, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # import pdb
        # pdb.set_trace()
        self.velodyne_path = os.path.join(self.root_path, 'velodyne')
        self.label_path = os.path.join(self.root_path, 'label_02')
        self.calib_path = os.path.join(self.root_path, 'calib')

        self.search_dim = [self.point_cloud_range[x::3][1] - self.point_cloud_range[x::3][0] for x in range(3)]

        self.key_frame_only = True
        version = 'v1.0-trainval'
        self.nusc = NuScenes(version=version, dataroot=self.root_path, verbose=False)
        self.version = version
        self.category_name="bus"#xuws
        self.min_points = 1
        self.track_instances = self.filter_instance(self.split, self.category_name.lower(), self.min_points)
        
        self.generate_split_list(self.split)

        self.refer_box = None
        self.first_points = None
        self.sequence_points = None



        # import pdb
        # pdb.set_trace()
        if self.mode == 'train' or self.mode == 'val':
            self.search_func = self.train_val_search_func
        else:
            self.search_func = self.test_search_func
        

        
    def filter_instance(self, split, category_name=None, min_points=-1):
        """
        This function is used to filter the tracklets.

        split: the dataset split
        category_name:
        min_points: the minimum number of points in the first bbox
        """
        # import pdb
        # pdb.set_trace()
        if category_name is not None:
            general_classes = tracking_to_general_class[category_name]
        instances = []
        scene_splits = nuscenes.utils.splits.create_splits_scenes()
        for instance in self.nusc.instance:
            anno = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', anno['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            instance_category = self.nusc.get('category', instance['category_token'])['name']
            if scene['name'] in scene_splits[split] and anno['num_lidar_pts'] >= min_points and \
                    (category_name is None or category_name is not None and instance_category in general_classes):
                instances.append(instance)
        return instances
    def generate_split_list(self, split):
        # import pdb
        # pdb.set_trace()

        list_of_tracklet_anno = []
        self.first_frame_index = [0]
        number = 0
        list_of_tracklet_len = []
        for instance in self.track_instances:
            track_anno = []
            curr_anno_token = instance['first_annotation_token']

            while curr_anno_token != '':

                ann_record = self.nusc.get('sample_annotation', curr_anno_token)
                sample = self.nusc.get('sample', ann_record['sample_token'])
                sample_data_lidar = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

                curr_anno_token = ann_record['next']
                if self.key_frame_only and not sample_data_lidar['is_key_frame']:
                    continue
                track_anno.append({"sample_data_lidar": sample_data_lidar, "box_anno": ann_record})

            list_of_tracklet_anno.append(track_anno)
            number += len(track_anno)
            self.first_frame_index.append(number)
            list_of_tracklet_len.append(len(track_anno))

        # every tracklet
        self.one_track_infos = self.get_whole_relative_frame(list_of_tracklet_anno)
        print('track_info_len: ',len(self.one_track_infos))
        self.first_frame_index[-1] -= 1

    def get_whole_relative_frame(self, tracklets_infos):
        all_infos = []
        for one_infos in tracklets_infos:
            track_length = len(one_infos)
            relative_frame = 1
            for frame_info in one_infos:
                frame_info["relative_frame"] = relative_frame
                frame_info["track_length"] = track_length
                relative_frame += 1
                all_infos.append(frame_info)

        return all_infos

    def get_lidar(self, anno):


        sample_data_lidar = anno['sample_data_lidar']
        box_anno = anno['box_anno']
        bb = Box(box_anno['translation'], box_anno['size'], Quaternion(box_anno['rotation']),
                 name=box_anno['category_name'], token=box_anno['token'])
        pcl_path = os.path.join(self.root_path, sample_data_lidar['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        cs_record = self.nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        poserecord = self.nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        bb = np.concatenate([bb.center.reshape(-1,3),bb.wlh.reshape(-1,3),bb.orientation.yaw_pitch_roll[0].reshape(-1,1)],axis=1)
        bb[:,[3,4]] = bb[:,[4,3]]

        # points = np.transpose(pc.points)
        # points[:,:2] = points[:,:2] - bb[0,:2]
        # bb[:,:2] = bb[:,:2] - bb[0,:2]
        # # # vis_demo =vis.Visualizer(points = points, bbox3d = bb)
        # # # vis_demo.add_bboxes(bb)
        # # # vis_demo.show()
        # import pdb
        # pdb.set_trace()
        # V.draw_scenes(points=points, ref_boxes = bb,ref_scores=bb, ref_labels= bb)
        return np.transpose(pc.points), bb

    def get_label(self, idx):
        return self.one_track_infos[idx]

    def get_calib(self, sequence):
        calib_file = os.path.join(self.calib_path,'{}.txt'.format(sequence))
        assert Path(calib_file).exists()
        return calibration_kitti.Calibration(calib_file)

    def rotat_point(self, points, ry):
        R_M = np.array([[np.cos(ry), -np.sin(ry), 0],
                                [np.sin(ry), np.cos(ry), 0],
                                [0, 0,  1]])  # (3, 3)
        rotated_point = np.matmul(points[:,:3], R_M)  # (N, 3)
        return rotated_point

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return (self.first_frame_index[-1]+1) * self.total_epochs

        return (self.first_frame_index[-1]+1)

    def __getitem__(self, index):
        return self.get_tracking_item(index)
  
    def find_template_idx(self, index, intervel=1):
        if self.mode == 'train' or self.mode == 'val':
            # import pdb
            # pdb.set_trace()
            search_anno = self.one_track_infos[index]
            search_relative_frame = search_anno['relative_frame']
            search_whole_length = search_anno['track_length']
            search_min_index = max(0, search_relative_frame-intervel)
            search_max_index = min(search_relative_frame+intervel, search_whole_length)
            template_relative_frame = np.random.randint(search_min_index, search_max_index)
            template_index = index + template_relative_frame - search_relative_frame
        else:
            template_index = index - 1
        # import pdb
        # pdb.set_trace()
        return template_index

    def template_preprocess_lidar(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        # import pdb
        # pdb.set_trace()
        cx, cy, cz, dx, dy, dz, ry = label
        label_center = np.stack((cx, cy, cz))

        corner = box_utils.boxes_to_corners_3d(label.reshape(1, 7))[0]

        flag = box_utils.in_hull(point[:,:3], corner)
        crop_points = point[flag]
        new_label = copy.deepcopy(label)
        crop_points[:,:3] -= label_center
        new_label[:3] -=label_center

        return crop_points, new_label

    def search_preprocess_lidar(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        # import pdb
        # pdb.set_trace()
        cx, cy, cz, dx, dy, dz, ry = label
        corner = box_utils.boxes_to_corners_3d(label.reshape(1, 7))[0]
        object_flag = box_utils.in_hull(point[:,:3], corner)
        object_points = point[object_flag]
        if object_points.shape[0] != 0:
            # offset_xy = np.random.uniform(low=-sample_dx/self.offset_down, high=sample_dx/self.offset_down, size=2)
            # offset_x = cx + offset_xy[0]
            # offset_y = cy + offset_xy[1]

            offset_x = cx
            offset_y = cy

            sample_center = np.stack((offset_x, offset_y, cz))

            search_area = np.stack((offset_x, offset_y, 0, sample_dx, sample_dy, sample_dz, 0))

            # get points in search area
            search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]

            flag = box_utils.in_hull(point[:,:3], search_corner)
            crop_points = point[flag]
            # import pdb
            # pdb.set_trace()
            # search_area = np.stack((offset_x, offset_y, 0, sample_dx, sample_dy, 10, 0))

            # # get points in search area
            # search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]

            # flag = box_utils.in_hull(point[:,:3], search_corner)
            # crop_points = point[flag]

            crop_points[:,:3] -= sample_center
            # import pdb
            # pdb.set_trace()
            # label[:2] = label[:2] - sample_center[:2] 
            # label=label.reshape(1, 7)
            # V.draw_scenes(points=crop_points, ref_boxes = label,ref_scores=label, ref_labels= label)
            return crop_points, sample_center
        else:
            # offset_xy = np.random.uniform(low=-sample_dx/2, high=sample_dx/2, size=2)
            # offset_x = cx + offset_xy[0]
            # offset_y = cy + offset_xy[1]
            offset_x = cx
            offset_y = cy

            sample_center = np.stack((offset_x, offset_y, cz))
            search_area = np.stack((offset_x, offset_y, 0, sample_dx, sample_dy, sample_dz, 0))
            # import pdb
            # pdb.set_trace()
            search_corner = box_utils.boxes_to_corners_3d(search_area.reshape(1, 7))[0]
            
            flag = box_utils.in_hull(point[:,:3], search_corner)
            crop_points = point[flag]
            crop_points[:,:3] -= sample_center
            return crop_points, sample_center
    def test_get_search_area(self, point, label, sample_dx=6, sample_dy=6, sample_dz=3):
        cx, cy, cz, dx, dy, dz, ry = label
        sample_center = np.stack((cx, cy, cz))

        area = np.stack((cx, cy, 0, sample_dx, sample_dy, sample_dz, 0)).reshape(1, 7)
        corner = box_utils.boxes_to_corners_3d(area)[0]
        object_flag = box_utils.in_hull(point[:,:3], corner)
        area_points = point[object_flag]      
        area_points[:,:3] -= sample_center

        # # bb = label.reshape
        # # points = local_search_points
        # import pdb
        # pdb.set_trace()
        # label[:2] = label[:2] - sample_center[:2] 
        # label=label.reshape(1, 7)
        # # V.draw_scenes(points=area_points, ref_boxes = label,ref_scores=label, ref_labels= label)
        # V.draw_scenes(points=point[:,:3]-sample_center, ref_boxes = label,ref_scores=label, ref_labels= label)
        return area_points, sample_center

    def get_tracking_item(self, index):
        search_info = self.get_label(index)

        gt_name = self.category_name
        # import pdb
        # pdb.set_trace()

        template_index = self.find_template_idx(index)
        template_info = self.get_label(template_index)
        
        search_points, gt_boxes_lidar = self.get_lidar(search_info)
        # points = np.transpose(pc.points)
        # points[:,:2] = points[:,:2] - bb[0,:2]
        # bb[:,:2] = bb[:,:2] - bb[0,:2]
        # vis_demo =vis.Visualizer(points = points, bbox3d = bb)
        # vis_demo.add_bboxes(bb)
        # vis_demo.show()
        template_points, template_gt_box_lidar = self.get_lidar(template_info)
        
        local_search_points, center_offset, local_template_points, local_template_label = self.search_func(gt_name, gt_boxes_lidar[0], template_gt_box_lidar[0], search_points, template_points)


        # import pdb
        # pdb.set_trace()
        norm_tpoints = self.rotat_point(local_template_points, local_template_label[6])
        point_r = local_template_points[:,3].reshape(-1,1)
        norm_tpoints = np.hstack((norm_tpoints, point_r))
        # import pdb
        # pdb.set_trace()
        if self.mode == 'train' or self.mode == 'val':
            if norm_tpoints.shape[0] <= 20 or local_search_points.shape[0] <= 20:
                # import pdb
                # pdb.set_trace()
                # print(f"norm_tpoints.shape[0] = {norm_tpoints.shape[0]}")
                # print(f"local_search_points.shape[0] = {local_search_points.shape[0]}")
                return self.__getitem__(np.random.randint(0, self.__len__()))
        # import pdb
        # pdb.set_trace()
        
        norm_tlabel = copy.deepcopy(local_template_label)
        norm_tlabel_xyz = self.rotat_point(local_template_label.reshape(-1,7), local_template_label[6]).reshape(-1)
        norm_tlabel[:3] = norm_tlabel_xyz
        norm_tlabel[6] = 0

        # bb = local_template_label.reshape(1,-1)
        # points = local_template_points
        # import pdb
        # pdb.set_trace()
        # # points[:,:2] = points[:,:2] - bb[0,:2]
        # bb[:,:2] = bb[:,:2] - bb[:,:2]
        # vis_demo =vis.Visualizer(points = points, bbox3d = bb)
        # vis_demo.add_bboxes(bb)
        # vis_demo.show()

        # bb = gt_boxes_lidar
        # points = local_search_points
        # import pdb
        # pdb.set_trace()
        # # points[:,:2] = points[:,:2] - bb[0,:2]
        # bb[:,:2] = bb[:,:2] - center_offset[:2]
        # vis_demo =vis.Visualizer(points = points, bbox3d = bb)
        # vis_demo.add_bboxes(bb)
        # vis_demo.show()

        # import pdb
        # pdb.set_trace()
        input_dict = {
            'gt_names': gt_name,
            'gt_boxes': gt_boxes_lidar,#维度存在问题
            'search_points': local_search_points,
            'center_offset': center_offset.reshape(1,-1),
            'object_dim': norm_tlabel[3:6].reshape(1, 3),
            'template_box': norm_tlabel.reshape(1, 7),
            'template_gt_box': template_gt_box_lidar.reshape(1, 7),
        }

        # import pdb
        # pdb.set_trace()
        # bb = label.reshape
        # points = local_search_points
        
        # label[:2] = label[:2] - label[:2] 
        # label=label.reshape(1, 7)
        # V.draw_scenes(points=area_points, ref_boxes = label,ref_scores=label, ref_labels= label)
        # V.draw_scenes(points=point[:,:3]-sample_center, ref_boxes = label,ref_scores=label, ref_labels= label)
        # points = local_search_points
        # label = gt_boxes_lidar.copy()
        # label[:,:3] = label[:,:3] - center_offset[:3] 
        # V.draw_scenes(points=points, ref_boxes = label,ref_scores=label, ref_labels= label)
                
        # points = norm_tpoints
        # label = norm_tlabel.copy().reshape(-1,7)
        # V.draw_scenes(points=points, ref_boxes = label,ref_scores=label, ref_labels= label)

        # points = local_template_points
        # label = local_template_label.copy().reshape(-1,7)
        # V.draw_scenes(points=points, ref_boxes = label,ref_scores=label, ref_labels= label)


        if self.mode == 'test':
            input_dict.update({
                'or_search_points': local_search_points,
                'or_template_points': norm_tpoints,
            })

        if (self.first_points is not None) and (self.mode == 'test'):
            # first & previous
            norm_tpoints = np.vstack((self.first_points, norm_tpoints))

        input_dict.update({
            'template_points': norm_tpoints,
        })
        # import pdb
        # pdb.set_trace()
        data_dict = self.prepare_data(data_dict=input_dict)
        if self.mode == 'train' or self.mode == 'val':
            tv = data_dict['template_voxels']
            sv = data_dict['search_voxels']
            if sv.shape[0] <= 20 or tv.shape[0] <= 20:
                # import pdb
                # pdb.set_trace()
                return self.__getitem__(np.random.randint(0, self.__len__()))
        # import pdb
        # pdb.set_trace()
        return data_dict

    def train_val_search_func(self, gt_name, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        if self.refer_box is not None:
            search_box = self.refer_box
            template_box = self.refer_box
        else:
            gt_boxes_lidar = template_gt_box_lidar
            template_box = template_gt_box_lidar

        # import pdb
        # pdb.set_trace()

        local_search_points, center_offset = self.search_preprocess_lidar(search_points, gt_boxes_lidar, self.search_dim[0], self.search_dim[1], self.search_dim[2])
        local_template_points, local_template_label = self.template_preprocess_lidar(template_points, template_gt_box_lidar, self.search_dim[0], self.search_dim[1], self.search_dim[2])

        return local_search_points, center_offset, local_template_points, local_template_label

    def test_search_func(self, gt_name, gt_boxes_lidar, template_gt_box_lidar, search_points, template_points):
        if self.refer_box is not None:
            # print("self.refer_box")
            search_box = self.refer_box
            template_box = self.refer_box
        else:
            # print("template_gt_box_lidar")
            search_box = template_gt_box_lidar
            template_box = template_gt_box_lidar

        # import pdb
        # pdb.set_trace()
        # local_search_points_f_1(search_points), center_offset f_0(x,y,z)
        local_search_points, center_offset = self.test_get_search_area(search_points, search_box, self.search_dim[0], self.search_dim[1], self.search_dim[2])
        local_template_points, local_template_label = self.template_preprocess_lidar(template_points, template_box, self.search_dim[0], self.search_dim[1], self.search_dim[2])

        return local_search_points, center_offset, local_template_points, local_template_label

    def set_refer_box(self, refer_box):
        self.refer_box = refer_box

    def set_first_points(self, points):
        self.first_points = points
    
    def reset_all(self):
        self.refer_box = None
        self.first_points = None
        self.sequence_points = None