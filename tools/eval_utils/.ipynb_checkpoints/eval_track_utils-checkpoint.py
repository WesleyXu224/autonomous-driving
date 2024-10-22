import pickle
import time

import numpy as np
import torch
import tqdm
import copy
from smat.models import load_data_to_gpu
from smat.utils import common_utils
from smat.ops.iou3d_nms import iou3d_nms_utils
from smat.utils import box_utils
from .track_eval_metrics import AverageMeter, Success_torch, Precision_torch

def eval_track_one_epoch(model, dataloader, epoch_id, logger, dataset_cls, dist_test=False, save_to_file=False, result_dir=None):
    dataset = dataloader.dataset
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    first_index = dataset.first_frame_index
    model.eval()
    total_inference_time = 0.0
    inference_count = 0
    Success_main = Success_torch()
    Success_main_bev = Success_torch()
    Precision_main = Precision_torch()

    pbar = tqdm.tqdm(total=len(first_index)-1, leave=False, desc='tracklets', dynamic_ncols=True)
    for f_index in range(len(first_index)-1):
        st = first_index[f_index]
        if f_index == len(first_index)-2:
            ov = first_index[f_index+1]+1
        else:
            ov = first_index[f_index+1]
        
        first_point = dataset[st+1]['template_voxels']   
        length = ov - st - 1
        
        if length > 0:
            for index in range(st+1, ov):
                # print("ov",ov)
                data = dataset[index]
                if index == st+1:
                    previou_box = data['template_gt_box'].reshape(7)
                    first_point = data['or_template_points']
                    Success_main.add_overlap(torch.ones(1).cuda())
                    Success_main_bev.add_overlap(torch.ones(1).cuda())
                    Precision_main.add_accuracy(torch.zeros(1).cuda())

                batch_dict = dataset.collate_batch([data])
                # print("batch_dict",batch_dict)
                template_voxels = batch_dict['template_voxels']
                search_voxels = batch_dict['search_voxels']
                
                load_data_to_gpu(batch_dict)
                gt_box = batch_dict['gt_boxes'].view(-1)[:7]
                center_offset = batch_dict['center_offset'][0]
                
                # avoid spconv bug with sparse points
#                 torch.cuda.synchronize()
#                 start = time.time()
#                 torch.cuda.synchronize()
#                 start_time = time.perf_counter()
                # avoid spconv bug with sparse points
                try:
                    with torch.no_grad():           
                        pred_box = model(batch_dict).view(-1)
                    if dataset_cls=='nus':
                        pred_box[:3]+=center_offset[:3] #nus
                    else:
                        pred_box[:2]+=center_offset[:2] #kitti
#                     torch.cuda.synchronize()
#                     end = time.time()
#                     inference_time = end - start
#                     torch.cuda.synchronize()
#                     elapsed = time.perf_counter() - start_time
        
                except BaseException:
                    pred_box = torch.from_numpy(previou_box).float().cuda()

#                 total_inference_time += elapsed  # 更新总推理时间
#                 inference_count += 1  # 更新推理次数
#                 print('Inference takes {:.3f}s for one image'.format(elapsed))
                iou3d,iou2d = iou3d_nms_utils.boxes_iou3d_gpu(pred_box.view(1,-1), gt_box.view(1,-1))
                iou3d = iou3d.squeeze()
                iou2d = iou2d.squeeze()
                accuracy = torch.norm(pred_box[:3] - gt_box[:3])
                # with open('Success_main.txt', 'w') as f:
                #     f.write(str(accuracy.item()))
                Success_main.add_overlap(iou3d)
                Success_main_bev.add_overlap(iou2d)
                Precision_main.add_accuracy(accuracy)
                # with open('Success_main.txt', 'w') as f:
                #     f.write(str(Success_main.average.item()))
                # with open('Success_main.txt', 'w') as f:
                #     f.write(str(Success_main.average.item()) + '\n')
#                 with open('Success_main_nus_car.txt', 'a') as f:
#                     f.write(str(Success_main.average.item()) + '\n')
#                 with open('Precision_main_nus_car.txt', 'a') as f:
#                     f.write(str(Precision_main.average.item()) + '\n')
                dataset.set_first_points(first_point)
                dataset.set_refer_box(pred_box.cpu().numpy())
                previou_box = pred_box.cpu().numpy()
#             with open('Success_main_nus.value.txt', 'a') as f:
#                 f.write(str(Success_main.value) + '\n')
            # print("Success_main.value",Success_main.value)
            dataset.reset_all()
        pbar.update()
    pbar.close()
    avs = Success_main.average.item()
    avs_bev = Success_main_bev.average.item()
    avp = Precision_main.average.item()
#     average_inference_time = total_inference_time / inference_count
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    ret_dict = {}

    logger.info('Success: %f' % (avs))
    logger.info('Success_bev: %f' % (avs_bev))
    logger.info('Precision: %f' % (avp))
#     logger.info('ave_time: %f' %(average_inference_time))
    ret_dict['test/Success'] = avs
    ret_dict['test/Success_bev'] = avs_bev
    ret_dict['test/Precision'] = avp

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
