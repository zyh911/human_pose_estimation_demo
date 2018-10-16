#!usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

def initialize_arguments():
    arg_dict = {
        # global arguments
        'experiment_name':              '20180507_video_pose_estimation_v1',
        'model_name':                   'video_v1',
        # 'model_name':                   'mb2_dualch',
        'project_root':                 '/root/zyh3/poseestimation/pose_siyuan',
        'gpuid':                        '0,1',
        # arguments for dataset
        'train_image_txt_path':         '/root/group-competition/poseEstimation/data/AI_Challenger_keypoint/train.txt',
        'val_image_txt_path':           '/root/group-competition/poseEstimation/data/AI_Challenger_keypoint/val.txt',
        'input_height':                 25, # 128, should mannually make sure the size of patch fits the size of neural net!
        'input_width':                  25, # 128,
        'input_channels':               4,
        'batch_size_train':             16,
        'batch_size_val':               16,
        'num_workers':                  10,
        'visible_level':                1,
        'resize_size':                  384,
        'sigma':                        7,
        'stride':                       4,
        'weight_sampler':               False,
        'theta':                        1.0,

        # arguments for training
        'is_finetune':                  False,
        # 'finetune_model':               '/root/ysy2/projects/mt_pose/human-pose-estimation/snapshots/20180328_stratch/mobile_gcn_iters_63000.pth.tar',
        'finetune_model':               './snapshots/20180404_mbv2_gcnm_mask/mb_mask_iters_63000.pth.tar',
        # 'gpu_ids':                      0, # can be a number or a tuple
        'val_interval':                 5000,
        'val_iters':                    500,
        'print_interval':               5,
        'num_epochs':                   300,
        'num_iter_max':                 5000000,
        'snapshot_interval':            5000, # should be divided exactly by val_interval, for saving intermediate images
        'snapshot_save_dir':            'snapshots/',
        'intermediate_image_dir':       'snapshots/intermediate_images/',
        'log_save_dir':                 'logs/',
        'grad_thresh':                  20,

        #arguments for optimizer and loss
        'optimizer':                    'adam',
        'lr':                           1e-4,
        'momentum':                     0.9,
        'weight_decay':                 0.0005,
        'lr_step_size':                 2000,
        'lr_step_ratio':                0.9,
        'gamma':                        0.5,
        'step_size':                    60000,

        'dataset_id':                   1,

    }

    arg_parser = argparse.ArgumentParser('parse args')
    for arg_key in arg_dict.keys():
        arg_val = arg_dict[arg_key]
        if arg_key.endswith('_path') or arg_key.endswith('_dir'):
            arg_val = os.path.join(arg_dict['project_root'], arg_val) # important to used abs path
        arg_parser.add_argument('--'+arg_key, type=type(arg_val), default=arg_val)

    return arg_parser.parse_args()


if __name__ == '__main__':
    initialize_arguments()
