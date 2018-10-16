import os
import sys
sys.path.append('../../')
import option_show_groundtruth
import option_show_groundtruth_1
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
# from data.pose_loader_sampler import PoseData as PoseDataAI
# from data.pose_loader_ai import PoseData as PoseDataAI
# from data.AILoader import PoseData as PoseDataAI
from data.zyh_loader import PoseData as PoseDataAI
import data.zyh_transforms as Mytransforms
# from net import resnet18_fpn,mobilenetv2_gcn,src_mobilenetv2_gcn,mbv2_mgcn,mb2_192,mb2_fpn5
import model_defi
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tools.AverageMeter import AverageMeter
from tqdm import tqdm
import numpy as np
import tools.utils as utils
import importlib
from glob import glob as glob
import cv2
import argparse
import tools.poseBenchmark as pb
from PIL import ImageDraw
from PIL import Image
from PIL import ImageFont

def load_pretrained(network, save_path,isDataParallel=False, pop_list = []):
    pretrained_dict = torch.load(save_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    if not isDataParallel:
        pretrained_dict_ = {}
        for k, v in pretrained_dict.items():
            pretrained_dict_[k[7:]] = v
        pretrained_dict = pretrained_dict_
    for p in pop_list:
        pretrained_dict.pop(p)
    # pretrained_dict.pop('module.classification_4.weight')
    # pretrained_dict.pop('module.classification_4.bias')

    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #print pretrained_dict
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network

# def adjust_learning_rate(optimizer, decay_rate=.9):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = param_group['lr'] * decay_rate

def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    return lr



def get_parameters(model, base_lr, isdefault=True):
    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': base_lr},
            {'params': lr_2, 'lr': base_lr * 2.},
            {'params': lr_4, 'lr': base_lr * 4.},
            {'params': lr_8, 'lr': base_lr * 8.}]

    return params, [1., 2., 4., 8.]

def handleImages(model,imgdir,save_dir,iters,opts):
    imgnames = glob(os.path.join(imgdir,'*.jpg'))
    imgnames+=glob(os.path.join(imgdir, '*.png'))
    for img_name in imgnames:
        img_basename = os.path.basename(img_name)
        canvas = utils.process_45_args(model, img_name, {'boxsize':opts.resize_size,'stride':opts.stride})
        save_name = os.path.join(save_dir,'{:0>8}_{}'.format(iters,img_basename))
        cv2.imwrite(save_name, canvas)

def handleMAPs(model,image_path,samples,anno_path,json_save_path):
    img_names = glob(os.path.join(image_path, '*.jpg'))
    img_names = img_names[:samples]
    for i, imgname in enumerate((img_names)):
        pre_name = os.path.basename(imgname).split('.')[0]
        subset = utils.getsubsets(model, imgname)
        pb.savePredictJson(subset, os.path.join(json_save_path, pre_name + '.json'))
    mAP = pb.compareDir(anno_path, json_save_path)
    return mAP



def show_groundtruth(opts):

#=============================model===========================================================
    # module_name = 'net.'+opts.model_name
    # module = importlib.import_module(module_name)
    '''
    model = model_defi.get_model()
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(opts.test_snapshot_path)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.MSELoss().cuda()
    '''

#=============================dataset==========================================================
    test_transform = Mytransforms.Compose([
                    Mytransforms.Crop(opts.resize_size),
                    Mytransforms.RandomRotate(40),
                ])
    visible_level = opts.visible_level

    test_dataset = PoseDataAI({
                'desc_file_path': opts.test_image_txt_path,
                'data_root': '/',
                'crop_size': opts.resize_size,
                'rotate': 40,
                'stride': opts.stride,
                'visible_level': visible_level,
                'sigma':opts.sigma,
                'theta':opts.theta,
                'dataset_id':opts.dataset_id,
    }, transformer=test_transform,isMask=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=opts.batch_size_test, shuffle=True, num_workers=opts.num_workers,
                                                   pin_memory=True)

#=====================================test=====================================================

    iters = 0

    weight = 46 * 46 * 38 / 2.0
    weight = weight
    snap_dir = os.path.join(opts.snapshot_save_dir,opts.experiment_name)
    if not os.path.isdir(snap_dir):
        os.makedirs(snap_dir)

    test_img_save_dir = os.path.join(snap_dir,'gt_images')

    if not os.path.isdir(test_img_save_dir):
        os.makedirs(test_img_save_dir)

    '''
    test_log = open(os.path.join(snap_dir, 'test_log.log'), 'w')
    '''

    colors = ['gray', 'red', 'brown', 'lawngreen', 'gold', 'orange', 'white', 'darkslategray', 'cyan', 'blue', 'navy', 'purple', 'pink', 'yellow']
    edges = [[], [0], [1], [1], [2], [3], [4], [5], [2], [3], [8], [9], [10], [11]]
    cnt = 0
    for i, (imgs, labels) in enumerate(test_loader):
        '''
        test_losses = AverageMeter()
        '''
        imgs = Variable(imgs).cuda()
        labels = Variable(labels).cuda()
        '''
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss = loss * weight
        test_losses.update(loss.data[0], 1)
        '''
        imgs_array = imgs.cpu().numpy()
        labels_array = labels.cpu().numpy()
        '''
        preds_array = preds.cpu().detach().numpy()
        '''

        for j in range(imgs_array.shape[0]):
            img_array = (imgs_array[j, 0:3, :, :].transpose((1, 2, 0)) * 255.0 + 128.0).astype(np.uint8)
            width_ = img_array.shape[1]
            '''
            image = Image.fromarray(img_array)
            '''
            image2 = Image.fromarray(img_array)

            '''
            draw = ImageDraw.Draw(image)
            '''
            draw2 = ImageDraw.Draw(image2)
            '''
            pred_array = preds_array[j, :, :, :]
            '''
            label_array = labels_array[j, :, :, :]
            '''
            joints_dict = {}
            for k in range(pred_array.shape[0]):
                indice = np.argmax(pred_array[k])
                x_ = indice // width_
                y_ = indice % width_
                joints_dict[k] = (x_, y_)
                # print(width_, indice, x_, y_)
                circle_bar = 5
                draw.ellipse((y_ - circle_bar, x_ - circle_bar, y_ + circle_bar, x_ + circle_bar), colors[k], colors[k])
                for l in edges[k]:
                    x_1, y_1 = joints_dict[l]
                    draw.line((y_1, x_1, y_, x_), 'bisque')
            image.save(test_img_save_dir + '/{}.jpg'.format(cnt))
            '''
            joints_dict2 = {}
            for k in range(label_array.shape[0]):
                indice = np.argmax(label_array[k])
                x_ = indice // width_
                y_ = indice % width_
                joints_dict2[k] = (x_, y_)
                # print(width_, indice, x_, y_)
                circle_bar = 5
                draw2.ellipse((y_ - circle_bar, x_ - circle_bar, y_ + circle_bar, x_ + circle_bar), colors[k], colors[k])
                for l in edges[k]:
                    x_1, y_1 = joints_dict2[l]
                    draw2.line((y_1, x_1, y_, x_), 'bisque')
            image2.save(test_img_save_dir + '/{}_gt.jpg'.format(cnt))
            print(cnt)
            cnt += 1

    '''
    test_info = 'loss:{}'.format(test_lossed.avg)
    print(test_info)
    test_log.write(test_info + '\n')
    test_log.flush()

    test_log.close()
    '''

def overrideOpts(opts,args):
    opts.experiment_name = args.experiment_name
    opts.model_name = args.model_name
    if args.is_finetune:
        opts.is_finetune = True
        opts.finetune_model = args.pretrained
    else:
        opts.is_finetune = False
    opts.gpuid = args.gpuid
    opts.resize_size = args.resize_size
    opts.stride = args.stride
    opts.sigma = args.sigma

    return opts

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',default=None, type=str,
                        dest='experiment_name', help='experiment_name')
    parser.add_argument('--gpuid', default='0,1,2,3',type=str,
                        dest='gpuid', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--stride', default=4, type=int,
                        dest='stride', help='stride')
    parser.add_argument('--resize_size', default=384, type=int,
                        dest='resize_size', help='resize_size')
    parser.add_argument('--sigma', default=7, type=float,
                        dest='sigma', help='sigma')
    parser.add_argument('--model_name', type=str,
                        dest='model_name', help='model_name')

    # parser.add_argument('--val_dir', default=None, nargs='+', type=str,
    #                     dest='val_dir', help='the path of val file')
    # parser.add_argument('--num_classes', default=1000, type=int,
    #                     dest='num_classes', help='num_classes (default: 1000)')
    return parser.parse_args()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    # args = parse()
    # model = construct_model(args)
    opts = option_show_groundtruth_1.initialize_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpuid
    show_groundtruth(opts)


