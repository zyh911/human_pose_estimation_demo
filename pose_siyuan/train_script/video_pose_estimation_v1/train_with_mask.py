import os
import sys
sys.path.append('../../')
import option
import option_1
import option_finetune
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
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['base_lr'] * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
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
    params = [{'params': lr_1, 'base_lr': base_lr},
            {'params': lr_2, 'base_lr': base_lr * 2.},
            {'params': lr_4, 'base_lr': base_lr * 4.},
            {'params': lr_8, 'base_lr': base_lr * 8.}]

    return params

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



def train_val(opts):

#=============================model===========================================================
    # module_name = 'net.'+opts.model_name
    # module = importlib.import_module(module_name)
    model = model_defi.get_model()
    model = nn.DataParallel(model).cuda()
    utils.initialize_weights(model)
    if opts.is_finetune:
        model = load_pretrained(model, opts.finetune_model, True)

#============================optimizer========================================================
    criterion = nn.MSELoss().cuda()
    params = get_parameters(model, opts.lr, False)
    # if not opts.is_finetune:
    #     optimizer = torch.optim.Adam(params, opts.lr, weight_decay=opts.weight_decay)
    # else:
    #     optimizer = torch.optim.SGD(params, opts.lr, weight_decay=opts.weight_decay)

    optimizer_adam = torch.optim.Adam(params, opts.lr, weight_decay=opts.weight_decay)
    optimizer_sgd = torch.optim.SGD(params, opts.lr, weight_decay=opts.weight_decay)
    optimizer = optimizer_adam

#=============================dataset==========================================================
    train_transform = Mytransforms.Compose([
                    Mytransforms.Crop(opts.resize_size),
                    Mytransforms.RandomRotate(40),
                ])
    valid_transform = Mytransforms.Compose([Mytransforms.Crop(opts.resize_size),])

    visible_level = opts.visible_level

    train_dataset = PoseDataAI({
                'desc_file_path': opts.train_image_txt_path,
                'data_root': '/',
                'crop_size': opts.resize_size,
                'rotate': 40,
                'stride': opts.stride,
                'visible_level': visible_level,
                'sigma':opts.sigma,
                'theta':opts.theta,
                'dataset_id':opts.dataset_id,
    }, transformer=train_transform,isMask=True)
    # sample_weights = train_dataset.get_sample_weight_by_invisible_joints()
    sample_weights = utils.loadVisibleWeights('../../SampleWeight.txt')
    sample_weights = np.array(sample_weights)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))

    valid_dataset = PoseDataAI({
                'desc_file_path': opts.val_image_txt_path,
                'data_root': '/',
                'crop_size': opts.resize_size,
                'rotate': 40,
                'stride': opts.stride,
                'visible_level': visible_level,
                'sigma':opts.sigma,
                'theta':opts.theta,
                'dataset_id':opts.dataset_id,
    }, transformer=valid_transform,isMask=True)
    if opts.weight_sampler:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opts.batch_size_train, shuffle=False, num_workers=opts.num_workers,
                                                   pin_memory=True,sampler = sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opts.batch_size_train, shuffle=True, num_workers=opts.num_workers,
                                                   pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=opts.batch_size_val, shuffle=True, num_workers=opts.num_workers,
                                               pin_memory=True)

#=====================================train=====================================================

    iters = 0
    model.train()

    weight = 46 * 46 * 38 / 2.0
    weight = weight
    snap_dir = os.path.join(opts.snapshot_save_dir,opts.experiment_name)
    if not os.path.isdir(snap_dir):
        os.makedirs(snap_dir)

    test_imgs_dir = opts.project_root+'/test_imgs'
    test_img_save_dir = os.path.join(snap_dir,'imgs')

    if not os.path.isdir(test_img_save_dir):
        os.makedirs(test_img_save_dir)

    train_log = open(os.path.join(snap_dir,'train_log.log'),'w')
    valid_log = open(os.path.join(snap_dir,'valid_log.log'),'w')

    policy_parameter = {}
    policy_parameter['gamma'] = opts.gamma
    policy_parameter['step_size'] = opts.step_size
    optim_judge = True

    iters_sgd = 0

# image_path,samples,anno_path,json_save_path
    benchmark_args ={
        'image_path': '/root/group-pose-estimation/group-qxc_ysy_jh/qxc/data/20180404_AI_Challange_compressed/AI_Challenger_keypoint/ai_challenger_keypoint_validation_20170911',
        'samples': 100,
        'anno_path': '/root/group-pose-estimation/group-qxc_ysy_jh/qxc/data/20180404_AI_Challange_compressed/AI_Challenger_keypoint/ai_challenger_keypoint_validation_20170911/json',
        'json_save_path': os.path.join(snap_dir,'jsons')
    }
    if not os.path.isdir(benchmark_args['json_save_path']):
        os.makedirs(benchmark_args['json_save_path'])


    for epoch in range(opts.num_epochs):
        for i,(imgs,labels) in enumerate(tqdm(train_loader)):
            # if iters>100 and optim_judge:
            #     optimizer = optimizer_sgd
            if epoch > 200:
                optimizer = optimizer_sgd
                learning_rate = adjust_learning_rate(optimizer, iters_sgd, opts.lr,
                                                     policy_parameter=policy_parameter)
                iters_sgd+=1
            else:
                learning_rate = adjust_learning_rate(optimizer, iters, opts.lr,
                                                     policy_parameter=policy_parameter)
            # print(imgs.shape)


            imgs = Variable(imgs).cuda()
            labels = Variable(labels).cuda()
            preds = model(imgs)
            loss = criterion(preds,labels)
            loss = loss * weight
            # print(loss)
            # print(type(loss))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), opts.grad_thresh)
            optimizer.step()
            iters+=1
            if iters%opts.print_interval == 0:
                info = "Train epoch:{:0>3} iters:{:0>8} loss:{}".format(epoch,iters,loss.data[0])
                print(info)
                train_log.write(info+'\n')
                train_log.flush()

            if iters%opts.snapshot_interval == 0:
                save_name = os.path.join(snap_dir,"{}_iters_{}.pth.tar".format(opts.model_name,iters))
                torch.save(model.state_dict(), save_name)


            if iters%opts.val_interval == 0:
                model.eval()
                val_iters = 0
                test_losses = AverageMeter()
                for i, (imgs_val, labels_val) in enumerate(valid_loader):
                    # print("valid: {}".format(i))
                    imgs_val = Variable(imgs_val).cuda()
                    labels_val = Variable(labels_val).cuda()
                    preds_val = model(imgs_val)
                    loss_val = criterion(preds_val,labels_val)
                    loss_val = loss_val * weight
                    test_losses.update(loss_val.data[0], 1)
                    val_iters += 1
                    if val_iters >= opts.val_iters:
                        break
                test_info = 'Valid epoch:{:0>3} iters:{:0>8} loss:{}'.format(epoch, iters, test_losses.avg)
                print(test_info)
                valid_log.write(test_info + '\n')
                valid_log.flush()
                # mAP = handleMAPs(model, **benchmark_args)
                # test_info = "Valid epoch:{:0>3} iters:{:0>8} mAP:{}".format(epoch,iters,mAP)
                # print(test_info)
                # valid_log.write(test_info + '\n')
                # valid_log.flush()
                # handleImages(model, test_imgs_dir, test_img_save_dir, iters, opts)
                model.train()

        # adjust_learning_rate(optimizer,opts.lr_step_ratio)

    train_log.close()
    valid_log.close()

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
    opts = option_finetune.initialize_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpuid
    train_val(opts)


