import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
from PIL import Image
import cv2
import zyh_transforms
import Mytransforms
import time

def read_kpt(kpt_list):
    kpt = []
    for kpt_dict in kpt_list:
        kpt_part = [kpt_dict['y'], kpt_dict['x'], kpt_dict['is_visible'], kpt_dict['id']]
        kpt.append(kpt_part)
    return kpt

def read_json(json_file):
    fp = open(json_file)
    data = json.load(fp)
    kpts = []
    bbs = []

    # print('pic', data['human_name'])
    human_list = data['human_list']

    i = -1
    area_max = -1
    for key, human in enumerate(human_list):
        kpt = read_kpt(human['human_keypoints'])
        x_small = -1
        y_small = -1
        x_large = -1
        y_large = -1
        for single_kpt in kpt:
            if single_kpt[2] == 0:
                if x_small == -1 or x_small > single_kpt[0]:
                    x_small = single_kpt[0]
                if x_large == -1 or x_large < single_kpt[0]:
                    x_large = single_kpt[0]
                if y_small == -1 or y_small > single_kpt[1]:
                    y_small = single_kpt[1]
                if y_large == -1 or y_large < single_kpt[1]:
                    y_large = single_kpt[1]
        center_x = (x_small + x_large) // 2
        center_y = (y_small + y_large) // 2
        halfbar_x = (x_large - x_small) // 2
        halfbar_y = (y_large - y_small) // 2
        if halfbar_x < halfbar_y:
            halfbar = halfbar_y
        else:
            halfbar = halfbar_x
        halfbar_expand = int(halfbar * 1.2)
        bbs.append([center_x - halfbar_expand, center_x + halfbar_expand, center_y - halfbar_expand, center_y + halfbar_expand])
        area = 0
        if x_small != 0 and x_large != 0 and y_small != 0 and y_large != 0:
            area = (x_large - x_small) * (y_large - y_small)
        if area > area_max:
            i = key
            area_max = area
        kpts.append(kpt)

    return kpts[i], bbs[i]

def read_json_1(json_file):
    fp = open(json_file)
    data = json.load(fp)
    kpts = []
    bbs = []

    # print('pic', data['human_name'])
    human_list = data['human_list']

    i = -1
    area_max = -1
    for key, human in enumerate(human_list):
        kpt = read_kpt(human['human_keypoints'])
        x_small = -1
        y_small = -1
        x_large = -1
        y_large = -1
        for single_kpt in kpt:
            if single_kpt[2] == 1 or single_kpt[2] == 2:
                single_kpt[2] = 0
                if x_small == -1 or x_small > single_kpt[0]:
                    x_small = single_kpt[0]
                if x_large == -1 or x_large < single_kpt[0]:
                    x_large = single_kpt[0]
                if y_small == -1 or y_small > single_kpt[1]:
                    y_small = single_kpt[1]
                if y_large == -1 or y_large < single_kpt[1]:
                    y_large = single_kpt[1]
            else:
                single_kpt[2] = 2
        center_x = (x_small + x_large) // 2
        center_y = (y_small + y_large) // 2
        halfbar_x = (x_large - x_small) // 2
        halfbar_y = (y_large - y_small) // 2
        if halfbar_x < halfbar_y:
            halfbar = halfbar_y
        else:
            halfbar = halfbar_x
        halfbar_expand = int(halfbar * 1.2)
        bbs.append([center_x - halfbar_expand, center_x + halfbar_expand, center_y - halfbar_expand, center_y + halfbar_expand])
        area = 0
        if x_small != 0 and x_large != 0 and y_small != 0 and y_large != 0:
            area = (x_large - x_small) * (y_large - y_small)
        if area > area_max:
            i = key
            area_max = area
        kpts.append(kpt)

    return kpts[i], bbs[i]

def generate_heatmap(heatmap, kpt, stride, sigma,visible_level):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    # start = 0
    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] > visible_level:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    # heatmap[h][w][j + 1] += math.exp(-dis)
                    heatmap[h][w][j + 1] = max(heatmap[h][w][j + 1],math.exp(-dis))
                    if heatmap[h][w][j + 1] > 1:
                        heatmap[h][w][j + 1] = 1

    return heatmap

# class GenerateHeatmap():
#     def __init__(self,sigma):
#         # self.output_res = output_res
#         # self.num_parts = num_parts
#         # sigma = self.output_res/64
#         self.sigma = sigma
#         size = 6*sigma + 3
#         x = np.arange(0, size, 1, float)
#         y = x[:, np.newaxis]
#         x0, y0 = 3*sigma + 1, 3*sigma + 1
#         self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#
#     def __call__(self, keypoints,output_res,num_parts,stride,visible_level):
#         hms = np.zeros(shape = ( output_res, output_res,num_parts+1), dtype = np.float32)
#         sigma = self.sigma
#         for p in keypoints:
#             for idx, pt in enumerate(p):
#                 if pt[2]<=visible_level:
#                     x, y = int(pt[0]/float(stride)), int(pt[1]/float(stride))
#                     if x<0 or y<0 or x>=output_res or y>=output_res:
#                         #print('not in', x, y)
#                         continue
#                     ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
#                     br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)
#
#                     c,d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
#                     a,b = max(0, -ul[1]), min(br[1], output_res) - ul[1]
#
#                     cc,dd = max(0, ul[0]), min(br[0], output_res)
#                     aa,bb = max(0, ul[1]), min(br[1], output_res)
#                     hms[aa:bb,cc:dd,idx+1] = np.maximum(hms[aa:bb,cc:dd,idx+1], self.g[a:b,c:d])
#         hms[:,:,0] = 1.0 - np.max(hms[:, :, 1:], axis=2)
#         return hms

class GenerateHeatmap():
    def __init__(self,sigma):
        # self.output_res = output_res
        # self.num_parts = num_parts
        # sigma = self.output_res/64
        self.sigma = sigma
        # print("sigma: {}".format(sigma))
        size = 6*sigma + 3
        # size = int(size)
        # size = float(size)
        # sigma = (float(size)-3)/6
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints,output_res,num_parts,stride,visible_level):
        # hms = np.zeros(shape = ( output_res, output_res,num_parts+1), dtype = np.float32)
        hms_res = output_res * stride
        hms = np.zeros(shape=(hms_res, hms_res, num_parts + 1), dtype=np.float32)

        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[2]<=visible_level:
                    # x, y = int(pt[0]/float(stride)), int(pt[1]/float(stride))
                    x, y = int(pt[0]), int(pt[1])
                    # if x<0 or y<0 or x>=output_res or y>=output_res:
                    if x < 0 or y < 0 or x >= hms_res or y >= hms_res:
                        #print('not in', x, y)
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], hms_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], hms_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], hms_res)
                    aa,bb = max(0, ul[1]), min(br[1], hms_res)
                    hms[aa:bb,cc:dd,idx+1] = np.maximum(hms[aa:bb,cc:dd,idx+1], self.g[a:b,c:d])
        hms[:,:,0] = 1.0 - np.max(hms[:, :, 1:], axis=2)

        output = np.zeros(shape = ( output_res, output_res,num_parts+1), dtype = np.float32)
        for i in range(num_parts+1):
            output[:,:,i] = cv2.resize(hms[:,:,i], (output_res, output_res))

        return output

class GenerateWeightMask():
    def __init__(self,rad,weight):
        self.rad = rad
        self.g = weight*np.ones((2*rad+1,2*rad+1))

    def __call__(self, keypoints,output_res,num_parts,num_pafs,stride):
        mask = np.ones(shape=(output_res, output_res, num_parts+2*num_pafs + 1), dtype=np.float32)
        rad = self.rad
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[2]==2:
                    x, y = int(pt[0]/float(stride)), int(pt[1]/float(stride))
                    if x<0 or y<0 or x>=output_res or y>=output_res:
                        #print('not in', x, y)
                        continue
                    ul = int(x - rad), int(y - rad)
                    br = int(x + rad+1), int(y + rad+1)

                    c,d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], output_res)
                    aa,bb = max(0, ul[1]), min(br[1], output_res)

                    mask[aa:bb,cc:dd,idx+1] =  self.g[a:b,c:d]
                    mask[aa:bb, cc:dd, 0] = self.g[a:b, c:d]
        return mask

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta,visible_level):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] > visible_level or kpts[j][b][2] > visible_level:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector

def generate_fused_vector(vector, cnt, kpts, vec_pair, stride, theta,visible_level):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1
    vector_fused = np.zeros((vector.shape[0],vector.shape[1],2))

    for i in range(channel):
        vector_fused[:, :, 0] = vector_fused[:, :, 0] + vector[:,:,2 * i]
        vector_fused[:, :, 1] = vector_fused[:, :, 1] + vector[:, :, 2 * i+1]

    return vector_fused

class PoseData(data.Dataset):
    def __init__(self, args, transformer = None,isMask=False):
        self.args = args
        self.data_list = self.read(self.args['desc_file_path'])
        self.crop_size = int(self.args['crop_size'])
        self.rotate = int(self.args['rotate'])
        self.stride = int(self.args['stride'])
        self.visible_level = int(self.args['visible_level'])
        self.transformer = transformer

        self.vec_pair = [[0, 0, 0, 1, 1, 2, 4, 3, 5, 2, 3, 8, 9, 10,11],
                         [1, 2, 3, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12,13]]  # different from openpose
        # self.vec_pair = [[0, 1, 1, 2, 4, 3, 5, 2, 3, 8, 9, 10,11],
        #                  [1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12,13]]  # different from openpose
        self.isMask = isMask
        if 'theta' in self.args:
            self.theta = float(self.args['theta'])
        else:
            self.theta = 1.0
        self.sigma = float(self.args['sigma'])
        self.generateHeatmap = GenerateHeatmap(self.sigma)
        self.kernel = self.zyh_genGaussianKernel(self.sigma)

        self.dataset_id = self.args['dataset_id']

    def zyh_genGaussianKernel(self, sigma=1):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        kernel = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return kernel

    def zyh_genHeatmap(self, width, height, kpt, kernel):
        heatmap = np.zeros((14, height, width))
        bar = kernel.shape[0]
        for items in kpt:
            if items[2] == 0:
                x = int(items[0])
                y = int(items[1])
                up = x - bar / 2
                up_kernel = 0
                if up < 0:
                    up_kernel = 0 - up
                    up = 0
                down = x + bar / 2 + 1
                down_kernel = bar
                if down > height:
                    down_kernel = bar - (down - height)
                    down = height
                left = y - bar / 2
                left_kernel = 0
                if left < 0:
                    left_kernel = 0 - left
                    left = 0
                right = y + bar / 2 + 1
                right_kernel = bar 
                if right > width:
                    right_kernel = bar - (right - width)
                    right = width
                heatmap[items[3], up:down, left:right] = kernel[up_kernel:down_kernel, left_kernel:right_kernel]
        return heatmap

    def zyh_mixHeatmap(self, heatmap):
        return np.max(heatmap, axis=0)

    def __getitem__(self, index):
        image_path, json_path, ex_json_path = self.data_list[index]

        # print(1)
        no_ex = False
        if json_path == ex_json_path:
            no_ex = True

        # img = Image.open(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # cv2.imwrite('test.png',img)

        if self.dataset_id == 0:
            kpt, bb = read_json(json_path)
            ex_kpt, ex_bb = read_json(ex_json_path)
        elif self.dataset_id == 1:
            kpt, bb = read_json_1(json_path)
            ex_kpt, ex_bb = read_json_1(ex_json_path)

        if self.transformer is not None:
            img, kpt, ex_kpt = self.transformer(img, kpt, bb, ex_kpt, ex_bb)

        # width, height = img.size
        height, width = img.shape[0], img.shape[1]
        img_temp = np.zeros((height, width, 3))
        img_temp[:, :, 0], img_temp[:, :, 1], img_temp[:, :, 2] = img[:, :, 2], img[:, :, 1], img[:, :, 0]
        img = img_temp
        # mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape(
        #     (height / self.stride, width / self.stride, 1))

        heatmap = self.zyh_genHeatmap(width, height, kpt, self.kernel)
        if no_ex:
            mixed_heatmap = np.zeros((height, width))
        else:
            ex_heatmap = self.zyh_genHeatmap(width, height, ex_kpt, self.kernel)
            mixed_heatmap = self.zyh_mixHeatmap(ex_heatmap)

        # img = zyh_transforms.to_tensor(np.array(img)).div(255.0)
        img = zyh_transforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0], [255.0, 255.0, 255.0])  # mean, std
        heatmap = torch.Tensor(heatmap)
        mixed_heatmap = torch.Tensor(mixed_heatmap[np.newaxis, :, :])
        # print(2)

        return torch.cat((img, mixed_heatmap)), heatmap

    def getSourceInfo(self, index):
        img_path, json_path = self.data_list[index]
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        mask = np.ones((img.shape[0], img.shape[1]))
        kpt, center, scale = read_json(json_path, crop_size=self.crop_size)
        if self.transformer is not None:
            img, mask, kpt, center = self.transformer(img, mask, kpt, center, scale)
        height, width, _ = img.shape

        # mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape(
        #     (height / self.stride, width / self.stride, 1))
        #
        heatmap = np.zeros((height / self.stride, width / self.stride, len(kpt[0]) + 1), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, kpt, self.sigma, self.stride,self.visible_level)

        # heatmap = np.zeros((height , width , len(kpt[0]) + 1), dtype=np.float32)
        # heatmap = generate_heatmap(heatmap, kpt, 1, self.sigma,self.visible_level)
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background
        #
        #
        # vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
        # cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0])), dtype=np.int32)
        #
        # vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta,self.visible_level)
        #
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), [128.0, 128.0, 128.0],
                                   [255.0, 255.0, 255.0])  # mean, std
        # mask = Mytransforms.to_tensor(mask)
        heatmap = Mytransforms.to_tensor(heatmap)
        # vecmap = Mytransforms.to_tensor(vecmap)

        return img, heatmap

    def __len__(self):
        return len(self.data_list)





    def read(self, desc_file_path):
        import codecs

        result = []
        # data_root = self.args.get("data_root", None)

        with codecs.open(desc_file_path, 'r', 'utf-8') as txt:
            for line in txt:
                line = line.strip()
                image_path, json_path, ex_json_path = line.encode('utf-8').split(' ')

                result.append((image_path, json_path, ex_json_path))

            print("read from {} total {} ".format(desc_file_path, len(result)))

        return result



if __name__ == '__main__':
    args = dict()
    args['desc_file_path'] = '/root/ysy2/projects/mt_pose/human-pose-estimation/valid_files.txt'
    args['data_root'] = '/root/data-collection/AI_Challenger_keypoint/ai_challenger_keypoint_validation_20170911'
    args['crop_size'] = '384'
    args['rotate'] = 40
    args['stride'] = 2
    args['visible_level'] = 1
    args['sigma'] = 7
    args['theta'] = 2
    transform = Mytransforms.Compose([
                #Mytransforms.RandomHorizontalFlip(),
                Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(384),
                #Mytransforms.RandomHorizontalFlip(),
            ])
    p = PoseData(args, transform)

    import matplotlib
    import matplotlib.pyplot as plt

    for i, data in enumerate(p):

        #img, (heatmap, vecmap) = data
        img, map = data
        heatmap = map[:15,:,:]
        vecmap = map[15:,:,:]
        # print(i, img.shape, heatmap.shape, vecmap.shape)

        imgs = img.numpy()
        heats = heatmap.numpy()
        vectors = vecmap.numpy()


        for i in range(1):
            img = imgs[:, :, :]
            img = img.transpose(1, 2, 0)
            img *= 128
            img += 128
            img /= 255
            plt.imshow(img)
            plt.show()
            #plt.close()



            heatmaps = heats[:, :, :]
            heatmaps = heatmaps.transpose(1, 2, 0)
            heatmaps = cv2.resize(heatmaps, (384, 384))
            for j in range(0, 1):
                heatmap = heatmaps[:, :, j]
                heatmap = heatmap.reshape((384, 384, 1))
                heatmap *= 255
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # heatmap = heatmap.reshape((368,368,1))
                heatmap /= 255
                # result = heatmap * 0.4 + img * 0.5
                print j
                plt.imshow(img)
                plt.imshow(heatmap, alpha=0.5)
                plt.show()
                plt.close()

            vecs = vectors[:, :, :]
            vecs = vecs.transpose(1, 2, 0)
            vecs = cv2.resize(vecs, (384, 384))
            for j in range(0, 2, 2):
                vec = np.abs(vecs[:, :, j])
                vec += np.abs(vecs[:, :, j + 1])
                vec[vec > 1] = 1
                vec = vec.reshape((384, 384, 1))
                # vec[vec > 0] = 1
                vec *= 255
                vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
                vec = vec.reshape((384, 384))
                vec /= 255
                print j
                plt.imshow(img)
                # result = vec * 0.4 + img * 0.5
                plt.imshow(vec, alpha=0.5)
                plt.show()
                plt.close()

    print 'Done'




