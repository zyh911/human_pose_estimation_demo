import torch
import torch.nn as nn
import argparse
import os
import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import sys
import cv2
import matplotlib.pyplot as plt
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
def load_network(network, save_path):
    pretrained_dict = torch.load(save_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    pretrained_dict_ = {}
    for k, v in pretrained_dict.items():
        pretrained_dict_[k[7:]] = v
    pretrained_dict = pretrained_dict_
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network
def loadVisibleWeights(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    weights = [float(line) for line in lines]
    return weights


def load_pretrained(network, save_path,isDataParallel=False):
    pretrained_dict = torch.load(save_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    if isDataParallel:
        pretrained_dict_ = {}
        for k, v in pretrained_dict.items():
            pretrained_dict_[k[7:]] = v
        pretrained_dict = pretrained_dict_
    pretrained_dict.pop('module.classification_4.weight')
    pretrained_dict.pop('module.classification_4.bias')

    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #print pretrained_dict
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network
def normalize(origin_img):
    origin_img = np.array(origin_img, dtype=np.float32)
    origin_img -= 128.0
    origin_img /= 256.0

    return origin_img

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def process_45(model, input_path):
    limbSeq = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]
    linkSeq = [[1, 2],  [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]

    mapIdx = [[15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34],
              [35, 36], \
              [37, 38], [39, 40], [41, 42], [43, 44]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    boxsize = 384
    scale_search = [0.5, 1.0, 1.5]
    # scale_search = [1.0]
    stride = 4
    padValue = 0.
    thre_point = 0.05
    # thre_line = 0.05
    thre_line = 0.05
    stickwidth = 4
    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape
    # scale_search = [1.0]
    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 15))  # num_point
    paf_avg = np.zeros((height, width, 30))  # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, 32, padValue)

        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        mask = np.ones((1, 1, input_img.shape[2] / stride, input_img.shape[3] / stride), dtype=np.float32)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())
        mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())

        # get the features
        tic = time.time()
        output = model(input_var)
        toc = time.time()
        # print (' cnn processing time is %.5f' % (toc - tic))

        heat1 = output[:, :15, :, :]
        vec1 = output[:, 15:, :, :]

        # get the heatmap
        heatmap = heat1.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec1.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0

    for part in range(1, 15):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        # plt.imshow(map)
        # plt.show()

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 20  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x - 15 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1,
                                                                                        0)  # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    print(0.8 * len(score_midpts))
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3],
                                          reverse=True)  # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    '''
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j
                    '''

                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # print '%d person detected ' % subset.shape[0]

    # draw points
    canvas = cv2.imread(input_path)
    for i in range(14):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # draw lines
    for i in range(len(linkSeq)):
        for n in range(len(subset)):
            index = subset[n][np.array(linkSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def showHeatmapSum(heatmaps):
    heatmap_sum = np.zeros((heatmaps.shape[0], heatmaps.shape[1]))
    for i in range(1,15):
        heatmap_sum+=heatmaps[:,:,i]
    heatmap_sum[np.where(heatmap_sum<0)]=0
    max_num = np.max(heatmap_sum)
    if max_num>0:
        heatmap_sum = heatmap_sum/max_num * 255
    else:
        heatmap_sum = heatmap_sum*0
    # plt.imshow(heatmap_sum)
    # plt.show()
    heatmap_sum = heatmap_sum.astype(np.uint8)
    heatmap_sum = cv2.applyColorMap(heatmap_sum, cv2.COLORMAP_JET)
    cv2.imwrite("heatmap_temp.png",heatmap_sum)
    plt.imshow(heatmap_sum)
    plt.show()


def process_45_args(model, input_path,args):
    limbSeq = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]
    linkSeq = [[1, 2],  [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]

    mapIdx = [[15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34],
              [35, 36], \
              [37, 38], [39, 40], [41, 42], [43, 44]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    boxsize = args['boxsize']
    scale_search = [0.5, 1.0, 1.5]
    # scale_search = [1.0]
    stride = args['stride']
    padValue = 0.
    thre_point = 0.05
    # thre_line = 0.05
    thre_line = 0.05
    stickwidth = 4
    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape
    # scale_search = [1.0]
    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 15))  # num_point
    paf_avg = np.zeros((height, width, 30))  # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, 64, padValue)

        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        mask = np.ones((1, 1, input_img.shape[2] / stride, input_img.shape[3] / stride), dtype=np.float32)
        # print(input_img.shape)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())
        mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())

        # get the features
        tic = time.time()
        output = model(input_var)
        toc = time.time()
        # print (' cnn processing time is %.5f' % (toc - tic))

        heat1 = output[:, :15, :, :]
        vec1 = output[:, 15:, :, :]

        # get the heatmap
        heatmap = heat1.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        # showHeatmapSum(heatmap)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec1.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)

    # showHeatmapSum(heatmap_avg)

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0

    for part in range(1, 15):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        # plt.imshow(map)
        # plt.show()

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 20  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x - 15 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1,
                                                                                        0)  # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    # print(0.8 * len(score_midpts))
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3],
                                          reverse=True)  # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    '''
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j
                    '''

                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    # print '%d person detected ' % subset.shape[0]

    # draw points
    canvas = cv2.imread(input_path)
    for i in range(14):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    # draw lines
    for i in range(len(linkSeq)):
        for n in range(len(subset)):
            index = subset[n][np.array(linkSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def getsubsets(model, input_path,args = {'boxsize':384,'stride':4}):
    limbSeq = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]

    mapIdx = [[15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34],
              [35, 36], \
              [37, 38], [39, 40], [41, 42], [43, 44]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    boxsize = args['boxsize']
    scale_search = [0.5, 1.0, 1.5]
    # scale_search = [1.0]
    stride = args['stride']
    padValue = 0.
    thre_point = 0.05
    # thre_line = 0.05
    thre_line = 0.05
    stickwidth = 4
    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape

    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 15))  # num_point
    paf_avg = np.zeros((height, width, 30))  # num_vector

    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, 64, padValue)

        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        mask = np.ones((1, 1, input_img.shape[2] / stride, input_img.shape[3] / stride), dtype=np.float32)

        input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())
        mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())

        # get the features

        # print (' cnn processing time is %.5f' % (toc - tic))
        output = model(input_var)
        heat1 = output[:,:15,:,:]
        vec1 = output[:,15:,:,:]

        # get the heatmap
        heatmap = heat1.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec1.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)
    # plt.imshow(heatmap_avg[:,:,1])
    # plt.show()

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0


    for part in range(1, 15):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 20  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x - 15 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1,
                                                                                        0)  # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3],
                                          reverse=True)  # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    '''
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j
                    '''

                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])



    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 6 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    numOfPeople = subset.shape[0]
    keypoints = np.zeros((numOfPeople,14,4))

    for n in range(numOfPeople):
        for i in range(14):
            index = subset[n,i]
            if index==-1:
                x,y,z,score =0,0,0,0
            else:
                x = candidate[int(index), 0]
                y = candidate[int(index), 1]
                z = 1
                score = candidate[int(index), 2]
            keypoints[n,i,:] = np.array([x,y,z,score])

    # for n in range(numOfPeople):
    #     for i in range(14):
    #         # cv2.imshow('a',map_ori)
    #         x = int(keypoints[n, i, 0])
    #         y = int(keypoints[n, i, 1])
    #         cv2.circle(origin_img, (x,y), 4, colors[i], thickness=-1)
    #     cv2.imwrite('test.png',origin_img)

    return keypoints

def getsubsets_parallet(model, input_path,args = {'boxsize':384,'stride':4}):
    limbSeq = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 5], [5, 7], [4, 6], [6, 8], [3, 9], [4, 10], [9, 11],
               [10, 12], [11, 13], [12, 14]]

    mapIdx = [[15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32], [33, 34],
              [35, 36], \
              [37, 38], [39, 40], [41, 42], [43, 44]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    boxsize = args['boxsize']
    scale_search = [0.5, 1.0, 1.5]
    # scale_search = [1.0]
    stride = args['stride']
    padValue = 0.
    thre_point = 0.05
    # thre_line = 0.05
    thre_line = 0.05
    stickwidth = 4
    origin_img = cv2.imread(input_path)
    normed_img = normalize(origin_img)

    height, width, _ = normed_img.shape

    multiplier = [x * boxsize / height for x in scale_search]

    heatmap_avg = np.zeros((height, width, 15))  # num_point
    paf_avg = np.zeros((height, width, 30))  # num_vector



    for m in range(len(multiplier)):
        scale = multiplier[m]

        # preprocess
        imgToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgToTest_padded, pad = padRightDownCorner(imgToTest, 32, padValue)

        input_img = np.transpose(imgToTest_padded[:, :, :, np.newaxis], (3, 2, 0, 1))  # required shape (1, c, h, w)
        # mask = np.ones((1, 1, input_img.shape[2] / stride, input_img.shape[3] / stride), dtype=np.float32)
        input_img_parallent = np.zeros(2*input_img.shape[0],input_img.shape[1],input_img.shape[2],input_img.shape[3])
        input_img_parallent[0,:,:,:] = input_img
        input_img_parallent[1,:,:,:] = input_img
        input_var = torch.autograd.Variable(torch.from_numpy(input_img_parallent).cuda())
        # mask_var = torch.autograd.Variable(torch.from_numpy(mask).cuda())

        # get the features

        # print (' cnn processing time is %.5f' % (toc - tic))
        output = model(input_var)
        output = output[0,:,:,:]
        heat1 = output[:,:15,:,:]
        vec1 = output[:,15:,:,:]

        # get the heatmap
        heatmap = heat1.data.cpu().numpy()
        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)

        # get the paf
        paf = vec1.data.cpu().numpy()
        paf = np.transpose(np.squeeze(paf), (1, 2, 0))  # (h, w, c)
        paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf / len(multiplier)
    # plt.imshow(heatmap_avg[:,:,1])
    # plt.show()

    all_peaks = []  # all of the possible points by classes.
    peak_counter = 0


    for part in range(1, 15):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[:, 1:] = map[:, :-1]
        map_right = np.zeros(map.shape)
        map_right[:, :-1] = map[:, 1:]
        map_up = np.zeros(map.shape)
        map_up[1:, :] = map[:-1, :]
        map_down = np.zeros(map.shape)
        map_down[:-1, :] = map[1:, :]

        # get the salient point and its score > thre_point
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre_point))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # (w, h)

        # a point format: (w, h, score, number)
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    connection_all = []  # save all of the possible lines by classes.
    special_k = []  # save the lines, which haven't legal points.
    mid_num = 20  # could adjust to accelerate (small) or improve accuracy(large).

    for k in range(len(mapIdx)):

        score_mid = paf_avg[:, :, [x - 15 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]

        lenA = len(candA)
        lenB = len(candB)

        if lenA != 0 and lenB != 0:
            connection_candidate = []
            for i in range(lenA):
                for j in range(lenB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])  # the vector of BA
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the vector between A and B.
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in
                                      range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in
                                      range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * height / norm - 1,
                                                                                        0)  # ???
                    criterion1 = len(np.nonzero(score_midpts > thre_line)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            # sort the possible line from large to small order.
            connection_candidate = sorted(connection_candidate, key=lambda x: x[3],
                                          reverse=True)  # different from openpose, I think there should be sorted by x[3]
            connection = np.zeros((0, 5))

            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0: 3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # the number of A point, the number of B point, score, A point, B point
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(lenA, lenB):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):
                found = 0
                flag = [False, False]
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    # fix the bug, found == 2 and not joint will lead someone occur more than once.
                    # if more than one, we choose the subset, which has a higher score.
                    '''
                    if subset[j][indexA] == partAs[i]:
                        if flag[0] == False:
                            flag[0] = found
                            subset_idx[found] = j
                            flag[0] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[0]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[0]] = j
                    if subset[j][indexB] == partBs[i]:
                        if flag[1] == False:
                            flag[1] = found
                            subset_idx[found] = j
                            flag[1] = True
                            found += 1
                        else:
                            ids = subset_idx[flag[1]]
                            if subset[ids][-1] < subset[j][-1]:
                                subset_idx[flag[1]] = j
                    '''

                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found equals to 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])



    # delete som rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 6 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    numOfPeople = subset.shape[0]
    keypoints = np.zeros((numOfPeople,14,4))

    for n in range(numOfPeople):
        for i in range(14):
            index = subset[n,i]
            if index==-1:
                x,y,z,score =0,0,0,0
            else:
                x = candidate[int(index), 0]
                y = candidate[int(index), 1]
                z = 1
                score = candidate[int(index), 2]
            keypoints[n,i,:] = np.array([x,y,z,score])

    # for n in range(numOfPeople):
    #     for i in range(14):
    #         # cv2.imshow('a',map_ori)
    #         x = int(keypoints[n, i, 0])
    #         y = int(keypoints[n, i, 1])
    #         cv2.circle(origin_img, (x,y), 4, colors[i], thickness=-1)
    #     cv2.imwrite('test.png',origin_img)

    return keypoints