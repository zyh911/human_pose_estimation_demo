from __future__ import division
import json
import numpy as np
import glob
import os
import tqdm
# import sys
# if sys.version_info<(3,0):
delta = 2 * np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                      0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                      0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
class singleResult(object):
    def __init__(self):
        pass
    
    def __str__(self):
        pass

def loadAnnoJson(json_path):
    
    file = open(json_path,'r') 
    anno_json = json.load(file)
    human_list = anno_json['human_list']
    file.close()
    keypoints = np.zeros((len(human_list),14,3))
    bboxs = np.zeros((len(human_list),4))
    for i,l in enumerate(human_list):
        pt_list = l['human_keypoints']
        human_rect_dict = l['human_rect'] 
        y,x,w,h = human_rect_dict['y'],human_rect_dict['x'],\
                    human_rect_dict['w'],human_rect_dict['h']
        bboxs[i,:] = np.array([y,x,w,h])
        for j,pt in enumerate(pt_list):
            pt_x = float(pt['x'])
            pt_y = float(pt['y'])
            pt_id = pt['id']
            pt_visible = pt['is_visible']
            keypoints[i,pt_id,0] = pt_x
            keypoints[i,pt_id,1] = pt_y
            # keypoints[i,pt_id,3] = pt_id
            keypoints[i,pt_id,2] = pt_visible
    annos = {}
    annos['all_keypoints'] = keypoints
    # print(keypoints)
    annos['all_bboxs'] = bboxs
    return annos   

def loadPredictJson(json_path):
    file = open(json_path,'r') 
    anno_json = json.load(file)
    human_list = anno_json['human_list']
    file.close()
    keypoints = np.zeros((len(human_list),14,3))
    bboxs = np.zeros((len(human_list),4))
    for i,l in enumerate(human_list):
        pt_list = l['human_keypoints']
        for j,pt in enumerate(pt_list):
            pt_x = float(pt['x'])
            pt_y = float(pt['y'])
            pt_id = pt['id']
            pt_visible = pt['is_visible']
            keypoints[i,pt_id,0] = pt_x
            keypoints[i,pt_id,1] = pt_y
            keypoints[i,pt_id,2] = pt_visible
    predicts = {}
    predicts['all_keypoints'] = keypoints
    return predicts

def loadPredictJsonWithScore(json_path):
    file = open(json_path,'r')
    anno_json = json.load(file)
    human_list = anno_json['human_list']
    file.close()
    keypoints = np.zeros((len(human_list),14,4))
    bboxs = np.zeros((len(human_list),4))
    for i,l in enumerate(human_list):
        pt_list = l['human_keypoints']
        for j,pt in enumerate(pt_list):
            pt_x = float(pt['x'])
            pt_y = float(pt['y'])
            pt_id = pt['id']
            pt_visible = pt['is_visible']
            keypoints[i,pt_id,0] = pt_x
            keypoints[i,pt_id,1] = pt_y
            keypoints[i,pt_id,2] = pt_visible
            keypoints[i,pt_id,3] = float(pt['score'])
    predicts = {}
    predicts['all_keypoints'] = keypoints
    return predicts

def loadPredictJsonWithScoreAndBk(json_path):
    file = open(json_path,'r')
    anno_json = json.load(file)
    human_list = anno_json['human_list']
    file.close()
    keypoints = np.zeros((len(human_list),14,5))
    bboxs = np.zeros((len(human_list),4))
    for i,l in enumerate(human_list):
        pt_list = l['human_keypoints']
        for j,pt in enumerate(pt_list):
            pt_x = float(pt['x'])
            pt_y = float(pt['y'])
            pt_id = pt['id']
            pt_visible = pt['is_visible']
            keypoints[i,pt_id,0] = pt_x
            keypoints[i,pt_id,1] = pt_y
            keypoints[i,pt_id,2] = pt_visible
            keypoints[i,pt_id,3] = float(pt['score'])
            keypoints[i, pt_id, 4] = float(pt['bk'])
    predicts = {}
    predicts['all_keypoints'] = keypoints
    return predicts

def savePredictJson(keypoints,savename):
    assert keypoints.shape[2]>=3, "keypoints dimention error: shall be n*14*(3+)"
    pad= True if keypoints.shape[2]==2 else False
    human_list = []    
    for i in range(keypoints.shape[0]):
        points = keypoints[i]
        points_list = []
        for j in range(points.shape[0]):
            pt =  points[j]
            if pt.shape[0]<4:
                x = int(pt[0])
                y = int(pt[1])
                z = int(pt[2])
                pt_dict = {'x':x,'y':y,'is_visible':z,'id':j}
            elif pt.shape[0]==4:
                x = int(pt[0])
                y = int(pt[1])
                z = int(pt[2])
                score = float(pt[3])
                pt_dict = {'x':x,'y':y,'is_visible':z,'id':j,'score':score}
            elif pt.shape[0]==5:
                x = int(pt[0])
                y = int(pt[1])
                z = int(pt[2])
                score = float(pt[3])
                bk = float(pt[4])
                pt_dict = {'x':x,'y':y,'is_visible':z,'id':j,'score':score,'bk':bk}
            points_list.append(pt_dict)
        human_dict = {'human_keypoints':points_list}
        human_list.append(human_dict)
    json_dict = {'human_list':human_list}
    j = json.dumps(json_dict,indent=4)
    f = open(savename,'w')
    f.write(j)
    f.close()

def calculateMAP(oks_all,oks_num):
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold)/np.float32(oks_num))
    mAP = np.mean(average_precision)
    return mAP

def calculatePCK():
    #TODO: PCK
    pass
# calculate oks
def calculateOKS(annos,predicts,delta):
    all_anno_keypoints = annos['all_keypoints']
    all_predicts_keypoints = predicts['all_keypoints']
    all_human_bboxs = annos['all_bboxs']

    anno_human_count = all_anno_keypoints.shape[0]
    predict_human_count  = all_predicts_keypoints.shape[0]

    if predict_human_count==0:
        print("Find one that no person detected!")
        return np.zeros((anno_human_count, 1)),anno_human_count,predict_human_count

    oks = np.zeros((anno_human_count, predict_human_count))

    for i in range(anno_human_count):
        anno = all_anno_keypoints[i]
        visible = anno[:,2] == 1
        bbox = all_human_bboxs[i]
        h = bbox[3]
        w = bbox[2]
        scale = np.float32(h*w)
        if np.sum(visible) == 0:
            for j in range(predict_human_count):
                oks[i, j] = 0
        else:
            for j in range(predict_human_count):
                predict = all_predicts_keypoints[j]
                dis = np.sum((anno[visible, :2] \
                    - predict[visible, :2])**2, axis=1)           
                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))

    return oks,anno_human_count,predict_human_count

def determinePersonByOKS(annos,predicts):
    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    oks,stdHmNum,preHmNum = calculateOKS(annos,predicts,delta)

    # if preHmNum==0:
    #     return -1
    # else:
    personIds = np.argmax(oks,axis=1)
    return personIds

def linkScores(annos,predicts,delta,personIds,thresh,isbk=False):
    all_anno_keypoints = annos['all_keypoints']
    all_predicts_keypoints = predicts['all_keypoints']
    all_human_bboxs = annos['all_bboxs']
    anno_human_count = all_anno_keypoints.shape[0]
    predict_human_count  = all_predicts_keypoints.shape[0]
    oks = np.zeros((anno_human_count, predict_human_count))
    result = []
    if predict_human_count==0:
        return result
    for i in range(anno_human_count):
        anno = all_anno_keypoints[i]
        person = personIds[i]
        predict = all_predicts_keypoints[person]
        visible = anno[:,2] == 1
        no_visible = anno[:,2] == 2
        bbox = all_human_bboxs[i]
        h = bbox[3]
        w = bbox[2]
        scale = np.float32(h*w)
        for j in range(anno.shape[0]):
            if predict[j,2]==0:
                continue
            part_dist = (anno[j,:2]-predict[j,:2])**2
            part_dist = np.sum(part_dist)
            part_oks = np.exp(-part_dist/2/delta[j]**2/(scale+1))
            if part_oks>thresh:
                if not isbk:
                    result.append([predict[j,3],anno[j,2]])
                else:
                    result.append([predict[j, 3],predict[j, 4],anno[j, 2]])
    return result

def countVisibleDir(anno_dir,predict_dir,thresh,isbk=False):
    predict_files = glob.glob(os.path.join(predict_dir, '*.json'))
    results = []
    for i,predict_file in enumerate(predict_files):
        predict_basename = os.path.basename(predict_file)
        print(str(i)+" comparing "+predict_basename)
        anno_file = os.path.join(anno_dir,predict_basename)
        if not os.path.exists(anno_file):
            print("No corresponding anno file found for {}".format(predict_basename))
            continue
        annos = loadAnnoJson(anno_file)
        if not isbk:
            predicts = loadPredictJsonWithScore(predict_file)
        else:
            predicts = loadPredictJsonWithScoreAndBk(predict_file)
        personIds = determinePersonByOKS(annos,predicts)
        result = linkScores(annos,predicts,delta,personIds,thresh,isbk)
        results = results+result
    return results

def getAnnoVisibleDir(anno_files):
    results = []
    for i, anno_file in enumerate(anno_files):
        anno_basename = os.path.basename(anno_file)
        print(str(i) + " getting " + anno_basename)
        annos = loadAnnoJson(anno_file)
        keypoints = annos['all_keypoints']
        for person in keypoints:
            for pt in person:
                results.append([pt[2]])

    return results


def getSingleImageOKS(annos,predicts):
    result = singleResult()
    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
                        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
                        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    oks,stdHmNum,preHmNum = calculateOKS(annos,predicts,delta)
    oks_num = np.max(oks.shape)
    oks = np.max(oks, axis=1)

    result.stdHmNum = stdHmNum
    result.preHmNum = preHmNum
    result.oks = oks
    result.oks_num = oks_num

    return oks,oks_num

def compareSingle(anno_dir,predict_file):
    predict_basename = os.path.basename(predict_file)
    anno_file = os.path.join(anno_dir,predict_basename)
    if not os.path.exists(anno_file):
        print("No corresponding anno file found for {}".format(predict_basename))
        return
    annos = loadAnnoJson(anno_file)
    predicts = loadPredictJson(predict_file)
    oks,num = getSingleImageOKS(annos,predicts)
    mAP = calculateMAP(oks,num)
    return mAP

def compareDir(anno_dir,predict_dir,detail_path = None):
    predict_files = glob.glob(os.path.join(predict_dir,'*.json'))
    oks_all = np.zeros((0))
    oks_num = 0
    all_infos = []
    if len(predict_files) ==0:
        print("No predict files found! Check predict_dir please!")
        return
    for i,predict_file in enumerate((predict_files)):
        predict_basename = os.path.basename(predict_file)
        print(str(i)+" comparing "+predict_basename)
        anno_file = os.path.join(anno_dir,predict_basename)
        if not os.path.exists(anno_file):
            print("No corresponding anno file found for {}".format(predict_basename))
            continue
        annos = loadAnnoJson(anno_file)
        predicts = loadPredictJson(predict_file)
        oks,num = getSingleImageOKS(annos,predicts)
        oks_all = np.concatenate((oks_all,oks),axis=0)
        oks_num +=num
        singleMAP = calculateMAP(oks,num)
        info = "{} {} {} {}".format(predict_basename.split('.')[0],
                                annos['all_keypoints'].shape[0],
                                predicts['all_keypoints'].shape[0],
                                singleMAP
                                ) 

    mAP = calculateMAP(oks_all,oks_num)
    return mAP
    
def saveDetail(details,save_path):
    
    pass




    
        










