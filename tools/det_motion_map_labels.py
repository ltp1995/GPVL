import numpy as np
import pickle
import os
import json
import shutil
import cv2
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import visvalingamwyatt as vw
import sys
import re
###### generate texts of det motion map
map_infos = pickle.load(open('datasets/gpvl_maps_info_train.pkl', 'rb'))
ann_train_root = 'datasets/vad_nuscenes_infos_temporal_train.pkl'
ann_infos= pickle.load(open(ann_train_root, 'rb'))['infos']
captions_pretrain = {}
captions_pretrain_map={}
captions_pretrain_motion={}
captions_pretrain_det={}
pretrain_ids=[]
numx=0
def select_closest_points(matrix, top_n=10):
    points = matrix[:, :2]
    distances = np.sqrt(np.sum(points**2, axis=1))
    sorted_indices = np.argsort(distances)
    top_indices = sorted_indices[:top_n]
    return matrix[top_indices]
for ann_info in ann_infos:
    token_name= ann_info['token']
    ### maps
    map_classes_prompts=['the lane belongs to divider', 'the lane belongs to pedestrain crossing', 'the lane belongs to boundary']
    description_maps=''
    if token_name in map_infos:
        ### maps
        map_classes = map_infos[token_name][0]
        map_pts = map_infos[token_name][1]
        map_pts2 = map_pts.astype(np.float64)
        map_pts2 = np.round(map_pts2, 1)
        maps_tmp = ""
        for j in range(len(map_pts2)):
            ####
            label=map_classes[j]
            if label==1 or label==2:
                map_class_prompt = map_classes_prompts[label]
                ####
                pts2 = map_pts2[j]
                simplifier = vw.Simplifier(pts2)
                pts_simple=simplifier.simplify(number=4)
                map_pts_prompt = "{" + ", ".join([f"[{x},{y}]" for x, y in pts_simple]) + "}"
                maps_tmp = maps_tmp + map_pts_prompt + ','
        description_maps = 'Definition of lane location is [abscissa, ordinate], and lane locations are' + ' ' + maps_tmp[:-1] + '.'
        ### boxes
        gt_boxes = ann_info['gt_boxes'][:, [0, 1, 3, 4, 6]]
        gt_boxes2 = np.round(gt_boxes, 1)
        if len(gt_boxes2) > 10:
            gt_boxes2 = select_closest_points(gt_boxes2)
        result_str = "{" + ", ".join([f"[{x}, {y}, {w}, {l}, {a}]" for x, y, w, l, a in gt_boxes2]) + "}"
        description_boxes = 'Definition of object location is [abscissa, ordinate, width, length, rotation angle], and other object locations on the road are ' + result_str + '.'
        ### trajectory
        gt_agent_fut_trajs = ann_info['gt_agent_fut_trajs']
        agent_nums = gt_agent_fut_trajs.shape[0]
        gt_agent_fut_trajs = np.reshape(gt_agent_fut_trajs, (agent_nums, 6, 2))
        gt_agent_fut_trajs2 = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_boxes_trajs = gt_boxes[:, :2].reshape(agent_nums, 1, 2)
        gt_boxes_trajs2 = np.repeat(gt_boxes_trajs, 6, axis=1)
        gt_agent_fut_trajs3 = gt_agent_fut_trajs2 + gt_boxes_trajs2
        gt_agent_fut_trajs3 = np.round(gt_agent_fut_trajs3, 1)
        trajs_tmp = ""
        #### filter the trajs v2
        threshold = 0.3
        displacements = np.linalg.norm(np.diff(gt_agent_fut_trajs3, axis=1), axis=2)
        static_mask = np.all(displacements < threshold, axis=1)
        gt_agent_fut_trajs_filter = gt_agent_fut_trajs3[~static_mask]
        agent_nums=len(gt_agent_fut_trajs_filter)
        #### filter the trajs
        for i in range(agent_nums):
            fut_trajs = gt_agent_fut_trajs_filter[i]
            fut_trajs = fut_trajs[[1,3,5],:]
            result_str2 = "{" + ", ".join([f"[{x},{y}]" for x, y in fut_trajs]) + "}"
            trajs_tmp = trajs_tmp + result_str2 + ','
        description_trajs = 'Definition of trajectory is [abscissa, ordinate], and future trajectories of other agents in next three seconds are' +' ' + trajs_tmp[:-1] + '.'
        description_global = description_boxes + ' ' + description_trajs + ' ' + description_maps
        description_global_list=[]
        description_boxes_list=[]
        description_trajs_list=[]
        description_maps_list=[]
        description_global_list.append(description_global)
        description_boxes_list.append(description_boxes)
        description_trajs_list.append(description_trajs)
        description_maps_list.append(description_maps)
        captions_pretrain[token_name] = description_global_list
        captions_pretrain_det[token_name] = description_boxes_list
        captions_pretrain_motion[token_name] = description_trajs_list
        captions_pretrain_map[token_name] = description_maps_list
        pretrain_ids.append(token_name)
        numx+=1
        print(numx)
with open('datasets/nuscene_gpvl_v2/captions_pretrain_det_motion_map.json', 'w') as f:
    json.dump(captions_pretrain, f)
with open('datasets/nuscene_gpvl_v2/captions_pretrain_det.json', 'w') as f:
    json.dump(captions_pretrain_det, f)
with open('datasets/nuscene_gpvl_v2/captions_pretrain_motion.json', 'w') as f:
    json.dump(captions_pretrain_motion, f)
with open('datasets/nuscene_gpvl_v2/captions_pretrain_map.json', 'w') as f:
    json.dump(captions_pretrain_map, f)
with open('datasets/nuscene_gpvl_v2/pretrain_ids.json', 'w') as f:
    json.dump(pretrain_ids, f)