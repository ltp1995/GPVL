import sys
import numpy as np
import os
import copy
import torch
import time
import os.path as osp
import json
import pickle
import warnings
import pickle
import math
from model.bert_tokenizer import BertTokenizer
import sqlite3
##
root = '/root/data1/ltp/datasets/openscene/mini/openscene-v1.1/meta_datas/mini/'
names = os.listdir(root)
aaa=[]
for name in names:
    pklpath = root + name
    pklfile = pickle.load(open(pklpath, 'rb'))
    aaa+=pklfile
with open('/root/data1/ltp/datasets/openscene/mini/openscene-v1.1/meta_datas/openscene_mini.pkl', 'wb') as f:
    pickle.dump(aaa, f)
##
infos2 = pickle.load(open('/root/data1/ltp/codes/ad/tmp/nupaln/openscene-v1.1/meta_datas/mini/2021.10.01.19.16.42_veh-28_02011_02410.pkl', 'rb'))
#infos_openscene=pickle.load(open('/root/data1/ltp/codes/ad/tmp/nupaln/openscene-v1.1/meta_datas/private_test_e2e/private_test_e2e.pkl', 'rb'))
infos_singapore = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_val_singapore.pkl', 'rb'))
infos = pickle.load(open('/root/data1/ltp/codes/ad/VAD/outputs_boston/res_singapore_epoch60.pkl', 'rb'))['bbox_results']
lens = len(infos)
score_l2_1, score_l2_2, score_l2_3, score_col_1, score_col_2, score_col_3=0,0,0,0,0,0
for info in infos:
    metric = info['metric_results']
    score_l2_1 += metric['plan_L2_1s']
    score_l2_2 += metric['plan_L2_2s']
    score_l2_3 += metric['plan_L2_3s']
    score_col_1 += metric['plan_obj_box_col_1s']
    score_col_2 += metric['plan_obj_box_col_1s']
    score_col_3 += metric['plan_obj_box_col_1s']
l2_1 = score_l2_1/lens
l2_2 = score_l2_2/lens
l2_3 = score_l2_3/lens
col_1 = score_col_1/lens
col_2 = score_col_2/lens
col_3 = score_col_3/lens
infos_singapore = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_val_singapore.pkl', 'rb'))['infos']
val_names = []
for info in infos_singapore:
    name = info['token']
    val_names.append(name)
infos_val = json.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/nuscenes_map_anns_val.json'))['GTs']
infos_val_singapore = {}
infos_val_singapore['GTs']=[]
for info in infos_val:
    name = info['sample_token']
    if name in val_names:
        infos_val_singapore['GTs'].append(info)
with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/nuscenes_map_anns_val_singapore.json', 'w') as f:
    json.dump(infos_val_singapore, f)
infos_te_singapore={}
infos_te_boston={}
infos_te_singapore['metadata']=infos_tr['metadata']
infos_te_boston['metadata']=infos_tr['metadata']
infos_te_boston['infos']=[]
infos_te_singapore['infos']=[]
locations=[]
for info in infos_tr['infos']:
    location = info['map_location']
    locations.append(location)
    if location=='boston-seaport':
       infos_te_boston['infos'].append(info)
    if location=='singapore-queenstown' or location=='singapore-onenorth' or location=='singapore-hollandvillage':
        infos_te_singapore['infos'].append(info)
a=set(locations)
with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_train_boston.pkl', 'wb') as f:
    pickle.dump(infos_te_boston, f)
with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_train_singapore.pkl', 'wb') as f:
    pickle.dump(infos_te_singapore, f)
##
root4 = '/root/data1/ltp/codes/ad/Omindrive/omnidrive_data/planning/train/'
root3 = '/root/data1/ltp/codes/ad/Omindrive/omnidrive_data/desc/train/'
root2 = '/root/data1/ltp/codes/ad/Omindrive/omnidrive_data/conv/train/'
root1 = '/root/data1/ltp/codes/ad/Omindrive/omnidrive_data/keywords/train/'
names = os.listdir(root4)
infos4_all=[]
for name in names:
    path4 = root4 + name
    path1 = root1 + name
    path2 = root2 + name
    path3 = root3 + name
    info1 = json.load(open(path1))
    info2 = json.load(open(path2))
    info3 = json.load(open(path3))
    info4 = json.load(open(path4))
    infos_all.append(info4)
##
captions = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene-versions/nuscene_v2/captions_nuscene_train.json'))
##
captions_desc={}
root1 = '/root/data1/ltp/codes/ad/Omindrive/omnidrive_data/desc/val/'
names = os.listdir(root1)
infos1_all=[]
for name in names:
    captions_tmp=[]
    path1 = root1 + name
    info1= json.load(open(path1))
    desc = info1['description']
    captions_tmp.append(desc)
    token_name = name[:-5]
    captions_desc[token_name]=captions_tmp
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene-versions/nuscene_v2/ominidrive/captions_desc_val.json', 'w') as f:
    json.dump(captions_desc, f)
######################################################################################
res= pickle.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/output/nuscene_base_det_motion_map_with_caption_prevtrajs_learnvisual_v2/nuscene-caption-lr9e-6-bs68-test6-train6f-bev-5e-6/res.pkl', 'rb'))
res_mine = pickle.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/output/nuscene_base_det_motion_map_with_caption_prevtrajs_learnvisual_v2/nuscene-caption-lr9e-6-bs68-test6-train6f-bev-5e-6/sent2np_trajs/step_5839_tv_trajs.pkl','rb'))
results = pickle.load(open('/root/data1/ltp/codes/ad/VAD/test/VAD_base_e2e/Tue_Jul__9_14_13_06_2024/pts_bbox/results_nusc.pkl', 'rb'))
#####
with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/trajs/trajs_val_last_3time.pkl', 'rb') as file:
    infos = pickle.load(file)
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/nuscene_infos/val_cmd.pickle', 'rb') as file:
    infos_cmd = pickle.load(file)
captions_nuscene_val = {}
for name in infos_cmd:
    sentences=[]
    cmd = infos_cmd[name]
    if cmd[0]==1:
        command = '{turn right}.'
    elif cmd[1]==1:
        command = '{turn left}.'
    elif cmd[2]==1:
        command = '{go straight}.'
    if name in infos:
        data = infos[name]
        result_str = "{" + ", ".join([f"[{x}, {y}]" for x, y in data]) + "}"
        sentence = 'The self-driving car will ' + command + ' ' + 'The trajectory of the car for the last 3 timestamps was ' + result_str + '.'
        sentences.append(sentence)
    else:
        sentence = 'The self-driving car will ' + command
        sentences.append(sentence)
    captions_nuscene_val[name]=sentences
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene-versions/nuscene_v2/captions_nuscene_val_prevtrajs.json', 'w') as f:
    json.dump(captions_nuscene_val, f)
##### captions with ego_status
captions_train_val_prev_trajs=json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train_val_prevtrajs.json'))
captions_train=json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train.json'))
ego_status_train = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/ego_status_train.pkl', 'rb'))
ego_status_val = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/ego_status_val.pkl', 'rb'))
ego_status_train.update(ego_status_val)
#######
captions_with_status_pretrain_nuscene_train={}
for token_name in captions_train_val_prev_trajs:
    caption1 = captions_train_val_prev_trajs[token_name]
    ego_lcf_feat = ego_status_train[token_name]['ego_lcf_feat']
    ego_lcf_feat=ego_lcf_feat.astype(np.float64)
    ego_lcf_feat2 = np.round(ego_lcf_feat, 1)
    speed = ", ".join([f"[{ego_lcf_feat2[0]}, {ego_lcf_feat2[1]}]"])
    acceleration = ", ".join([f"[{ego_lcf_feat2[2]}, {ego_lcf_feat2[3]}]"])
    size = ", ".join([f"[{ego_lcf_feat2[5]}, {ego_lcf_feat2[6]}]"])
    description_speed = 'The speed is' + ' ' + speed + '.'
    description_acceleration = 'The acceleration is' + ' '+ acceleration + '.'
    description_angular = 'The angular speed is' + ' ' + str(ego_lcf_feat2[4]) + '.'
    description_size = 'The box size is' + ' '+ size + '.'
    description_status = description_speed + ' ' + description_acceleration + ' ' + description_angular + ' ' + description_size
    caption_with_status = caption1[0] + ' ' + description_status
    caption_with_status_list=[]
    caption_with_status_list.append(caption_with_status)
    captions_with_status_pretrain_nuscene_train[token_name]=caption_with_status_list
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train_val_prevtrajs_status.json', 'w') as f:
    json.dump(captions_with_status_pretrain_nuscene_train, f)
a=1
### 1dfecb8189f54b999f4e47ddaa677fd0
#info = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_val.pkl', 'rb'))
# ann_captions_train = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train.json'))
# ann_infos= pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_train.pkl', 'rb'))['infos']
# ego_status_train = {}
# ego_status_val={}
# for ann_info in ann_infos:
#     token_name = ann_info['token']
#     his_trajs = ann_info['gt_ego_his_trajs']
#     lcf_feat = ann_info['gt_ego_lcf_feat']
#     ego_status_train[token_name]={}
#     ego_status_train[token_name]['ego_lcf_feat']=lcf_feat
#     ego_status_train[token_name]['ego_his_trajs']=his_trajs
# with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/ego_status_train.pkl', 'wb') as f:
#     pickle.dump(ego_status_train, f)
#################################
map_root = '/root/data1/ltp/codes/ad/VAD/data/nuscene_map_infos/train/'
ann_train_root = '/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl'
captions_ego = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train.json'))
ann_infos= pickle.load(open(ann_train_root, 'rb'))['infos']
captions_train = {}
numy=0
for ann_info in ann_infos:
    token_name= ann_info['token']
    ### maps
    map_path = map_root + token_name + '.pkl'
    if os.path.exists(map_path):
        map_pts = pickle.load(open(map_path, 'rb'))
        map_pts2 = map_pts[0].astype(np.float64)
        map_pts2 = np.round(map_pts2, 1)
        maps_tmp = ""
        for j in range(len(map_pts2)):
            pts2 = map_pts2[j]
            result_str3 = "{" + ", ".join([f"[{x}, {y}]" for x, y in pts2]) + "}"
            maps_tmp = maps_tmp + result_str3 + ' '
        description_maps = 'The locations of maps on the road are  ' + maps_tmp + '.'
        ### boxes
        #gt_boxes = ann_info['gt_boxes'][:, :3]
        gt_boxes = ann_info['gt_boxes'][:, :2]
        gt_boxes2 = np.round(gt_boxes, 1)
        points=gt_boxes2
        sorted_points = sorted(points, key=lambda point: point[0]**2 + point[1]**2)
        gt_boxes2=np.array(sorted_points)
        if len(gt_boxes2)>20:
            numy+=1
        gt_boxes2 = gt_boxes2[:20,:]
        result_str = "{" + ", ".join([f"[{x}, {y}]" for x, y in gt_boxes2]) + "}"
        description_boxes = 'The bounding boxes of other agents on the road are ' + result_str + '.'
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
        numx=1
        des_ego = captions_ego[token_name][0]
        for i in range(agent_nums):
            fut_trajs = gt_agent_fut_trajs3[i]
            xabs=abs(fut_trajs[5][0]-fut_trajs[0][0])
            yabs = abs(fut_trajs[5][1] - fut_trajs[0][1])
            if xabs>5 or yabs>5:
               numx+=1
               result_str2 = "{" + ", ".join([f"[{x}, {y}]" for x, y in fut_trajs]) + "}"
               trajs_tmp = trajs_tmp + result_str2 + ' '
        if numx>1:
            description_trajs = 'The future trajectories of other agents on the road are ' + trajs_tmp + '.'
            description2 = des_ego + ' '+ description_boxes + ' ' + description_trajs
        else:
            description2 = des_ego + ' ' + description_boxes
        des=[]
        des.append(description2)
        captions_train[token_name]=des
print(numy)
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_ego_det_motion_map.json', 'w') as f:
    json.dump(captions_train, f)
####################################################################################################
bert_tokenizer = BertTokenizer("/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/pretrained_weights/bert-base-uncased-vocab.txt")
cls_token = bert_tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
sep_token = bert_tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
map_root = '/root/data1/ltp/codes/ad/VAD/data/nuscene_map_infos/train/'
ann_train_root = '/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl'
ann_infos= pickle.load(open(ann_train_root, 'rb'))['infos']
captions_train = {}
map_root2 = '/root/data1/ltp/codes/ad/VAD/data/nuscene_det_map_trajs_tokenizer/map/train/'
det_root='/root/data1/ltp/codes/ad/VAD/data/nuscene_det_map_trajs_tokenizer/det/train/'
traj_root = '/root/data1/ltp/codes/ad/VAD/data/nuscene_det_map_trajs_tokenizer/traj/train/'
numx=0
for ann_info in ann_infos:
    token_name= ann_info['token']
    ### maps
    map_path = map_root + token_name + '.pkl'
    if os.path.exists(map_path):
        numx+=1
        print(numx)
        map_pts = pickle.load(open(map_path, 'rb'))
        map_pts2 = map_pts[0].astype(np.float64)
        map_pts2 = np.round(map_pts2, 1)
        maps_tmp = ""
        maps_all=[]
        for j in range(len(map_pts2)):
            pts2 = map_pts2[j]
            result_str3 = ", ".join([f"[{x}, {y}]" for x, y in pts2])
            tokenized_text = bert_tokenizer.tokenize(result_str3)
            txt_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)
            #print(txt_tokens.size())
            output = torch.zeros(250, dtype=torch.long)
            output[:len(txt_tokens)] = txt_tokens
            txt_tokens_tensor = torch.tensor(output).unsqueeze(0)
            maps_all.append(txt_tokens_tensor)
            #maps_tmp = maps_tmp + result_str3 + ' '
        maps_all_tensor = torch.cat(maps_all, dim=0)
        map_path2 = map_root2 + token_name + '.pt'
        torch.save(maps_all_tensor, map_path2)
        #description_maps = 'The locations of maps on the road are  ' + maps_tmp + '.'
        ### boxes
        #gt_boxes = ann_info['gt_boxes'][:, :3]
        gt_boxes = ann_info['gt_boxes'][:, :2]
        gt_boxes2 = np.round(gt_boxes, 1)
        ##
        boxes_all=[]
        for x, y in gt_boxes2:
            result_str = f"[{x}, {y}]"
            tokenized_text = bert_tokenizer.tokenize(result_str)
            txt_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)
            #print(txt_tokens.size())
            output = torch.zeros(15, dtype=torch.long)
            output[:len(txt_tokens)] = txt_tokens
            txt_tokens_tensor = torch.tensor(output).unsqueeze(0)
            boxes_all.append(txt_tokens_tensor)
        boxes_all_tensor = torch.cat(boxes_all, dim=0)
        box_path = det_root + token_name + '.pt'
        torch.save(boxes_all_tensor, box_path)
        #result_str = "{" + ", ".join([f"[{x}, {y}]" for x, y in gt_boxes2]) + "}"
        #description_boxes = 'The bounding boxes of other agents on the road are ' + result_str + '.'
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
        trajs_all=[]
        for i in range(agent_nums):
            fut_trajs = gt_agent_fut_trajs3[i]
            result_str2 = ", ".join([f"[{x}, {y}]" for x, y in fut_trajs])
            tokenized_text = bert_tokenizer.tokenize(result_str2)
            txt_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
            txt_tokens = torch.tensor(txt_tokens, dtype=torch.long)
            #
            print(txt_tokens.size())
            output = torch.zeros(80, dtype=torch.long)
            output[:len(txt_tokens)] = txt_tokens
            txt_tokens_tensor = torch.tensor(output).unsqueeze(0)
            trajs_all.append(txt_tokens_tensor)
            #trajs_tmp = trajs_tmp + result_str2 + ' '
        trajs_all_tensor = torch.cat(trajs_all, dim=0)
        traj_path = traj_root + token_name + '.pt'
        torch.save(trajs_all_tensor, traj_path)
        # description_trajs = 'The future trajectories of other agents on the road are ' + trajs_tmp + '.'
        # description1 = description_boxes + ' ' + description_trajs + ' ' + description_maps
        # des_ego = captions_ego[token_name][0]
        # description2 = des_ego + ' '+ description_boxes + ' ' + description_trajs
        # des=[]
        # des.append(description2)
        # captions_train[token_name]=des
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_det_motion_map.json', 'w') as f:
    json.dump(captions_train, f)