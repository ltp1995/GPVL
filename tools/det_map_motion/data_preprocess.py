import sys
sys.path.append('')
import numpy as np
import argparse
import os
import copy
import torch
import time
import os.path as osp
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import pickle
## fut trajs
gt_train_infos = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_train.pkl', 'rb'))['infos']
gt_val_infos= pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_val.pkl', 'rb'))['infos']
captions_nuscene_train_fut_trajs={}
for info in gt_val_infos:
    his_trajs= info['gt_ego_his_trajs']
    fut_trajs= info['gt_ego_fut_trajs']
    cmd = info['gt_ego_fut_cmd']
    token_name = info['token']
    sentences = []
    if cmd[0] == 1:
        command = '{turn right}.'
    elif cmd[1] == 1:
        command = '{turn left}.'
    elif cmd[2] == 1:
        command = '{go straight}.'
    data = fut_trajs.astype(np.float64)
    data = np.round(data, 1)
    result_str = "{" + ", ".join([f"[{x}, {y}]" for x, y in data]) + "}"
    sentence = 'The self-driving car will ' + command + ' ' + 'The future trajectory in the next 6 timestamps is ' + result_str + '.'
    sentences.append(sentence)
    captions_nuscene_train_fut_trajs[token_name] = sentences
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene-versions/nuscene_v2/captions_nuscene_val.json', 'w') as f:
    json.dump(captions_nuscene_train_fut_trajs, f)
########################################################################
## prev trajs
gt_train_infos = pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_train.pkl', 'rb'))['infos']
gt_val_infos= pickle.load(open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-origin/vad_nuscenes_infos_temporal_val.pkl', 'rb'))['infos']
train_val_fut_his_trajs={}
for info in gt_train_infos:
    his_trajs= info['gt_ego_his_trajs']
    fut_trajs= info['gt_ego_fut_trajs']
    cmd_vad = info['gt_ego_fut_cmd']
    token_name = info['token']
    train_val_fut_his_trajs[token_name]={}
    train_val_fut_his_trajs[token_name]['his_trajs']=his_trajs
    train_val_fut_his_trajs[token_name]['fut_trajs']=fut_trajs
    train_val_fut_his_trajs[token_name]['cmd'] = cmd_vad
for info in gt_val_infos:
    his_trajs= info['gt_ego_his_trajs']
    fut_trajs= info['gt_ego_fut_trajs']
    cmd_vad = info['gt_ego_fut_cmd']
    token_name = info['token']
    train_val_fut_his_trajs[token_name]={}
    train_val_fut_his_trajs[token_name]['his_trajs']=his_trajs
    train_val_fut_his_trajs[token_name]['fut_trajs']=fut_trajs
    train_val_fut_his_trajs[token_name]['cmd'] = cmd_vad
captions_nuscene_train_val_prevtrajs = {}
for name in train_val_fut_his_trajs:
    sentences=[]
    cmd = train_val_fut_his_trajs[name]['cmd']
    if cmd[0]==1:
        command = '{turn right}.'
    elif cmd[1]==1:
        command = '{turn left}.'
    elif cmd[2]==1:
        command = '{go straight}.'
    data = train_val_fut_his_trajs[name]['his_trajs']
    data = data.astype(np.float64)
    data = np.round(data, 1)
    result_str = "{" + ", ".join([f"[{x}, {y}]" for x, y in data]) + "}"
    sentence = 'The self-driving car will ' + command + ' ' + 'The trajectory of the car for the last 2 timestamps was ' + result_str + '.'
    sentences.append(sentence)
    captions_nuscene_train_val_prevtrajs[name]=sentences
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene-versions/nuscene_v2/captions_nuscene_train_val_prevtrajs.json', 'w') as f:
    json.dump(captions_nuscene_train_val_prevtrajs, f)
######################################
# info=json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/aaa.json'))
# info2=info[:50000]
# a=torch.tensor(info2, dtype=torch.long)
######
# train_names = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/train_ids.json'))
#captions = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_det_motion_map.json'))
#captions_ego = json.load(open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_train.json'))
# #######
# captions_ego_det_map_motion={}
# for token_name in captions_ego:
#     caption3=[]
#     caption1=captions_ego[token_name][0]
#     if token_name in captions:
#         caption2 = captions[token_name][0]
#         caption = caption1 + ' ' + caption2
#     else:
#         caption = caption1
#     caption3.append(caption)
#     captions_ego_det_map_motion[token_name]=caption3
# with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_ego_det_motion_map.json', 'w') as f:
#     json.dump(captions_ego_det_map_motion, f)
# ##
# train_names_3=[]
# for name in captions:
#     train_names_3.append(name)
# with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/train_ids_det_motion_map.json', 'w') as f:
#     json.dump(train_names_3, f)
map_root = '/root/data1/ltp/codes/ad/VAD/data/nuscene_map_infos/train/'
#############################################################################################################################################
ann_train_root = '/root/data1/ltp/codes/ad/VAD/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl'
ann_infos= pickle.load(open(ann_train_root, 'rb'))['infos']
captions_train = {}
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
        for i in range(agent_nums):
            fut_trajs = gt_agent_fut_trajs3[i]
            result_str2 = "{" + ", ".join([f"[{x}, {y}]" for x, y in fut_trajs]) + "}"
            trajs_tmp = trajs_tmp + result_str2 + ' '
        description_trajs = 'The future trajectories of other agents on the road are ' + trajs_tmp + '.'
        description1 = description_boxes + ' ' + description_trajs + ' ' + description_maps
        des_ego = captions_ego[token_name][0]
        description2 = des_ego + ' '+ description_boxes + ' ' + description_trajs
        des=[]
        des.append(description2)
        captions_train[token_name]=des
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_det_motion_map.json', 'w') as f:
    json.dump(captions_train, f)######
# with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes/trajs/trajs_val_last_3time.pkl', 'rb') as file:
#     infos = pickle.load(file)
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
with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/captions_nuscene_val_prevtrajs.json', 'w') as f:
    json.dump(captions_nuscene_val, f)
# with open('/root/data1/ltp/codes/vision-language/VALOR-mine/VALOR-v1/datasets/nuscene/names_nuscene_val.json', 'w') as f:
#     json.dump(names_nuscene_val, f)
#from clip.tokenizers import SimpleTokenizer
# clip_tokenizer = SimpleTokenizer()
# tokenized_text_clip = clip_tokenizer.encode(text)
# bert_tokenizer = BertTokenizer("./pretrained_weights/bert-base-uncased-vocab.txt")
# tokenized_text_bert = bert_tokenizer.tokenize(text)
###############################
train_traj_infos={}
train_cmd_infos={}
with open('/root/data1/ltp/codes/ad/VAD/tmp/nuscene/vad_nuscenes_infos_temporal_train.pkl', 'rb') as file:
      data_train = pickle.load(file)
infos_train = data_train['infos']
arrays=[]
arrays1=[]
names_train={}
names_val={}
imnames = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
for info in infos_train:
    name = info['token']
    ## names
    paths =[]
    cams = info['cams']
    for imname in imnames:
        path = cams[imname]['data_path']
        paths.append(path)
    names_train[name]=paths
    ## trajs
    trajs = info['gt_ego_fut_trajs_ltp']
    cmd = info['gt_ego_fut_cmd']
    trajs= trajs[:,:2]
    #trajs = trajs * 100
    # trajs = (trajs + 13.965177) * 388.62618622
    # trajs=(trajs+4.16458)*701.592237
    #trajs = np.round(trajs).astype(int)
    trajs = np.round(trajs, 1)
    trajs = np.array([['{:.1f}'.format(x) for x in row] for row in trajs])
    train_traj_infos[name]=trajs
    train_cmd_infos[name]=cmd
    arrays.append(trajs)
#with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-ltp/train_paths_valor.json', 'w') as f:
#    json.dump(names_train, f)
with open("/root/data1/ltp/codes/ad/VAD/data/nuscenes-ltp/train_trajs_valor_v2.pickle", "wb") as file:
    pickle.dump(train_traj_infos, file)
# ########################################################## 25.731668
# with open('/root/data1/ltp/codes/ad/VAD/tmp/nuscene/vad_nuscenes_infos_temporal_val.pkl', 'rb') as file:
#     data_val = pickle.load(file)
# val_traj_infos={}
# val_cmd_infos={}
# infos_val = data_val['infos']
# for info in infos_val:
#     name = info['token']
#     ## names
#     paths = []
#     cams = info['cams']
#     for imname in imnames:
#         path = cams[imname]['data_path']
#         paths.append(path)
#     names_val[name] = paths
#     ## trajs
#     trajs = info['gt_ego_fut_trajs_ltp']
#     trajs = trajs[:, :2]
#     #trajs = trajs*100
#     #trajs = (trajs+13.965177)*388.62618622
#     #trajs =(trajs + 4.16458) * 701.592237
#     #trajs=np.round(trajs).astype(int)
#     trajs = np.round(trajs, 1)
#     trajs = np.array([['{:.1f}'.format(x) for x in row] for row in trajs])
#     cmd = info['gt_ego_fut_cmd']
#     val_traj_infos[name]=trajs
#     val_cmd_infos[name]=cmd
#     arrays.append(trajs)
# #with open('/root/data1/ltp/codes/ad/VAD/data/nuscenes-ltp/val_paths_valor.json', 'w') as f:
# #    json.dump(names_val, f)
# with open("/root/data1/ltp/codes/ad/VAD/data/nuscenes-ltp/val_trajs_valor_v2.pickle", "wb") as file:
#     pickle.dump(val_traj_infos, file)