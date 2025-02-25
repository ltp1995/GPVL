# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys

sys.path.append('')
import os
import re
import mmcv
import copy
import time
import json
import torch
import argparse
import warnings
import numpy as np
import os.path as osp

from tqdm import tqdm
from multiprocessing import Pool

torch.multiprocessing.set_sharing_strategy('file_system')
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
# from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric, PlanningMetricCOWA
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    # parser.add_argument('config',              type=str, help='test config file path')
    parser.add_argument('--config', default='projects/configs/VAD/VAD_base_e2e.py', help='test config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')
    parser.add_argument('--vad_pred_results', type=str)
    parser.add_argument('--llm_pred_root',
                        default='/root/data1/ltp/codes/vision-language/VALOR-mine/GPVL-AAAI25-v2/output/nuscene_pretrain_base_captv/nuscene-caption-5e-6-with-prevtrajs-captv/results_test_nuscene_cap/',
                        type=str)
    parser.add_argument('--out', type=str, help='output result file in pickle format')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', type=str, choices=['none', 'pytorch', 'slurm', 'mpi'], help='job launcher',
                        default='none')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='Whether to fuse conv and bn, this will slightly increasethe inference speed')
    parser.add_argument('--tmpdir', type=str,
                        help='tmp directory used for collecting results from multiple workers, available when gpu-collect is not specified')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    parser.add_argument('--with_old_eval_res', action='store_true',
                        help='whether to compare llm results with old vad results.')
    parser.add_argument('--need_cumsum', default=True, action='store_true', help='whether to cumsum waypoints.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def compute_planner_metric_stp3(
        planning_metric,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
):
    """Compute planner metric for one sample same as stp3."""
    metric_dict = {
        'plan_L2_1s': 0,
        'plan_L2_2s': 0,
        'plan_L2_3s': 0,
        'plan_obj_col_1s': 0,
        'plan_obj_col_2s': 0,
        'plan_obj_col_3s': 0,
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
    }
    metric_dict['fut_valid_flag'] = fut_valid_flag
    future_second = 3
    assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
    segmentation, pedestrian = planning_metric.get_label(gt_agent_boxes, gt_agent_feats)
    occupancy = torch.logical_or(segmentation, pedestrian)

    for i in range(future_second):
        if fut_valid_flag:
            cur_time = (i + 1) * 2
            traj_L2 = planning_metric.compute_L2(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time]
            )
            obj_coll, obj_box_coll = planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, :cur_time].detach(),
                gt_ego_fut_trajs[:, :cur_time],
                occupancy)
            metric_dict['plan_L2_{}s'.format(i + 1)] = traj_L2
            metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
            metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
        else:
            metric_dict['plan_L2_{}s'.format(i + 1)] = 0.0
            metric_dict['plan_obj_col_{}s'.format(i + 1)] = 0.0
            metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = 0.0

    return metric_dict


def eval_single(index, sample_idx, input_dict, llm_outputs, vad_outputs, planning_metric, with_old_eval_res,
                need_cumsum):
    # start = time.time()
    ego_fut_pred = torch.tensor(llm_outputs[sample_idx])
    if need_cumsum:
        ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
    ego_fut_trajs = input_dict['ego_fut_trajs'][0].data[0][0][0]
    ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
    gt_bbox = input_dict['gt_bboxes_3d'][0].data[0][0]
    gt_attr_label = input_dict['gt_attr_labels'][0].data[0][0]
    fut_valid_flag = input_dict['fut_valid_flag'][0].item()
    # print('Dur preprocess: {:.4f}s'.format(time.time() - start))

    # start = time.time()
    llm_eval_results = compute_planner_metric_stp3(
        planning_metric=planning_metric,
        pred_ego_fut_trajs=ego_fut_pred[None],
        gt_ego_fut_trajs=ego_fut_trajs[None],
        gt_agent_boxes=gt_bbox,
        gt_agent_feats=gt_attr_label.unsqueeze(0),
        fut_valid_flag=fut_valid_flag
    )
    # print('Dur cal_metrics: {:.4f}s'.format(time.time() - start))
    # print('[{:>4d}] llm = {}'.format(index, llm_eval_results))

    if with_old_eval_res:
        ego_fut_preds_vad = vad_outputs[index]['pts_bbox']['ego_fut_preds']
        ego_fut_cmd_vad = vad_outputs[index]['pts_bbox']['ego_fut_cmd'][0, 0, 0]
        ego_fut_cmd_vad_idx = torch.nonzero(ego_fut_cmd_vad)[0, 0]
        ego_fut_pred_vad = ego_fut_preds_vad[ego_fut_cmd_vad_idx]
        ego_fut_pred_vad = ego_fut_pred_vad.cumsum(dim=-2)
        vad_eval_results_new = compute_planner_metric_stp3(
            planning_metric=planning_metric,
            pred_ego_fut_trajs=ego_fut_pred_vad[None],
            gt_ego_fut_trajs=ego_fut_trajs[None],
            gt_agent_boxes=gt_bbox,
            gt_agent_feats=gt_attr_label.unsqueeze(0),
            fut_valid_flag=fut_valid_flag
        )
        vad_eval_results_old = vad_outputs[index]['metric_results']
        # print('[{:>4d}] vad_new = {}'.format(index, vad_eval_results_new))
        # print('[{:>4d}] vad_old = {}'.format(index, vad_eval_results_old))
    return llm_eval_results


def extr_waypoints(data):
    if isinstance(data, str):
        waypoints = re.findall(r'(\[[-]*[0-9]+\.[-]*[0-9]+,[-]*[0-9]+\.[-]*[0-9]+\])', data.replace(' ', ''))
        data = [eval(x) for x in waypoints]
    return data


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=16,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    if args.with_old_eval_res:
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE
    ######
    resnames = os.listdir(args.llm_pred_root)
    scores_all={}
    for resname in resnames:
        llm_pred_path = args.llm_pred_root + resname
        print('we are now evaluating method:', resname)
        planning_metric_type = cfg.model.get('planning_metric_type', 'nuscenes')
        if planning_metric_type == 'nuscenes':
            planning_metric = PlanningMetric()
        elif planning_metric_type == 'cowa':
            planning_metric = PlanningMetricCOWA()
        else:
            raise NotImplementedError

        if args.out and os.path.exists(args.out):
            outputs = mmcv.load(args.out)
        else:
            if args.vad_pred_results and os.path.exists(args.vad_pred_results):
                vad_outputs = mmcv.load(args.vad_pred_results)
            elif args.with_old_eval_res:
                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    vad_outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
                else:
                    model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                                                      broadcast_buffers=False)
                    vad_outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
            else:
                vad_outputs = None

            with open(llm_pred_path, 'r') as f:
                llm_outputs = json.load(f)
                llm_outputs = {x['video_id']: extr_waypoints(x['caption']) for x in llm_outputs}

            outputs = []
            numx = 0
            for i, data in enumerate(tqdm(data_loader)):
                sample_idx = data_loader.dataset.get_data_info(i)['sample_idx']
                llm_eval_results = eval_single(i, sample_idx, data, llm_outputs, vad_outputs, planning_metric,
                                               args.with_old_eval_res, args.need_cumsum)
                outputs.append(llm_eval_results)
            # mmcv.dump(outputs, args.out)

        print('-------------- Planning --------------')
        metric_dict = None
        num_valid = 0
        for res in outputs:
            if res['fut_valid_flag']:
                num_valid += 1
            else:
                continue
            if metric_dict is None:
                metric_dict = copy.deepcopy(res)
            else:
                for k in res.keys():
                    metric_dict[k] += res[k]

        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))
        ##
        scores_all[resname]=metric_dict
    import pickle
    with open('./output/nuscene_pretrain_base_captv/nuscene-caption-5e-6-with-prevtrajs-captv/scores.pkl', 'wb') as f:
        pickle.dump(scores_all, f)


if __name__ == '__main__':
    main()