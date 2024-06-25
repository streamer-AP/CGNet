import argparse
import json
import os
import shutil
import time
from asyncio.log import logger

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from termcolor import cprint
from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from misc import tools
from misc.saver_builder import Saver
from misc.tools import MetricLogger, is_main_process
from models.tri_cropper import build_model
from tri_dataset import build_video_dataset as build_dataset
from tri_dataset import inverse_normalize
# from eingine.densemap_trainer import evaluate_counting, train_one_epoch
from tri_eingine import evaluate_similarity, train_one_epoch
from models.tri_sim_ot_b import similarity_cost
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import cv2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def read_pts(path):
    with open(path, "r") as f:
        lines = f.readlines()
        pts = []
        for line in lines:
            line = line.strip().split(",")
            pts.append([float(line[0]), float(line[1])])
        pts = np.array(pts)
    return pts


def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        # while apply ema model
        if k == "n_averaged":
            print(f"{k}:{v}")
            continue
        state_dict[k] = v
    return state_dict


def main(pair_cfg,  pair_ckpt):
    tools.init_distributed_mode(pair_cfg,)
    tools.set_randomseed(42 + tools.get_rank())
    # initilize the model
    model = model_without_ddp = build_model()
    model.load_state_dict(module2model(torch.load(pair_ckpt)["model"]))
    model.cuda()
    if pair_cfg.distributed:
        sync_model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            sync_model, device_ids=[pair_cfg.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build the dataset and dataloader
    dataset = build_dataset(pair_cfg.Dataset.test.root,
                            pair_cfg.Dataset.test.ann_dir)
    sampler = DistributedSampler(
        dataset, shuffle=False) if pair_cfg.distributed else None
    loader = DataLoader(dataset,
                        batch_size=pair_cfg.Dataset.val.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        num_workers=pair_cfg.Dataset.val.num_workers,
                        pin_memory=True)
    model.eval()
    video_results = {}
    interval = 15
    ttl = 5
    max_mem=5
    threshold=0.4
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            cnt_list = []
            video_name = labels["video_name"][0]
            img_names = labels["img_names"]
            w, h = labels["w"][0], labels["h"][0]

            img_name0 = img_names[0][0]
            pos_path0 = os.path.join(
                "locater/results", video_name, img_name0+".txt")
            pos0 = read_pts(pos_path0)
            z0 = model.forward_single_image(
                imgs[0, 0].cuda().unsqueeze(0), [pos0], True)[0]
            cnt_0 = len(pos0)
            cum_cnt = cnt_0
            cnt_list.append(cnt_0)
            selected_idx = [v for v in range(
                interval, len(img_names), interval)]
            pos_lists = []
            inflow_lists = []
            pos_lists.append(pos0)
            inflow_lists.append([1 for _ in range(len(pos0))])
            memory_features = [[z0[i]] for i in range(len(pos0))]
            ttl_list=[ttl for _ in range(len(pos0))]
            # if selected_idx[-1] != len(img_names)-1:
            #     selected_idx.append(len(img_names)-1)
            for i in selected_idx:
                img_name = img_names[i][0]
                pos_path = os.path.join(
                    "locater/results", video_name, img_name+".txt")
                pos = read_pts(pos_path)
                z = model.forward_single_image(
                    imgs[0, i].cuda().unsqueeze(0), [pos], True)[0]
                z = F.normalize(z, dim=-1)
                C = np.zeros((len(pos), len(memory_features)))
                for idx, pre_z in enumerate(memory_features):
                    pre_z = torch.stack(pre_z[-1:], dim=0).unsqueeze(0)
                    pre_z = F.normalize(pre_z, dim=-1)
                    sim_cost = torch.bmm(pre_z, z.unsqueeze(0).transpose(1, 2))
                    # sim_cost=1-similarity_cost(pre_z,z.unsqueeze(0))
                    sim_cost = sim_cost.cpu().numpy()[0]
                    sim_cost = np.min(sim_cost, axis=0)
                    C[:, idx] = sim_cost

                row_ind, col_ind = linear_sum_assignment(-C)
                sim_score = C[row_ind, col_ind]
                shared_mask = sim_score > threshold
                ori_shared_idx_list = col_ind[shared_mask]
                new_shared_idx_list = row_ind[shared_mask]
                outflow_idx_list = [i for i in range(len(pos)) if i not in row_ind[shared_mask]]

                for ori_idx, new_idx in zip(ori_shared_idx_list, new_shared_idx_list):
                    memory_features[ori_idx].append(z[new_idx])
                    ttl_list[ori_idx]=ttl
                for idx in outflow_idx_list:
                    memory_features.append([z[idx]])
                    ttl_list.append(ttl)
                pos_lists.append(pos)
                inflow_list = []
                for j in range(len(pos)):
                    if j in outflow_idx_list:
                        inflow_list.append(1)
                    else:
                        inflow_list.append(0)
                inflow_lists.append(inflow_list)
                cum_cnt += len(outflow_idx_list)
                cnt_list.append(len(outflow_idx_list))
                ttl_list=[ttl_list[idx]-1 for idx in range(len(ttl_list))]
                for idx in range(len(ttl_list)-1,-1,-1):
                    if ttl_list[idx]==0:
                        del memory_features[idx]
                        del ttl_list[idx]

            # conver numpy to list
            pos_lists = [pos_lists[i].tolist() for i in range(len(pos_lists))]

            video_results[video_name] = {
                "video_num": cum_cnt,
                "first_frame_num": cnt_0,
                "cnt_list": cnt_list,
                "frame_num": len(img_names),
                "pos_lists": pos_lists,
                "inflow_lists": inflow_lists,

            }
            print(video_name, video_results[video_name])
    with open("video_results_test.json", "w") as f:
        json.dump(video_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--pair_config", default="configs/crowd_sense.json")
    parser.add_argument(
        "--pair_ckpt", default="outputs/weights/best.pth")

    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if os.path.exists(args.pair_config):
        with open(args.pair_config, "r") as f:
            pair_configs = json.load(f)
        pair_cfg = edict(pair_configs)

    strtime = time.strftime('%Y%m%d%H%M') + "_" + os.path.basename(
        args.pair_config)[:-5]

    output_path = os.path.join(pair_cfg.Misc.tensorboard_dir, strtime)

    pair_cfg.Misc.tensorboard_dir = output_path
    pair_ckpt = args.pair_ckpt
    main(pair_cfg, pair_ckpt)
