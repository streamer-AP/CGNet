from math import sqrt
from typing import Iterable

import numpy as np
import torch
from scipy import spatial as ss

from utils import SmoothedValue, get_total_grad_norm, reduce_dict
from models.tri_sim_ot_b import similarity_cost
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum):
            vis[i] = False
        if dfs(a):
            ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign


def compute_metrics(pred_pts, pred_num, gt_pts, gt_num, sigma):
    if len(pred_pts) != 0 and gt_num == 0:
        fp = len(pred_pts)
        fn = 0
        tp = 0
    if len(pred_pts) == 0 and gt_num != 0:
        fn = gt_num
        fp = 0
        tp = 0
    if len(pred_pts) != 0 and gt_num != 0:

        pred_pts = pred_pts.cpu().detach().numpy()
        gt_pts = gt_pts.cpu().detach().numpy()
        print(pred_pts.shape, gt_pts.shape)
        dist_matrix = ss.distance_matrix(pred_pts, gt_pts, p=2)
        match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist <= sigma

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

        tp = tp_pred_index.shape[0]
        fp = fp_pred_index.shape[0]
        fn = fn_gt_index.shape[0]
    return tp, fp, fn


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, metric_logger: object, scaler: torch.cuda.amp.GradScaler, epoch, args):
    model.train()

    metric_logger.meters.clear()

    header = 'Epoch: [{}]'.format(epoch)
    metric_logger.set_header(header)
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for inputs, labels in metric_logger.log_every(data_loader):
        optimizer.zero_grad()
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.gpu)

        # y1,y2 = model(inputs)
        # if args.distributed:
        #     loss_dict = model.module.loss(y1,y2)
        # else:
        #     loss_dict = model.loss(y1,y2)
        z1,z2,y1,y2 = model(inputs)
        if args.distributed:
            loss_dict = model.module.loss(z1,z2,y1,y2)
        else:
            loss_dict = model.loss(z1,z2,y1,y2)

        all_loss = loss_dict["all"]
        loss_dict_reduced = reduce_dict(loss_dict)
        all_loss_reduced = loss_dict_reduced["all"]
        loss_value = all_loss_reduced.item()

        scaler.scale(all_loss).backward()
        scaler.unscale_(optimizer)

        if args.Misc.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.Misc.clip_max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(),
                                                  args.Misc.clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        for k in loss_dict_reduced.keys():
            metric_logger.update(**{k: loss_dict_reduced[k]})
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_similarity(model,data_loader,metric_logger,epoch,args):
    model.eval()
    metric_logger.meters.clear()
    header="Test"
    metric_logger.set_header(header)
    cnt=0
    for inputs,labels in metric_logger.log_every(data_loader):
        cnt+=1
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.gpu)
        # z1,z2=model(inputs)
        z1,z2,y1,y2=model(inputs)
        z1,z2,y1,y2=F.normalize(z1,dim=-1),F.normalize(z2,dim=-1),F.normalize(y1,dim=-1),F.normalize(y2,dim=-1)
        match_matrix=torch.bmm(torch.cat((z1,y1),dim=1),torch.cat((z2,y2),dim=1).transpose(1,2))
        all_match=linear_sum_assignment(1-match_matrix.cpu().detach().numpy()[0])
        
        pos_sim=1-similarity_cost(z1,z2).detach().cpu().numpy()[0]
        pos_match=np.argmax(pos_sim,axis=1)
        if cnt%100==0:
            print(cnt)
            print("pos_match")
            print(pos_match)
            print("all_match")
            print(all_match)
        pos_match_acc=pos_match==np.arange(pos_match.shape[0])
        all_match_acc=all_match[1][:pos_match.shape[0]]==np.arange(pos_match.shape[0])
        all_match_acc=all_match_acc.sum()/all_match_acc.shape[0]
        pos_match_acc=pos_match_acc.sum()/pos_match_acc.shape[0]
        metric_logger.update(pos_match_acc=pos_match_acc)
        metric_logger.update(all_match_acc=all_match_acc)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate_counting_and_locating(model, data_loader, metric_logger, epoch, args):
    model.eval()
    # criterion.eval()
    metric_logger.meters.clear()

    for prefix in ["default", "duplicate", "fuse"]:
        metric_logger.add_meter(f'{prefix}_mse',
                                SmoothedValue(window_size=1, fmt='{value:.5f}'))
        metric_logger.add_meter(f'{prefix}_mae',
                                SmoothedValue(window_size=1, fmt='{value:.1f}'))
        metric_logger.add_meter(f'{prefix}_tp',
                                SmoothedValue(window_size=1, fmt='{value:.1f}'))
        metric_logger.add_meter(f'{prefix}_fp',
                                SmoothedValue(window_size=1, fmt='{value:.1f}'))
        metric_logger.add_meter(f'{prefix}_fn',
                                SmoothedValue(window_size=1, fmt='{value:.1f}'))
        metric_logger.add_meter(f'{prefix}_cnt',
                                SmoothedValue(window_size=1, fmt='{value:.1f}'))
    header = "Test"
    metric_logger.set_header(header)
    sigma = 8

    for inputs, labels in metric_logger.log_every(data_loader):
        inputs = inputs.to(args.gpu)
        assert inputs.shape[0] == 1
        if args.distributed:
            out_dict = model.module.forward_points(inputs, threshold=0.9)
        else:
            out_dict = model.forward_points(inputs, threshold=0.9)
        metric_dict = {
            "cnt": torch.as_tensor(1., device=args.gpu),
        }
        for key in ["default_pts", "duplicate_pts", "fuse_pts"]:
            prefix = key.split("_")[0]
            gt_nums = labels[f"gt_{prefix}_num"].to(args.gpu).float()
            gt_pts = labels[f"gt_{prefix}_pts"][0,:gt_nums.long(),...].to(args.gpu).float()
            pred_pts = out_dict[key]
            mae = torch.abs(len(pred_pts)-gt_nums).data.mean()
            mse = ((len(pred_pts)-gt_nums)**2).data.mean()
            tp, fp, fn = compute_metrics(
                pred_pts, len(pred_pts), gt_pts, gt_nums, sigma)

            tp = torch.as_tensor(tp, device=args.gpu)
            fp = torch.as_tensor(fp, device=args.gpu)
            fn = torch.as_tensor(fn, device=args.gpu)
            metric_dict[f'{prefix}_mae'] = mae
            metric_dict[f'{prefix}_mse'] = mse
            metric_dict[f'{prefix}_tp'] = tp
            metric_dict[f'{prefix}_fp'] = fp
            metric_dict[f'{prefix}_fn'] = fn
        ########################################
        loss_dict_reduced = reduce_dict(metric_dict, average=True)
        for k in loss_dict_reduced.keys():
            metric_logger.update(**{k: loss_dict_reduced[k]})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.total for k, meter in metric_logger.meters.items()}
    print(metric_logger.meters["cnt"].total, metric_logger.meters["cnt"].count)
    for prefix in ["default", "duplicate", "fuse"]:
        tp = stats[f'{prefix}_tp']
        fp = stats[f'{prefix}_fp']
        fn = stats[f'{prefix}_fn']
        ap = tp / (tp + fp + 1e-7)
        ar = tp / (tp + fn + 1e-7)
        f1 = 2 * ap * ar / (ap + ar + 1e-7)
        stats[f'{prefix}_ap'] = ap
        stats[f'{prefix}_ar'] = ar
        stats[f'{prefix}_f1'] = f1
        stats[f"{prefix}_mae"] = stats[f"{prefix}_mae"]/stats["cnt"]
        stats[f"{prefix}_mse"] = sqrt(stats[f"{prefix}_mse"]/stats["cnt"])
    return stats
