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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tri_dataset import build_dataset
# from eingine.densemap_trainer import evaluate_counting, train_one_epoch
from tri_eingine import evaluate_similarity, train_one_epoch
from misc import tools
from misc.saver_builder import Saver
from misc.tools import MetricLogger, is_main_process
from models.tri_cropper import build_model
from optimizer import optimizer_builder, scheduler_builder
from torch.nn import SyncBatchNorm
from torch.cuda.amp import GradScaler
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    tools.init_distributed_mode(args)
    tools.set_randomseed(42 + tools.get_rank())

    # initilize the model
    model = model_without_ddp = build_model()
    model.cuda()
    if args.distributed:
        sync_model=SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            sync_model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # build the dataset and dataloader
    dataset_train = build_dataset(args.Dataset.train.root,args.Dataset.val.ann_dir, args.Dataset.val.max_len,train=True,step=15)
    dataset_val = build_dataset(args.Dataset.val.root,args.Dataset.val.ann_dir, args.Dataset.val.max_len,train=True,step=15)
    # dataset_train = build_dataset(args.Dataset.train.root,args.Dataset.val.ann_dir, args.Dataset.val.max_len)
    # dataset_val = build_dataset(args.Dataset.val.root,args.Dataset.val.ann_dir, args.Dataset.val.max_len)
    sampler_train = DistributedSampler(dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(dataset_val, shuffle=False) if args.distributed else None
    loader_train = DataLoader(dataset_train,
                              batch_size=args.Dataset.train.batch_size,
                              sampler=sampler_train,
                              shuffle=False,
                              num_workers=args.Dataset.train.num_workers,
                              pin_memory=True)
    loader_val = DataLoader(dataset_val,
                            batch_size=args.Dataset.val.batch_size,
                            sampler=sampler_val,
                            shuffle=False,
                            num_workers=args.Dataset.val.num_workers,
                            pin_memory=True)

    optimizer = optimizer_builder(args.Optimizer, model_without_ddp)

    scheduler = scheduler_builder(args.Scheduler, optimizer)
    saver = Saver(args.Saver)
    logger = MetricLogger(args.Logger)
    if is_main_process():
        if args.Misc.use_tensorboard:
            tensorboard_writer = SummaryWriter(args.Misc.tensorboard_dir)
    scaler = GradScaler()
    for epoch in range(args.Misc.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        stats = edict()
        stats.train_stats = train_one_epoch(model, loader_train,
                                            optimizer, logger,scaler, epoch,
                                            args)
        train_log_stats = {
                **{f'train_{k}': v
                for k, v in stats.train_stats.items()}, 'epoch': epoch
            }
        scheduler.step()
        if is_main_process():
            for key, value in train_log_stats.items():
                cprint(f'{key}:{value}', 'green')
                if args.Misc.use_tensorboard:
                    tensorboard_writer.add_scalar(key, value, epoch)
        if epoch % args.Misc.val_freq == 0:
            stats.test_stats = evaluate_similarity(model,  loader_val, logger, epoch, args)

            saver.save_on_master(model, optimizer, scheduler, epoch, stats)
            test_log_stats = {
                **{f'val_{k}': v
                for k, v in stats.test_stats.items()}, 'epoch': epoch
            }
            if is_main_process():
                for key,value in test_log_stats.items():
                    cprint(f'{key}:{value}', 'green')
                    if args.Misc.use_tensorboard:
                        tensorboard_writer.add_scalar(key, value, epoch)
        else:
            saver.save_inter(model, optimizer, scheduler, f"checkpoint{epoch:04}.pth", epoch, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DenseMap Head ")
    parser.add_argument("--config", default="configs/crowd_sense.json")
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            configs = json.load(f)
        cfg = edict(configs)
    print(cfg)

    strtime = time.strftime('%Y%m%d%H%M') + "_" + os.path.basename(
        args.config)[:-5]

    output_path = os.path.join(cfg.Misc.tensorboard_dir, strtime)
    if is_main_process():
        if not os.path.exists(cfg.Misc.tensorboard_dir):
            os.mkdir(cfg.Misc.tensorboard_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        # copy the config file to the output directory
        shutil.copy(args.config, output_path)

    cfg.Saver.save_dir = os.path.join(output_path, "checkpoints")
    cfg.Misc.tensorboard_dir = output_path

    main(cfg)
