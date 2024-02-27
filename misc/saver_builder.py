
import torch
import os
from .tools import is_main_process
class Saver():
    def __init__(self, args) -> None:
        self.save_dir=args.save_dir
        self.save_interval=args.save_interval
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir,exist_ok=True)
        self.save_best=args.save_best
        self.save_start_epoch = args.save_start_epoch
        self.min_value=0
        self.max_value=1e10
        self.reverse=args.reverse
        self.metric=args.metric

    def save(self, model, optimizer, scheduler, filename, epoch, stats={}):
        if is_main_process():
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'states':stats
            },os.path.join(self.save_dir, filename))

    def save_inter(self, model, optimizer, scheduler, name, epoch, stats={}):
        if epoch % self.save_interval == 0 and self.save_start_epoch <= epoch:
            self.save(model, optimizer, scheduler, name, epoch, stats)
    
    def save_on_master(self, model, optimizer, scheduler, epoch, stats={}):
        if is_main_process():
            self.save_inter(model, optimizer, scheduler, f"checkpoint{epoch:04}.pth", epoch, stats)
            
            if self.save_best and self.save_start_epoch <= epoch:
                if self.reverse and stats.test_stats[self.metric] > self.min_value:
                    self.min_value=max(self.min_value,stats.test_stats[self.metric])
                    self.save(model, optimizer, scheduler, f"best.pth", epoch, stats)
                elif not self.reverse and stats.test_stats[self.metric] < self.max_value:
                    self.max_value=min(self.max_value,stats.test_stats[self.metric])
                    self.save(model, optimizer, scheduler, f"best.pth", epoch, stats)

    def save_last(self, model, optimizer, scheduler, epoch, stats={}):
        if is_main_process():
            self.save(model, optimizer, scheduler, f"checkpoint_last.pth", epoch,stats)