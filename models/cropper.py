import torch
from torch import nn
from torch.nn import functional as F
from .backbone import build_backbone
from .tri_sim_ot_b import GML
import math
import copy
from typing import Optional, List, Dict, Tuple, Set, Union, Iterable, Any
from torch import Tensor


class Model(nn.Module):
    def __init__(self, stride=16, num_feature_levels=3, num_channels=[96, 192, 384], hidden_dim=256, freeze_backbone=False) -> None:
        super().__init__()
        self.backbone = build_backbone()
        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
        self.stride = stride
        self.num_feature_levels = num_feature_levels
        self.num_channels = num_channels
        input_proj_list = []
        
        self.ot_loss = GML()

    def forward(self, input):
        x = input["image_pair"]
        ref_points = input["ref_pts"][:, :, :input["ref_num"], :]
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        ref_point1 = ref_points[:, 0, ...]
        ref_point2 = ref_points[:, 1, ...]
    
        z1_lists=[]
        z2_lists=[]
        for b in range(x1.shape[0]):
            z1_list=[]
            z2_list=[]
            for pt1,pt2 in zip(ref_point1[b],ref_point2[b]):
                z1=self.get_crops(x1[b].unsqueeze(0),pt1)
                z2=self.get_crops(x2[b].unsqueeze(0),pt2)
                z1=F.interpolate(z1,(224,224))
                z2=F.interpolate(z2,(224,224))
                z1_list.append(z1)
                z2_list.append(z2)
            z1=torch.cat(z1_list,dim=0)
            z2=torch.cat(z2_list,dim=0)
            z1=self.backbone(z1)[0].flatten(2).flatten(1)
            z2=self.backbone(z2)[0].flatten(2).flatten(1)
            z1_lists.append(z1)
            z2_lists.append(z2)
        z1=torch.stack(z1_lists,dim=0)
        z2=torch.stack(z2_lists,dim=0)
        return z1,z2
            
    def get_crops(self, z, pt, window_size=64):
        h, w = z.shape[-2], z.shape[-1]
        x_min = pt[0]*w-window_size//2
        x_max = pt[0]*w+window_size//2
        y_min = pt[1]*h-window_size//2
        y_max = pt[1]*h+window_size//2
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        z = z[:, :, y_min:y_max, x_min:x_max]
        # pos_emb = self.pos2posemb2d(pt, num_pos_feats=z.shape[1]//2).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # z = z + pos_emb
        # z=F.adaptive_avg_pool2d(z,(1,1))
        # z=z.squeeze(3).squeeze(2)
        return z

    def loss(self, features1, features2):
        loss = self.ot_loss(features1, features2)
        loss_dict = {}
        loss_dict["all"] = loss
        return loss_dict


def build_model():
    model = Model()
    return model
