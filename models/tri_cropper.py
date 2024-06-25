import torch
from torch import nn
from torch.nn import functional as F
from .backbone import build_backbone
from .tri_sim_ot_b import ot_similarity
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
        ind_points0=input["independ_pts0"][:,:input["independ_num0"],...]
        ind_points1=input["independ_pts1"][:,:input["independ_num1"],...]
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        ref_point1 = ref_points[:, 0, ...]
        ref_point2 = ref_points[:, 1, ...]
    
        z1_lists=[]
        z2_lists=[]
        y1_lists=[]
        y2_lists=[]
        for b in range(x1.shape[0]):
            z1_list=[]
            z2_list=[]
            y1_list=[]
            y2_list=[]
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
            for pt in ind_points0[b]:
                y1=self.get_crops(x1[b].unsqueeze(0),pt)
                y1=F.interpolate(y1,(224,224))
                y1_list.append(y1)
            for pt in ind_points1[b]:
                y2=self.get_crops(x2[b].unsqueeze(0),pt)
                y2=F.interpolate(y2,(224,224))
                y2_list.append(y2)
            y1=torch.cat(y1_list,dim=0)
            y2=torch.cat(y2_list,dim=0)
            y1=self.backbone(y1)[0].flatten(2).flatten(1)
            y2=self.backbone(y2)[0].flatten(2).flatten(1)
            y1_lists.append(y1)
            y2_lists.append(y2)
        y1=torch.stack(y1_lists,dim=0)
        y2=torch.stack(y2_lists,dim=0)
        z1=torch.stack(z1_lists,dim=0)
        z2=torch.stack(z2_lists,dim=0)
        z1=F.normalize(z1,dim=-1)
        z2=F.normalize(z2,dim=-1)
        y1=F.normalize(y1,dim=-1)
        y2=F.normalize(y2,dim=-1)
        return z1,z2,y1,y2
    def forward_single_image(self,x,pts,absolute=False):
        z_lists=[]
        max_batch=20
        for b in range(x.shape[0]):
            z_list=[]
            z_lists.append([])
            for pt in pts[b]:
                z=self.get_crops(x[b].unsqueeze(0),pt,absolute=absolute)
                z=F.interpolate(z,(224,224))
                z_list.append(z)
            z_list=[z_list[i:i+max_batch] for i in range(0,len(z_list),max_batch)]

            for z in z_list:

                z=torch.cat(z,dim=0)
            
                z=self.backbone(z)[0].flatten(2).flatten(1)
                z_lists[-1].append(z)
            z_lists[-1]=torch.cat(z_lists[-1],dim=0)
        z=torch.stack(z_lists,dim=0)
        z=F.normalize(z,dim=-1)
        return z
    def get_crops(self, z, pt, window_size=[32,32,32,64],absolute=False):
        h, w = z.shape[-2], z.shape[-1]
        if absolute:
            x_min = pt[0]-window_size[0]
            x_max = pt[0]+window_size[1]
            y_min = pt[1]-window_size[2]
            y_max = pt[1]+window_size[3]
        else:
            
            x_min = pt[0]*w-window_size[0]
            x_max = pt[0]*w+window_size[1]
            y_min = pt[1]*h-window_size[2]
            y_max = pt[1]*h+window_size[3]
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        z = z[..., y_min:y_max, x_min:x_max]
        # pos_emb = self.pos2posemb2d(pt, num_pos_feats=z.shape[1]//2).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # z = z + pos_emb
        # z=F.adaptive_avg_pool2d(z,(1,1))
        # z=z.squeeze(3).squeeze(2)
        return z

    def loss(self, z1,z2,y1,y2):
        loss_dict = self.ot_loss([z1,y1], [z2,y2])
        # loss = self.ot_loss(z1, z2)

        loss_dict["all"] = loss_dict["permutation_cost"]+loss_dict["neg_cost"]*0.1
        return loss_dict


def build_model():
    model = Model()
    return model
