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
        for i in range(num_feature_levels):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(num_channels[i], hidden_dim,
                          kernel_size=1, stride=1, padding=0),
                nn.GELU(),
            ))
            for j in range(num_feature_levels-1-i):
                input_proj_list[i] += nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
        self.input_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim*num_feature_levels, hidden_dim *
                      num_feature_levels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(hidden_dim*num_feature_levels*8*8,hidden_dim*num_feature_levels),
            nn.ReLU(),
            nn.Linear(hidden_dim*num_feature_levels,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim*num_feature_levels),
            nn.ReLU(),
        )
        self.input_proj = nn.ModuleList(input_proj_list)
        self.ot_loss = GML()

    def forward(self, input):
        x = input["image_pair"]
        ref_points = input["ref_pts"][:, :, :input["ref_num"], :]
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        ref_point1 = ref_points[:, 0, ...]
        ref_point2 = ref_points[:, 1, ...]
        z2 = self.backbone(x2)
        z2_list = []
        for i in range(len(z2)):
            z2_list.append(self.input_proj[i](z2[i]))
        z2 = torch.cat(z2_list, dim=1)
        z2 = self.input_fuse(z2)
        features_2 = []
        for pt in ref_point2[0]:
            features_2.append(self.get_feature(z2, pt))
            z1 = self.backbone(x1)
        z1_list = []
        for i in range(len(z1)):
            z1_list.append(self.input_proj[i](z1[i]))

        z1 = torch.cat(z1_list, dim=1)
        z1 = self.input_fuse(z1)

        features_1 = []

        for pt in ref_point1[0]:
            features_1.append(self.get_feature(z1, pt))

        features_1 = torch.stack(features_1, dim=1).flatten(2)
        features_2 = torch.stack(features_2, dim=1).flatten(2)
        return features_1, features_2

    def get_feature(self, z, pt, window_size=8):
        h, w = z.shape[-2], z.shape[-1]
        x_min = pt[0]*w-window_size//2
        x_max = pt[0]*w+window_size//2
        y_min = pt[1]*h-window_size//2
        y_max = pt[1]*h+window_size//2
        x_min, x_max, y_min, y_max = int(x_min), int(
            x_max), int(y_min), int(y_max)
        z = z[:, :, y_min:y_max, x_min:x_max]
        x_pad_left = 0
        x_pad_right = window_size-z.shape[-1]
        y_pad_top = 0
        y_pad_bottom = window_size-z.shape[-2]
        z = F.pad(z, (x_pad_left, x_pad_right, y_pad_top, y_pad_bottom))
        # pos_emb = self.pos2posemb2d(pt, num_pos_feats=z.shape[1]//2).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # z = z + pos_emb
        # z=F.adaptive_avg_pool2d(z,(1,1))
        # z=z.squeeze(3).squeeze(2)
        z = z.flatten(2).flatten(1)
        return z

    def loss(self, features1, features2):
        loss = self.ot_loss(features1, features2)
        loss_dict = {}
        loss_dict["all"] = loss
        return loss_dict
    def pos2posemb2d(self, pos, num_pos_feats=128, temperature=1000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb


def build_model():
    model = Model()
    return model
