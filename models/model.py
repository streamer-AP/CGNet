
import numpy as np
import torch
from mmcv.cnn import get_model_complexity_info
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F

from .backbone import build_backbone
from .head import build_head as build_locating_head
from .fuse import build_head as build_fuse_head
from .local_bl import build_loss, DrawDenseMap
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = build_backbone()
        self.locating_head = build_locating_head()
        self.fuse_head=build_fuse_head()
        self.pt2dmap = DrawDenseMap(1,0,2)
        # self.get_model_complexity(input_shape=(6, 1280, 720))
        
    @autocast("cuda")
    def forward(self, x):
        x1=x[:,0:3,:,:]
        x2=x[:,3:6,:,:]
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)

        default_counting_map = self.locating_head([z1,z2])
        duplicate_counting_map = self.locating_head([z2,z1])
        fuse_counting_map=self.fuse_head([z1,z2],default_counting_map,duplicate_counting_map)
        return {
            "default_counting_map": F.interpolate(default_counting_map,scale_factor=2),
            "duplicate_counting_map": F.interpolate(duplicate_counting_map,scale_factor=2),
            "fuse_counting_map":F.interpolate(fuse_counting_map,scale_factor=2)
        }

    def get_model_complexity(self, input_shape):
        flops, params = get_model_complexity_info(self, input_shape)
        return flops, params


    @torch.no_grad()
    def _map2points(self,predict_counting_map,kernel,threshold,loc_kernel_size,loc_padding):
        device=predict_counting_map.device
        max_m=torch.max(predict_counting_map)
        threshold=max(0.1,threshold*max_m)
        low_resolution_map=F.interpolate(F.relu(predict_counting_map),scale_factor=0.5)
        H,W=low_resolution_map.shape[-2],low_resolution_map.shape[-1]

        unfolded_map=F.unfold(low_resolution_map,kernel_size=loc_kernel_size,padding=loc_padding)
        unfolded_max_idx=unfolded_map.max(dim=1,keepdim=True)[1]
        unfolded_max_mask=(unfolded_max_idx==loc_kernel_size**2//2).reshape(1,1,H,W)

        predict_cnt=F.conv2d(low_resolution_map,kernel,padding=loc_padding)
        predict_filter=(predict_cnt>threshold).float()
        predict_filter=predict_filter*unfolded_max_mask
        predict_filter=predict_filter.detach().cpu().numpy().astype(bool).reshape(H,W)

        pred_coord_weight=F.normalize(unfolded_map,p=1,dim=1)
        
        coord_h=torch.arange(H).reshape(-1,1).repeat(1,W).to(device).float()
        coord_w=torch.arange(W).reshape(1,-1).repeat(H,1).to(device).float()
        coord_h=coord_h.unsqueeze(0).unsqueeze(0)
        coord_w=coord_w.unsqueeze(0).unsqueeze(0)
        unfolded_coord_h=F.unfold(coord_h,kernel_size=loc_kernel_size,padding=loc_padding)
        pred_coord_h=(unfolded_coord_h*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
        unfolded_coord_w=F.unfold(coord_w,kernel_size=loc_kernel_size,padding=loc_padding)
        pred_coord_w=(unfolded_coord_w*pred_coord_weight).sum(dim=1,keepdim=True).reshape(H,W).detach().cpu().numpy()
        coord_h=pred_coord_h[predict_filter].reshape(-1,1)
        coord_w=pred_coord_w[predict_filter].reshape(-1,1)
        coord=np.concatenate([coord_w,coord_h],axis=1)

        pred_points=[[4*coord_w+loc_kernel_size/2.,4*coord_h+loc_kernel_size/2.] for coord_w,coord_h in coord]
        return pred_points
    
    @torch.no_grad()
    def forward_points(self, x, threshold=0.8,loc_kernel_size=3):
        assert loc_kernel_size%2==1
        assert x.shape[0]==1
        out_dict=self.forward(x)

        loc_padding=loc_kernel_size//2
        kernel=torch.ones(1,1,loc_kernel_size,loc_kernel_size).to(x.device).float()
        default_counting_map=out_dict["default_counting_map"].detach().float()
        duplicate_counting_map=out_dict["duplicate_counting_map"].detach().float()
        fuse_counting_map=out_dict["fuse_counting_map"].detach().float()
        default_points=self._map2points(default_counting_map,kernel,threshold,loc_kernel_size,loc_padding)
        duplicate_points=self._map2points(duplicate_counting_map,kernel,threshold,loc_kernel_size,loc_padding)
        fuse_points=self._map2points(fuse_counting_map,kernel,threshold,loc_kernel_size,loc_padding)
        out_dict["default_pts"]=torch.tensor(default_points).to(x.device).float()
        out_dict["duplicate_pts"]=torch.tensor(duplicate_points).to(x.device).float()
        out_dict["fuse_pts"]=torch.tensor(fuse_points).to(x.device).float()
        return out_dict

    def loss(self, out_dict, targets):
        map_loss=build_loss()
        loss_dict = {}
        device=out_dict["default_counting_map"].device
        gt_default_maps,gt_duplicate_maps,gt_fuse_maps=[],[],[]
        map_loss=map_loss.to(device)
        with torch.no_grad():
            for idx in range(targets["gt_default_pts"].shape[0]):
                gt_default_map=self.pt2dmap(targets["gt_default_pts"][idx],targets["gt_default_num"][idx],targets["h"][idx],targets["w"][idx]).to(device)
                gt_duplicate_map=self.pt2dmap(targets["gt_duplicate_pts"][idx],targets["gt_duplicate_num"][idx],targets["h"][idx],targets["w"][idx]).to(device)
                gt_fuse_map=self.pt2dmap(targets["gt_fuse_pts"][idx],targets["gt_fuse_num"][idx],targets["h"][idx],targets["w"][idx]).to(device)
                gt_default_maps.append(gt_default_map)
                gt_duplicate_maps.append(gt_duplicate_map)
                gt_fuse_maps.append(gt_fuse_map)
            gt_default_maps=torch.stack(gt_default_maps,dim=0)
            gt_duplicate_maps=torch.stack(gt_duplicate_maps,dim=0)
            gt_fuse_maps=torch.stack(gt_fuse_maps,dim=0)

        loss_dict["default"] = map_loss(out_dict["default_counting_map"],gt_default_maps)
        loss_dict["duplicate"] = map_loss(out_dict["duplicate_counting_map"],gt_duplicate_maps)
        loss_dict["fuse"] = map_loss(out_dict["fuse_counting_map"],gt_fuse_maps)
        loss_dict["all"] = loss_dict["default"] + loss_dict["duplicate"] + loss_dict["fuse"]
        return loss_dict

def build_model():
    return Model()
