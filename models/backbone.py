from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from timm import create_model
from timm.models import features
import torch

class Backbone(nn.Module):
    def __init__(self, name: str,pretrained:bool,out_indices:List[int], train_backbone: bool):
        super(Backbone,self).__init__()
        backbone=create_model(name,pretrained=pretrained,features_only=True, out_indices=out_indices)
        self.train_backbone = train_backbone
        self.backbone=backbone
        self.out_indices=out_indices
        if not self.train_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(False)
    def forward(self,x):
        x=self.backbone(x)
        for i in range(len(x)):
            x[i]=F.relu(x[i])
            
        return x
    
    @property
    def feature_info(self):
        return features._get_feature_info(self.backbone,out_indices=self.out_indices)


def build_backbone():
    backbone = Backbone("convnext_small_384_in22ft1k", True, [3], True)
    # backbone = Backbone("convnext_small_384_in22ft1k", True, [2], True)

    # backbone = Backbone("resnet50.a3_in1k", True, [2], True)

    return backbone

if __name__=="__main__":
    model=build_backbone()
    x=torch.randn(1,3,224,224)
    z=model(x)
    for f in z:
        print(f.shape)
