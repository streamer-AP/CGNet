from typing import Callable, Optional
from torchvision.datasets import VisionDataset
from PIL import Image
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
def transform():
    return A.Compose([
        A.Resize(720,1280),
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def video_transform():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
def inverse_normalize(img):
    img=img*torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img=img+torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    img=img*255
    return img

class VideoDataset(VisionDataset):
    def __init__(self, root: str, annotation_dir:str,transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths=os.listdir(root)
        self.video_paths=[os.path.join(root,video_path) for video_path in self.video_paths]
        self.annotations_paths=os.listdir(annotation_dir)
        self.annotations_paths=[os.path.join(annotation_dir,annotation) for annotation in self.annotations_paths]
        self.annotation={}
        self.videos=[]
        for annotation_path in self.annotations_paths:
            with open(annotation_path) as f:
                video_name=annotation_path.split('/')[-1].split('.')[0]

                self.annotation[video_name]=[]
                lines=f.readlines()
                for line in lines:
                    line=line.split()
                    file_name=line[0]
                    width,height=int(line[1]),int(line[2])
                    
                    data=[float(x) for x in line[3:] if x!=""]
                    cnt=len(data)//7


                    if len(data)>0:
                        ids=-1*np.ones((cnt,1))
                        pts=-1*np.ones((cnt,2))
                        bboxes=-1*np.ones((cnt,4))
                        data=np.array(data)
                        data=np.reshape(data,(-1,7))
                        ids=data[:,6].reshape(-1,1)
                        pts=data[:,4:6]
                        bboxes[:]=data[:,0:4]
                        bboxes[:,2]=bboxes[:,2]-bboxes[:,0]
                        bboxes[:,3]=bboxes[:,3]-bboxes[:,1]
                        pts[:,0]=pts[:,0]/width
                        pts[:,1]=pts[:,1]/height
                        bboxes[:,0]=bboxes[:,0]/width
                        bboxes[:,1]=bboxes[:,1]/height
                        bboxes[:,2]=bboxes[:,2]/width
                        bboxes[:,3]=bboxes[:,3]/height
                    else:
                        ids=-1*np.ones((1,1))
                        pts=-1*np.ones((1,2))
                        bboxes=-1*np.ones((1,4))
                    self.annotation[video_name].append({"file_name":file_name,"height":height,"width":width,"ids":ids,"pts":pts,"bboxes":bboxes,"cnt":cnt})
        for video_path in self.video_paths:
            video_name=video_path.split('/')[-1]
            self.videos.append({
                    "video_name":video_name,
                    "img_names":[],
                    "height":self.annotation[video_name][0]["height"],
                    "width":self.annotation[video_name][0]["width"],
                    "ids":[],
                    "pts":[],
                    "bboxes":[],
                    "cnt":[],
                })
            for i in range(0,len(self.annotation[video_name])):
                self.videos[-1]["img_names"].append(self.annotation[video_name][i]["file_name"])
                self.videos[-1]["ids"].append(self.annotation[video_name][i]["ids"])
                self.videos[-1]["pts"].append(self.annotation[video_name][i]["pts"])
                self.videos[-1]["bboxes"].append(self.annotation[video_name][i]["bboxes"])
                self.videos[-1]["cnt"].append(self.annotation[video_name][i]["cnt"])
    def __len__(self) -> int:
        return len(self.videos)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video=self.videos[index]
        video_name=video["video_name"]
        img_names=video["img_names"]
        height=video["height"]
        width=video["width"]
        ids=video["ids"]
        pts=video["pts"]
        bboxes=video["bboxes"]
        cnt=video["cnt"]
        imgs=[]
        for img_name in img_names:
            img_path=os.path.join(self.root,video_name,img_name)
            img=self.transforms(image=np.array(Image.open(img_path).convert("RGB")))["image"]
            imgs.append(img)
        imgs=torch.stack(imgs,dim=0)
        labels={
            "h":height,
            "w":width,
            "pts":pts,
            "bboxes":bboxes,
            "cnt":cnt,
            "video_name":video_name,
            "img_names":img_names,
        }
        return imgs,labels
        

class PairDataset(VisionDataset):
    def __init__(self, root: str, annotation_dir:str, max_len:int ,transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, train=True,step=20,interval=1,force_last=False) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.video_paths=os.listdir(root)
        self.video_paths=[os.path.join(root,video_path) for video_path in self.video_paths]
        self.annotations_paths=os.listdir(annotation_dir)
        self.annotations_paths=[os.path.join(annotation_dir,annotation) for annotation in self.annotations_paths]
        self.annotation={}
        self.pairs=[]
        self.max_len=max_len
        for annotation_path in self.annotations_paths:
            with open(annotation_path) as f:
                video_name=annotation_path.split('/')[-1].split('.')[0]

                self.annotation[video_name]=[]
                lines=f.readlines()
                for line in lines:
                    line=line.split()
                    file_name=line[0]
                    width,height=int(line[1]),int(line[2])
                    
                    data=[float(x) for x in line[3:] if x!=""]
                    cnt=len(data)//7

                    ids=-1*np.ones((max_len,1))
                    pts=-1*np.ones((max_len,2))
                    bboxes=-1*np.ones((max_len,4))
                    if len(data)>0:
                        ids=-1*np.ones((cnt,1))
                        pts=-1*np.ones((cnt,2))
                        bboxes=-1*np.ones((cnt,4))
                        data=np.array(data)
                        data=np.reshape(data,(-1,7))
                        ids=data[:,6].reshape(-1,1)
                        pts=data[:,4:6]
                        bboxes[:]=data[:,0:4]
                        bboxes[:,2]=bboxes[:,2]-bboxes[:,0]
                        bboxes[:,3]=bboxes[:,3]-bboxes[:,1]
                        pts[:,0]=pts[:,0]/width
                        pts[:,1]=pts[:,1]/height
                        bboxes[:,0]=bboxes[:,0]/width
                        bboxes[:,1]=bboxes[:,1]/height
                        bboxes[:,2]=bboxes[:,2]/width
                        bboxes[:,3]=bboxes[:,3]/height
                    self.annotation[video_name].append({"file_name":file_name,"height":height,"width":width,"ids":ids,"pts":pts,"bboxes":bboxes,"cnt":cnt})
        for video_path in self.video_paths:
            video_name=video_path.split('/')[-1]
            last_step=0
            for i in range(1,len(self.annotation[video_name])-step,interval):
                self.pairs.append({
                    "0":self.annotation[video_name][i],
                    "1":self.annotation[video_name][i+step],
                    "video_name":video_name,
                })
                last_step=i+step
            if force_last and last_step<len(self.annotation[video_name])-1:
                
                self.pairs.append({
                    "0":self.annotation[video_name][last_step],
                    "1":self.annotation[video_name][-1],
                    "video_name":video_name,
                })
        self.train=train
    def __len__(self) -> int:
        return len(self.pairs)  
    def add_noise(self,pts):
        noise=np.random.normal(scale=0.001,size=pts.shape)
        pts=pts+noise
        pts[pts>1]=1
        pts[pts<0]=0
        return pts
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pair=self.pairs[index]
        img0_path=os.path.join(self.root,pair["video_name"],pair["0"]["file_name"])
        img1_path=os.path.join(self.root,pair["video_name"],pair["1"]["file_name"])
        video_name=pair["video_name"]
        img_name1=pair["0"]["file_name"]
        img_name2=pair["1"]["file_name"]
        cnt_0=pair["0"]["cnt"]
        cnt_1=pair["1"]["cnt"]
        pt_0=pair["0"]["pts"]
        pt_1=pair["1"]["pts"]


        if self.train:
            pt_0=self.add_noise(pt_0)
            pt_1=self.add_noise(pt_1)
        bbox_0=pair["0"]["bboxes"]
        bbox_1=pair["1"]["bboxes"]
        id_0=pair["0"]["ids"]
        id_1=pair["1"]["ids"]
        if self.train and (pair["0"]["height"]!=pair["1"]["height"] or pair["0"]["width"]!=pair["1"]["width"] or cnt_0==0 or cnt_1==0):
            print("error")
            print(pair["0"]["height"],pair["1"]["height"],pair["0"]["width"],pair["1"]["width"],cnt_0,cnt_1)
            return self.__getitem__((index+1)%len(self))
        img0=self.transforms(image=np.array(Image.open(img0_path).convert("RGB")))["image"]
        img1=self.transforms(image=np.array(Image.open(img1_path).convert("RGB")))["image"]
        fused_pts_list0=[]
        fused_pts_list1=[]
        id_list0=[]
        id_list1=[]
        fused_num=0
        fused_num1=0
        id0_pt0_dict={id[0]:pt for id,pt in zip(id_0,pt_0)}
        id1_pt1_dict={id[0]:pt for id,pt in zip(id_1,pt_1)}
        independ0_list=[]
        independ1_list=[]

        for pt,id in zip(pt_0,id_0):
            if id in id_1:
                fused_pts_list0.append(pt)
                id_list0.append(id)
                fused_num+=1
                pt1=id1_pt1_dict[id[0]]
                fused_pts_list1.append(pt1)
                id_list1.append(id)
            else:
                independ0_list.append(pt)
        for pt,id in zip(pt_1,id_1):
            if id not in id_0:
                independ1_list.append(pt)
        if len(independ0_list)==0:
            independ0_list.append((0.25,0.25))
        if len(independ1_list)==0:
            independ1_list.append((0.75,0.75))
        if self.train and (fused_num<2 or fused_num>30 or len(independ0_list)==0 or len(independ1_list)==0):
            return self.__getitem__((index+1)%len(self))
        independ_pts0=-1*np.ones((self.max_len,2))
        independ_pts0[:len(independ0_list)]=np.array(independ0_list)
        independ_pts1=-1*np.ones((self.max_len,2))
        independ_pts1[:len(independ1_list)]=np.array(independ1_list)
        x=torch.cat([img0,img1],dim=0)
        fused_pts0=-1*np.ones((self.max_len,2))
        fused_pts1=-1*np.ones((self.max_len,2))

        if fused_num>0:
            fused_pts0[:fused_num]=np.array(fused_pts_list0)
            fused_pts1[:fused_num]=np.array(fused_pts_list1)

        fused_pts1=torch.from_numpy(fused_pts1).float()
        fused_pts0=torch.from_numpy(fused_pts0).float()
        # cv2_img0=inverse_normalize(img0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        # cv2_img1=inverse_normalize(img1).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        # img_pair=np.concatenate([cv2_img0,cv2_img1],axis=1)
        # img_pair=cv2.cvtColor(img_pair,cv2.COLOR_RGB2BGR)
        # for pt0,pt1 in zip(fused_pts0,fused_pts1):
        #     cv2.circle(img_pair,(int(pt0[0]*1280),int(pt0[1]*720)),5,(0,0,255),-1)
        #     cv2.circle(img_pair,(int(pt1[0]*1280)+1280,int(pt1[1]*720)),5,(0,0,255),-1)
        #     cv2.line(img_pair,(int(pt0[0]*1280),int(pt0[1]*720)),(int(pt1[0]*1280)+1280,int(pt1[1]*720)),(0,0,255),2)
        # cv2.imwrite(f"outputs/vision/{video_name}_{img_name1}_{img_name2}.jpg",img_pair)
        ref_pts=torch.stack([fused_pts0,fused_pts1],dim=0)
        labels={
            "h":pair["0"]["height"],
            "w":pair["0"]["width"],
            "gt_default_pts":pt_0,
            "gt_duplicate_pts":pt_1,
            "gt_fuse_pts0":fused_pts0,
            "gt_fuse_pts1":fused_pts1,

            "gt_default_num":pair["0"]["cnt"],
            "gt_duplicate_num":pair["1"]["cnt"],
            "gt_fuse_num":fused_num,
            "video_name":video_name,
            "img_name1":img_name1,
            "img_name2":img_name2,
        }

        inputs={
            "image_pair":x,
            "ref_pts":ref_pts,
            "ref_num":fused_num,
            "independ_pts0":torch.from_numpy(independ_pts0).float(),
            "independ_pts1":torch.from_numpy(independ_pts1).float(),
            "independ_num0":len(independ0_list),
            "independ_num1":len(independ1_list),
        }
        return inputs,labels

def build_dataset(root,annotation_dir,max_len,train=False,step=20,interval=1,force_last=False):
    transforms=transform()
    dataset=PairDataset(root,annotation_dir,max_len,transforms=transforms,train=train,step=step,interval=interval,force_last=force_last)
    return dataset

def build_video_dataset(root,annotation_dir):
    dataset=VideoDataset(root,annotation_dir,transforms=video_transform())
    return dataset
