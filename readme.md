# Weakly Supervised Video Crowd Counting


This is an officical implementation of the paper "Weakly Supervised Video Crowd Counting" in PyTorch. 

[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Weakly_Supervised_Video_Individual_Counting_CVPR_2024_paper.html)

## Introduction

Video Individual Counting (VIC) aims to predict the number of unique individuals in a single video. Existing methods learn representations based on trajectory labels for individuals, which are annotation-expensive. To provide a more realistic reflection of the underlying practical challenge, we introduce a weakly supervised VIC task, wherein trajectory labels are not provided. Instead, two types of labels are provided to indicate traffic entering the field of view (inflow) and leaving the field view (outflow). We also propose the first solution as a baseline that formulates the task as a weakly supervised contrastive learning problem under group-level matching. In doing so, we devise an end-to-end trainable soft contrastive loss to drive the network to distinguish inflow, outflow, and the remaining. To facilitate future study in this direction, we generate annotations from the existing VIC datasets SenseCrowd and CroHD and also build a new dataset, UAVVIC. Extensive results show that our baseline weakly supervised method outperforms supervised methods, and thus, little information is lost in the transition to the more practically relevant weakly supervised task.

## Pipeline

![pipeline](./statics/imgs/pipeline.png)

The inference pipeline of our CGNet and the weakly supervised representation learning method (WSRL). The pipeline comprises a frame-level crowd locator, an encoder, and a Memory-based individual count predictor (MCP). The locator predicts the coordinates for pedestrians. The encoder generates representations for each individual, and MCP predicts inflow counts and updates the individual templates stored in the memory. To pull the matched groups (X and Y) closer and push away individual pairs from unmatched groups, WSRL exploits inflow and outflow labels to optimize the encoder with a novel Group level Matching Loss (GML), which consists of a soft contrastive loss and a hinge loss.

## Preparation

Clone this repo into your local machine. 

``` bash
git clone https://github.com/streamer-AP/CGNet
cd CGNet
```

Create a new virtual environment and install the required packages. 

``` bash
conda create -n CGNet python=3.7
conda activate CGNet
pip install -r requirements.txt
```

Data Preparation
- **CroHD** : Download CroHD dataset from this [link](https://motchallenge.net/data/Head_Tracking_21/). Unzip ```HT21.zip``` and place ``` HT21``` into the folder (```Root/dataset/```). 
- **SenseCrowd** dataset: Download the dataset from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm) or from the original dataset [link](https://github.com/HopLee6/VSCrowd-Dataset). 

### Usage

1. We provide a toy example for the GML loss to quickly understand GML loss, which can also be used in other tasks. 

``` bash
cd models/
python tri_sim_ot_b.py
```
You can will the similarity matrix converging process like this:

<video src="./statics/imgs/sim.mp4" width="800" height="600" controls="controls"></video>


2. Inference.
   * Before inference, you need to get crowd localization result on a pre-trained crowd localization model. You can use [FIDTM](https://github.com/dk-liang/FIDTM.git), [STEERER](https://github.com/taohan10200/STEERER.git) or any other crowd localization model that output coordinates results.
   * We also provide a crowd localization results inferenced by FIDTM-HRNet-W48. You can download it from [Baidu disk](https://pan.baidu.com/s/1i9BXHab5pVYhZFCESD6F7Q?pwd=08zg). The data format follows:
   ```
   x y
   x y
   ```
   * Pretrained models can be downloaded from [Baidu disk](https://pan.baidu.com/s/1GZJM6sHlFULK56UTTlIhtg?pwd=pigo). Unzip it in the weight folder and run the following command.
   ``` bash
    python inference.py
   ```
   It will cost less than 2GB GPU memory. And using interval = 15(3s), the inference time will be less than 10 minutes for all datasets.
   * For the SenseCrowd dataset, the repoduced results of this repo is slightly better than the paper. The results are as follows:
   ```
   | Method | MAE | MSE | WRAE| 
    | ------ | --- | --- | --- |
    | Paper  | 8.86 | 17.69| 12.6|
    | Repo   | 8.84 | 19.00| 9.7|
    ```
3. Training.
   * For training, you need to prepare the dataset and the crowd localization results. The data format follows:
   ```
   x y
   x y
   ```
   * The training script is as follows:
   ``` bash
   python train.py
   ```
   It also supports multi-GPU training. You can set the number of GPUs by using the following command:
   ``` bash
   bash dist_train.sh 8
   ```

## Citation

If you find this work useful, please consider citing it.

``` bash
@inproceedings{liu2024weakly,
  title={Weakly Supervised Video Individual Counting},
  author={Liu, Xinyan and Li, Guorong and Qi, Yuankai and Yan, Ziheng and Han, Zhenjun and van den Hengel, Anton and Yang, Ming-Hsuan and Huang, Qingming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19228--19237},
  year={2024}
}
```

## Acknowledgement
We thank the authors of [FIDTM](https://github.com/dk-liang/FIDTM.git) and [DR.VIC](https://github.com/taohan10200/DRNet.git) for their excellent work.

