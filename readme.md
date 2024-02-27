# Weakly Supervised Video Crowd Counting

## Introduction

This is an officical implementation of the paper "Weakly Supervised Video Crowd Counting" in PyTorch. 

Video Individual Counting (VIC) aims to predict the number of unique individuals in a single video. Existing methods learn representations based on trajectory labels for individuals, which are annotation-expensive. To provide a more realistic reflection of the underlying practical challenge, we introduce a weakly supervised VIC task, wherein trajectory labels are not provided. Instead, two types of labels are provided to indicate traffic entering the field of view (inflow) and leaving the field view (outflow). We also propose the first solution as a baseline that formulates the task as a weakly supervised contrastive learning problem under group-level matching. In doing so, we devise an end-to-end trainable soft contrastive loss to drive the network to distinguish inflow, outflow, and the remaining. To facilitate future study in this direction, we generate annotations from the existing VIC datasets SenseCrowd and CroHD and also build a new dataset, UAVVIC. Extensive results show that our baseline weakly supervised method outperforms supervised methods, and thus, little information is lost in the transition to the more practically relevant weakly supervised task.

## Requirements

``` bash
pip install -r requirements.txt
```

## How to use

TODO
