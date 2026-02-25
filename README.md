# SynReEM: Synapse Reconstruction via Instance Structure Encoding in Anisotropic vEM Images

We develop SynReEM, an end-to-end synapse reconstruction framework for anisotropic volume electron microscopy scenarios.

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Getting Started](#getting-started)
  + [AEMC Data Preprocessing](#aemc-data-preprocessing)
  + [Segmentation process](#segmentation-process)
  + [Graph aggregation reconstruction](#graph-aggregation-reconstruction)
* [Acknowledgments](#acknowledgments)

## Overview

### The basic idea of SynReEM

<div  align="center">    
	<img src="https://github.com/fenglingbai/SynReEM/blob/main/fig/p2_motivation.png" width = "500px" />
</div>

### The basic architecture of SynReEM

<div  align="center">    
	<img src="https://github.com/fenglingbai/SynReEM/blob/main/fig/p3_SynReEM.png" width = "500px" />
</div>

### Synapse reconstruction results display


<video width="800" controls autoplay loop muted>
  <source src="https://github.com/fenglingbai/SynReEM/raw/refs/heads/main/video/Kasthuri.mp4" type="video/mp4">
  你的浏览器不支持视频播放，请下载视频查看：https://raw.githubusercontent.com/你的用户名/仓库名/分支名/视频文件路径.mp4
</video>

<!-- <div  align="center">    
	<img src="https://github.com/fenglingbai/SynReEM/blob/main/video/p3_SynReEM.png" width = "500px" />
</div> -->

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fenglingbai/SynReEM.git
cd ~/SynReEM
```

2. Create and activate the conda environment:
```bash
conda create --name SynReEM --file environment.txt -y
conda activate SynReEM
```

3. Verify PyTorch installation:
```bash
python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'
```

4. Install the nnUNet environment (integrated with this framework):
```bash
pip install -e .
```

## Getting Started

### AEMC Data Preprocessing

SynReEM first requires converting instance labels to AEMC labels to facilitate model learning.

Refer to the demo script for implementation details:
```
SynReEM\scripts\data_encode_demo.py
```

### Segmentation process

#### Data Preparation

Convert original tif data to nii.gz format compatible with the nnUNet framework:
```bash
python SynReEM\nnunet\dataset_conversion\Task603_synapse178synins.py
```

Set up the experimental plan:
```bash
python SynReEM\nnunet\experiment_planning\nUNet_plan_and_preprocess.py -t XXX --verify_dataset_integrity
```

#### Training

```bash
python SynReEM\nnunet\run\run_training_synreem.py 3d_fullres SynReEMTrainer TaskXXX_MYTASK FOLD --npz
```

**Example:**
```bash
python SynReEM\nnunet\run\run_training_synreem.py 3d_fullres SynReEMTrainer Task603_synapse178synins 4 --npz
```

#### Inference

```bash
python SynReEM\nnunet\inference\predict_synreem.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
```

### Graph aggregation reconstruction

Convert AEMC labels back to instance labels for final reconstruction results.

Refer to the demo script for implementation details:
```
SynReEM\scripts\data_decode_demo.py
```

## Acknowledgments

This project builds upon the following open-source frameworks:

- nnUNet (https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)
- EmbedSeg (https://github.com/juglab/EmbedSeg)
