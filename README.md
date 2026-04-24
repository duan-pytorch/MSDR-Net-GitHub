# MSDR-Net-GitHub
# MSDR-Net: Multi-Scale Dilated Residual Network with Efficient Channel Attention

Official PyTorch implementation of the paper:  
**"Multi-Scale Dilated Residual Network with Efficient Channel Attention for Automated Benign-Malignant Classification of Spinal Tumors on X-ray Images"**

## Overview

This repository provides the core architecture implementation of MSDR-Net, a deep learning model designed for automated benign-malignant classification of spinal tumors on X-ray images. The network features:

- **Multi-Branch Dilated Residual Blocks (MBRB)** with parallel convolutional paths (dilation rates D=1, 2, 3)
- **Efficient Channel Attention (ECA)** modules for adaptive feature recalibration
- A four-level encoder structure with an MLP classifier

## Model Architecture

The model strictly follows the architectural specifications described in Section 2.4 of the manuscript:

| Stage | Operation | Output Channels | Spatial Dimensions |
|-------|-----------|-----------------|-------------------|
| 1 | MBRB + ECA | 64 | 224×224 |
| 2 | MBRB + ECA | 128 | 112×112 |
| 3 | MBRB + ECA | 256 | 56×56 |
| 4 | MBRB + ECA | 512 | 28×28 |

## Repository Structure
MSDR-Net/
├── models/
│   ├── init.py
│   └── msdr_net.py          # Complete MSDR-Net architecture (~23.6M params)
├── train.py                 # Training script skeleton
├── eval.py                  # Evaluation script skeleton
├── dataset.py               # Dataset interface skeleton
├── utils/
│   └── metrics.py           # Evaluation metrics implementation
├── requirements.txt
└── README.md

## Usage

### Environment Setup
```bash
pip install -r requirements.txt

Model Instantiation
from models.msdr_net import MSDRNet
import torch

model = MSDRNet(num_classes=2)
x = torch.randn(1, 3, 224, 224)
logits = model(x)
print(f"Model parameters: {model.get_param_count():.2f}M")

Availability Notice
Due to institutional data governance policies and patient privacy protection requirements under the Regulations on Ethical Review of Biomedical Research Involving Humans, the following components are not publicly distributed:
Complete training scripts with hospital-specific data loading pipelines
Pre-trained model weights
Patient-level data preprocessing parameters (ROI coordinates, CLAHE statistics, Z-score normalization values derived from clinical cohorts)
Evaluation scripts containing internal PACS data paths and institutional configurations
These materials are available upon reasonable request to the corresponding authors:
Ningkui Niu: niuningkui6743242@163.com
Researchers are welcome to use the provided model architecture code to replicate the network on their own datasets.

Acknowledgments
This project is financially supported by the Yinchuan Science and Technology Plan Project (2025SF31) and the Ningxia Hui Autonomous Region Health Research Key Project (2025-NWZD-A001).


---

### **2. `requirements.txt`**

```text
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pillow>=10.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.66.0
tensorboard>=2.14.0
