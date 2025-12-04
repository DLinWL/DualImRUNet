# Overview

This repository contains the official PyTorch implementation for the paper:  
**"Leveraging Bi-Directional Channel Reciprocity for Robust Ultra-Low-Rate Implicit CSI Feedback with Deep Learning"**  
Accepted by **IEEE GLOBECOM 2025**.

DualImRUNet is an efficient uplink-assisted deep implicit CSI feedback framework incorporating two novel preprocessing techniques to achieve ultra-low feedback rates while maintaining high environmental robustness.


## 🚀 Installation

### Environment Setup

Create a conda environment with all required dependencies:

```bash
conda env create -f environment.yaml
conda activate DualImRUNet
```

### Requirements

- Python >= 3.8
- PyTorch == 2.1.0 (with CUDA 12.1 support)
- numpy >= 1.24.3
- scipy >= 1.10.1
- tensorboardX >= 2.2
- einops >= 0.7.0
- thop >= 0.1.1

**Note**: MKL threading is set to `GNU` in `main.py` to avoid libgomp conflicts on Linux systems.

## 📦 Dataset Preparation

### Dataset Structure

The expected dataset structure is:

```
../DualImRUNet_dataset/
├── trainvalset_env{env_num}_eigenvector_ad_enhanced.npy
├── trainvalset_env{env_num}_eigenvector_ad_enhanced_uplink.npy
├── testset_env30_eigenvector_ad_enhanced.npy
└── testset_env30_eigenvector_ad_enhanced_uplink.npy
```

### Dataset Configuration

The dataset files are determined by the following flags:
- `--env_num`: Number of environments for training (1, 2, 4, 8, 16, 32, 70, 100)
- `--ad_flag`: Angular-delay domain flag (0 or 1)
- `--enhanced_eigenvector_flag`: Enhanced eigenvector design flag (0 or 1)
- `--eig_flag`: Eigenvector flag (currently only supports 1)


## 📁 Project Structure

```
DualImRUNet/
├── main.py                      # Main training entrypoint
├── environment.yaml             # Conda environment specification
├── README.md                    # This file
├── models/
│   ├── __init__.py
│   └── DualImRUNet.py          # Core model definition (Transformer encoder-decoder)
├── utils/
│   ├── __init__.py
│   ├── data_process.py         
│   ├── solver_up.py            
│   ├── parser.py               
│   ├── scheduler.py           
│   ├── logger.py               
│   ├── statics.py             
│   ├── SparAlign.py            
│   └── init.py                 
├── checkpoints/                 # Model checkpoints (created at runtime)
│   ├── best_rho.pth
│   ├── best_nmse.pth
│   └── last.pth
└── data_vision/                 # TensorBoard logs (created at runtime)
    └── {run_tag}/
        ├── train/
        ├── test/
        ├── best/
        └── every/
```

## ✅ TODO
- Upload quantization-related files.
- Upload dataset files for the expected dataset root.

## 🎯 Usage

### Training

Basic training with recommended settings:

```bash
python main.py --gpu 0 --batch-size 200 --scheduler const --epochs 1000 \
  --cr 208 --ad_flag 1 --spalign_flag 1 --enhanced_eigenvector_flag 1 --env_num 70
```



### Monitoring Training Progress

View training logs with TensorBoard:

```bash
tensorboard --logdir=data_vision/{run_tag}
```

The run tag is automatically generated based on your configuration, e.g., `dual_envout70_eig1_enh1_ad1_sp1_cr208_d64_lrconst`.

## ⚙️ Configuration

### Key Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gpu` | int | 0 | GPU ID to use |
| `--batch-size` | int | 200 | Batch size for training |
| `--epochs` | int | 60 | Number of training epochs |
| `--cr` | int | 32 | Compression ratio (reciprocal, e.g., 208 means 1/208) |
| `--env_num` | int | 70 | Number of environments for training set |
| `--ad_flag` | int | 1 | Angular-delay domain flag |
| `--spalign_flag` | int | 0 | Sparsity alignment flag |
| `--enhanced_eigenvector_flag` | int | 0 | Enhanced bi-direction correlation design flag |

### Compression Ratios

Common compression ratios (reciprocal values):
- CR = 32 → 1/32 compression (26 dimensions)
- CR = 64 → 1/64 compression (13 dimensions)
- CR = 208 → 1/208 compression (4 dimensions, ultra-low rate)


## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{liu2025leveraging,
  title={Leveraging Bi-Directional Channel Reciprocity for Robust Ultra-Low-Rate Implicit CSI Feedback with Deep Learning},
  author={Liu, Zhenyu and Ma, Yi and Tafazolli, Rahim and Ding, Zhi},
  booktitle={IEEE Global Communications Conference (GLOBECOM)},
  year={2025},
}
```

## 🙏 Acknowledgement

We would like to express our sincere gratitude to the following open-source project:

- **[TransNet](https://github.com/Treedy2020/TransNet)**: This work builds upon the TransNet architecture, which provided the Single TransNet Encoder Layers and Single TransNet Decoder Layers. The multi-head attention mechanism and transformer blocks in our implementation are adapted from their excellent open-source code.

