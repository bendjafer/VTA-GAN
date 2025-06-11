# 🎬 OCR-GAN Video Training Execution Instructions

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Execution Commands](#execution-commands)
6. [Monitoring Training](#monitoring-training)
7. [Advanced Configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.7+
- **CUDA**: 10.0+ (for GPU training)
- **RAM**: Minimum 16GB, Recommended 32GB
- **GPU Memory**: Minimum 8GB VRAM for training
- **Storage**: ~10GB for dataset + model checkpoints

### Required Dependencies
The project uses specific package versions for compatibility:

```bash
# Core ML Libraries
torch==1.2.0
torchvision==0.4.0
opencv-python==4.5.3.56
numpy==1.16.4

# Visualization & Monitoring
visdom==0.1.8.8
matplotlib==3.1.0
prettytable==2.2.0

# Image Processing
Pillow>=6.2.0
scikit-learn==0.21.2
```

---

## 🛠 Environment Setup

### Step 1: Install Dependencies

#### Option A: Using requirements.txt (Recommended)
```bash
# Navigate to project directory
cd "D:\OCRGAN VIDEO ADAPTED"

# Install all dependencies
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
# Install PyTorch (CPU version - Windows)
pip install https://download.pytorch.org/whl/cpu/torch-1.2.0%2Bcpu-cp37-cp37m-win_amd64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.4.0%2Bcpu-cp37-cp37m-win_amd64.whl

# For GPU version (if CUDA 10.0 available)
pip install torch==1.2.0 torchvision==0.4.0

# Install other dependencies
pip install opencv-python==4.5.3.56 numpy==1.16.4 visdom==0.1.8.8
pip install matplotlib==3.1.0 prettytable==2.2.0 scikit-learn==0.21.2
```

### Step 2: Verify Installation
```python
# Test script to verify installation
import torch
import cv2
import numpy as np
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

if torch.cuda.is_available():
    print(f"GPU devices: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
```

---

## 📁 Dataset Preparation

### UCSD2 Dataset Structure
The model expects the following directory structure:

```
data/
└── ucsd2/
    ├── train/
    │   ├── Train001_snippet_0000/
    │   │   ├── 001.tif
    │   │   ├── 002.tif
    │   │   ├── ...
    │   │   └── 016.tif
    │   ├── Train001_snippet_0001/
    │   └── ...
    └── test/
        ├── Test001_snippet_0000/
        └── ...
```

### Data Requirements
- **Frame Format**: `.tif` or `.tiff` files
- **Frames per Snippet**: 16 frames (but configurable via `num_frames`)
- **Frame Naming**: Numerical sequence (001.tif, 002.tif, ...)
- **Resolution**: Any resolution (will be resized to `isize`)

### Verify Dataset Structure
```bash
# Check if dataset is properly structured
python -c "
import os
dataset_path = 'data/ucsd2'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

print(f'Train snippets: {len(os.listdir(train_path))}')
print(f'Test snippets: {len(os.listdir(test_path))}')

# Check first snippet
first_snippet = os.path.join(train_path, os.listdir(train_path)[0])
frames = [f for f in os.listdir(first_snippet) if f.endswith('.tif')]
print(f'Frames in first snippet: {len(frames)}')
print(f'Frame files: {sorted(frames)[:5]}...')
"
```

---

## ⚙️ Training Configuration

### Basic Configuration Options

#### Core Model Parameters
```python
# In train_video.py or via command line
opt.model = 'ocr_gan_video'        # Use video version of OCR-GAN
opt.dataset = 'ucsd2'              # Dataset name
opt.num_frames = 8                 # Frames per video snippet (8 or 16)
opt.batchsize = 32                 # Batch size
opt.isize = 32                     # Input image size (32x32)
opt.nc = 3                         # Number of channels (RGB)
opt.nz = 100                       # Latent vector size
opt.ngf = 64                       # Generator features
opt.ndf = 64                       # Discriminator features
```

#### Training Parameters
```python
opt.niter = 15                     # Number of training epochs
opt.lr = 0.0002                    # Learning rate
opt.beta1 = 0.5                    # Adam optimizer beta1
opt.workers = 8                    # Data loading workers
```

#### Loss Weights
```python
opt.w_adv = 1                      # Adversarial loss weight
opt.w_con = 50                     # Reconstruction loss weight
opt.w_lat = 1                      # Latent loss weight

# Temporal loss weights (if using temporal attention)
opt.w_temporal_consistency = 0.1    # Temporal consistency
opt.w_temporal_motion = 0.05       # Motion smoothness
opt.w_temporal_reg = 0.01          # Attention regularization
```

### Model-Specific Options

#### Enable Temporal Attention
```python
opt.use_temporal_attention = True   # Enable temporal attention modules
```

#### GPU Configuration
```python
opt.device = 'gpu'                 # Use 'gpu' or 'cpu'
opt.gpu_ids = '0'                  # GPU IDs (e.g., '0', '0,1', '0,1,2')
opt.ngpu = 1                       # Number of GPUs
```

---

## 🚀 Execution Commands

### Method 1: Basic Training (Recommended)
```bash
# Navigate to project directory
cd "D:\OCRGAN VIDEO ADAPTED"

# Run basic training with default parameters
python train_video.py
```

### Method 2: Training with Custom Parameters
```bash
# Training with specific configuration
python train_video.py \
    --batchsize 16 \
    --num_frames 16 \
    --isize 64 \
    --niter 25 \
    --lr 0.0001 \
    --use_temporal_attention \
    --w_temporal_consistency 0.1 \
    --verbose
```

### Method 3: GPU-Specific Training
```bash
# For single GPU
python train_video.py --gpu_ids 0 --ngpu 1

# For multiple GPUs
python train_video.py --gpu_ids 0,1 --ngpu 2

# For CPU training (if no GPU available)
python train_video.py --device cpu
```

### Method 4: Resume Training from Checkpoint
```bash
# Resume from previous checkpoint
python train_video.py --resume ./output/ocr_gan_video/ucsd2/train/
```

### Complete Training Command Example
```bash
python train_video.py \
    --dataset ucsd2 \
    --model ocr_gan_video \
    --batchsize 32 \
    --num_frames 8 \
    --isize 32 \
    --niter 15 \
    --lr 0.0002 \
    --w_adv 1 \
    --w_con 50 \
    --w_lat 1 \
    --use_temporal_attention \
    --w_temporal_consistency 0.1 \
    --w_temporal_motion 0.05 \
    --w_temporal_reg 0.01 \
    --gpu_ids 0 \
    --workers 8 \
    --verbose \
    --display
```

---

## 📊 Monitoring Training

### Training Output
During training, you'll see output like:
```
Training OCR-GAN Video on ucsd2
>> Training model ocr_gan_video on dataset ucsd2
✅ Temporal attention modules initialized for 8 frames
✅ Temporal loss module initialized
>> Training on device: cuda:0

Epoch [1/15]:
[100/500] Loss_D: 0.423, Loss_G: 2.156, Loss_Con: 1.234, Loss_Lat: 0.087
[200/500] Loss_D: 0.345, Loss_G: 1.987, Loss_Con: 1.123, Loss_Lat: 0.076
...
Training completed. AUC: 0.8234
```

### Key Metrics to Monitor
- **Loss_D**: Discriminator loss (should stabilize around 0.3-0.6)
- **Loss_G**: Generator loss (should decrease gradually)
- **Loss_Con**: Reconstruction loss (should decrease)
- **Loss_Lat**: Latent space loss (should stabilize)
- **AUC**: Final Area Under Curve score (higher is better)

### Visdom Visualization (Optional)
```bash
# Start Visdom server for real-time monitoring
python -m visdom.server

# Then add --display flag to training command
python train_video.py --display --display_port 8097
```

Access visualization at: `http://localhost:8097`

### Output Files
Training creates the following structure:
```
output/
└── ocr_gan_video/
    └── ucsd2/
        ├── train/
        │   ├── netG.pth          # Generator weights
        │   ├── netD.pth          # Discriminator weights
        │   ├── opt.txt           # Training options
        │   └── images/           # Sample images
        └── test/
            └── ...
```

---

## 🔧 Advanced Configuration

### Custom Training Script
Create your own training script with specific parameters:

```python
# custom_train.py
from options import Options
from lib.data.dataloader import load_video_data_FD_aug
from lib.models import load_model

def custom_train():
    # Create options
    opt = Options().parse()
    
    # Custom configuration
    opt.model = 'ocr_gan_video'
    opt.dataset = 'ucsd2'
    opt.num_frames = 16              # Use 16 frames instead of 8
    opt.batchsize = 16               # Smaller batch for larger videos
    opt.isize = 64                   # Higher resolution
    opt.niter = 30                   # More epochs
    opt.use_temporal_attention = True
    opt.w_temporal_consistency = 0.15
    opt.verbose = True
    
    # Load data and model
    data = load_video_data_FD_aug(opt, opt.dataset)
    model = load_model(opt, data, opt.dataset)
    
    # Train
    auc = model.train()
    print(f"Final AUC: {auc:.4f}")

if __name__ == '__main__':
    custom_train()
```

### Memory Optimization
For systems with limited memory:

```python
# Reduce batch size and frames
opt.batchsize = 8
opt.num_frames = 8
opt.workers = 4

# Use gradient accumulation
opt.accumulate_grad = 4
```

### Performance Optimization
For faster training:

```python
# Use mixed precision training (if supported)
opt.use_amp = True

# Increase workers for data loading
opt.workers = 12

# Use larger batch sizes with sufficient memory
opt.batchsize = 64
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
# Solutions:
python train_video.py --batchsize 8 --num_frames 8
python train_video.py --device cpu  # Use CPU if GPU memory insufficient
```

#### 2. Dataset Not Found
```bash
# Error: Cannot find dataset directory
# Solution: Check dataset path
python -c "import os; print(os.path.exists('data/ucsd2/train'))"
```

#### 3. Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Install missing packages
pip install -r requirements.txt
```

#### 4. Version Compatibility
```bash
# Error: PyTorch version mismatch
# Solution: Use specific versions
pip install torch==1.2.0 torchvision==0.4.0
```

#### 5. Slow Training
```bash
# Solutions for slow training:
# 1. Reduce image size
python train_video.py --isize 32

# 2. Reduce frames per snippet
python train_video.py --num_frames 8

# 3. Increase batch size (if memory allows)
python train_video.py --batchsize 64

# 4. Use multiple workers
python train_video.py --workers 12
```

### Performance Benchmarks

#### Expected Training Times (approximate)
- **CPU Training**: ~24 hours for 15 epochs
- **Single GPU (GTX 1080)**: ~4-6 hours for 15 epochs
- **Single GPU (RTX 3080)**: ~2-3 hours for 15 epochs
- **Multiple GPUs**: Proportionally faster

#### Memory Usage
- **8 frames, batch=32**: ~6GB GPU memory
- **16 frames, batch=32**: ~10GB GPU memory
- **8 frames, batch=16**: ~3GB GPU memory

### Validation During Training
The model automatically evaluates on test data after each epoch:
- **AUC Score**: Area Under ROC Curve for anomaly detection
- **Best Model**: Automatically saved when AUC improves
- **Early Stopping**: Training can be stopped if no improvement

---

## ✅ Training Checklist

Before starting training, ensure:

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset properly structured in `data/ucsd2/`
- [ ] GPU drivers installed (if using GPU)
- [ ] Sufficient disk space for checkpoints (~2-5GB)
- [ ] Visdom server running (if using visualization)

### Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify dataset
python -c "import os; print('Dataset OK' if os.path.exists('data/ucsd2/train') else 'Dataset missing')"

# 3. Start training
python train_video.py --verbose

# 4. Monitor training (optional)
# In another terminal:
python -m visdom.server
# Then add --display to training command
```

### Expected Results
- **Training Time**: 2-6 hours depending on hardware
- **Final AUC**: 0.75-0.85 (good performance)
- **Model Size**: ~100-200MB saved checkpoints
- **Convergence**: Usually within 10-15 epochs

The model should achieve competitive performance on UCSD2 dataset with AUC scores above 0.80 indicating successful anomaly detection capability.

---

**🎯 Training Complete!** Your OCR-GAN Video model will be ready for video anomaly detection after successful training.
