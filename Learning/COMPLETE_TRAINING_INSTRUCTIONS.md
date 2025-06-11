# 🚀 OCR-GAN Video - Complete Training Instructions for UCSD Ped2

## 📋 Quick Start (Recommended Parameters)

### 🎯 **Production Training Command (Recommended)**
```bash
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --num_frames 16 \
    --isize 64 \
    --batchsize 8 \
    --niter 50 \
    --lr 0.0002 \
    --use_temporal_attention \
    --device gpu \
    --gpu_ids 0 \
    --ngpu 1 \
    --name ucsd2_production \
    --w_adv 1 \
    --w_con 50 \
    --w_lat 1 \
    --w_temporal_consistency 0.1 \
    --w_temporal_motion 0.05 \
    --w_temporal_reg 0.01 \
    --print_freq 50 \
    --save_image_freq 100 \
    --verbose \
    --manualseed 42
```

### 🏃 **Quick Test Training (For Testing Setup)**
```bash
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --num_frames 8 \
    --isize 64 \
    --batchsize 4 \
    --niter 2 \
    --lr 0.0002 \
    --use_temporal_attention \
    --device gpu \
    --name ucsd2_test \
    --verbose
```

### 🔥 **High-Performance Training (If you have good GPU)**
```bash
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --num_frames 16 \
    --isize 128 \
    --batchsize 16 \
    --niter 100 \
    --lr 0.0002 \
    --use_temporal_attention \
    --device gpu \
    --gpu_ids 0 \
    --ngpu 1 \
    --name ucsd2_highres \
    --w_adv 1 \
    --w_con 50 \
    --w_lat 1 \
    --w_temporal_consistency 0.1 \
    --w_temporal_motion 0.05 \
    --w_temporal_reg 0.01 \
    --print_freq 25 \
    --save_image_freq 50 \
    --verbose \
    --manualseed 42
```

---

## 📊 Parameter Explanation

### 🎬 **Video-Specific Parameters**
- `--num_frames 16`: Number of frames per video snippet (8, 16, or 32)
- `--isize 64`: Frame resolution (64x64 recommended, 128x128 for high-res)
- `--model ocr_gan_video`: Use the video version of OCR-GAN

### 🔧 **Training Parameters**
- `--batchsize 8`: Batch size (adjust based on GPU memory)
- `--niter 50`: Number of training epochs
- `--lr 0.0002`: Learning rate (standard for GANs)
- `--beta1 0.5`: Adam optimizer momentum

### 🧠 **Temporal Attention Parameters**
- `--use_temporal_attention`: Enable temporal attention modules
- `--w_temporal_consistency 0.1`: Weight for temporal consistency loss
- `--w_temporal_motion 0.05`: Weight for motion preservation loss
- `--w_temporal_reg 0.01`: Weight for attention regularization

### 💪 **Loss Weights**
- `--w_adv 1`: Adversarial loss weight
- `--w_con 50`: Reconstruction loss weight (most important)
- `--w_lat 1`: Latent feature matching loss weight

### 💾 **Hardware & Output**
- `--device gpu`: Use GPU training
- `--gpu_ids 0`: Which GPU to use
- `--ngpu 1`: Number of GPUs
- `--name ucsd2_production`: Experiment name (creates folder structure)

### 📋 **Monitoring**
- `--print_freq 50`: Print losses every 50 iterations
- `--save_image_freq 100`: Save sample images every 100 iterations
- `--verbose`: Print detailed information
- `--manualseed 42`: Fixed random seed for reproducibility

---

## 📁 Output Structure

Training will create the following structure:
```
output/
└── ucsd2_production/        # Your experiment name
    └── ucsd2/               # Dataset name
        ├── train/
        │   ├── weights/     # 🎯 MODEL WEIGHTS SAVED HERE
        │   │   ├── netG_best.pth      # Best generator
        │   │   ├── netD_best.pth      # Best discriminator
        │   │   ├── netG_1.pth         # Epoch 1 weights
        │   │   ├── netD_1.pth
        │   │   ├── netG_2.pth         # Epoch 2 weights
        │   │   └── ...
        │   ├── images/      # Training sample images
        │   └── opt.txt      # Training parameters log
        └── test/
            └── images/      # Test results
```

---

## 🎛️ Memory & Performance Guidelines

### 💾 **GPU Memory Usage (Approximate)**

| Configuration | GPU Memory | Training Time/Epoch |
|---------------|------------|-------------------|
| 8 frames, batch=4, 64x64 | ~4GB | ~15 min |
| 16 frames, batch=8, 64x64 | ~8GB | ~30 min |
| 16 frames, batch=16, 128x128 | ~16GB | ~60 min |

### 🔧 **Memory Optimization**

If you get **CUDA out of memory**:
1. Reduce `--batchsize` (8 → 4 → 2)
2. Reduce `--num_frames` (16 → 8)
3. Reduce `--isize` (128 → 64)

```bash
# Low memory version
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --num_frames 8 \
    --isize 64 \
    --batchsize 2 \
    --niter 50 \
    --use_temporal_attention \
    --device gpu \
    --name ucsd2_lowmem
```

---

## 📊 Training Monitoring

### 🎯 **What to Watch**

1. **Loss Values**:
   - `err_g`: Generator loss (should decrease)
   - `err_d`: Discriminator loss (should stabilize around 0.5)
   - `err_g_temporal`: Temporal consistency loss (should decrease)

2. **AUC Score**: 
   - Shows anomaly detection performance
   - Higher is better (goal: >0.85)
   - Best model is automatically saved

### 📈 **Expected Training Progress**

```
Epoch 1/50: err_g=2.45, err_d=0.83, AUC=0.62
Epoch 5/50: err_g=1.89, err_d=0.67, AUC=0.71
Epoch 10/50: err_g=1.42, err_d=0.58, AUC=0.78
Epoch 25/50: err_g=1.15, err_d=0.52, AUC=0.84
Epoch 50/50: err_g=0.98, err_d=0.49, AUC=0.87
```

---

## 🔄 Resume Training

To continue from a saved checkpoint:
```bash
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --resume output/ucsd2_production/ucsd2/train/weights \
    --iter 25 \
    --niter 50 \
    --use_temporal_attention \
    --device gpu \
    --name ucsd2_production
```

---

## ⚠️ Common Issues & Solutions

### 🐛 **Issue 1: CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size and/or image size
```bash
--batchsize 2 --isize 32
```

### 🐛 **Issue 2: Dataset Not Found**
```
FileNotFoundError: data/ucsd2/train
```
**Solution**: Verify dataset path
```bash
ls data/ucsd2/train/  # Should show snippet folders
```

### 🐛 **Issue 3: Slow Training**
```
Very slow training progress
```
**Solution**: Check GPU usage
```bash
nvidia-smi  # Should show GPU utilization
```

### 🐛 **Issue 4: Loss Not Converging**
```
Losses not decreasing
```
**Solution**: Adjust learning rate
```bash
--lr 0.0001  # Reduce learning rate
```

---

## 🎓 Best Practices

### ✅ **Do's**
1. **Start small**: Test with 2 epochs first
2. **Monitor GPU**: Use `nvidia-smi` to check utilization
3. **Save frequently**: Weights are automatically saved
4. **Use fixed seed**: `--manualseed 42` for reproducibility
5. **Log everything**: Use `--verbose` for detailed output

### ❌ **Don'ts**
1. **Don't use CPU**: Video training is too slow on CPU
2. **Don't use huge batches**: Start with small batch sizes
3. **Don't skip temporal attention**: It's crucial for video
4. **Don't interrupt training**: Let it complete epochs properly

---

## 🚀 Quick Start Checklist

- [ ] ✅ Dataset in `data/ucsd2/` with proper structure
- [ ] ✅ GPU available and CUDA working
- [ ] ✅ All dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ At least 8GB GPU memory available
- [ ] ✅ At least 20GB disk space for outputs
- [ ] ✅ Run test command first (2 epochs)
- [ ] ✅ Then run full training (50+ epochs)

---

## 🎯 Final Command for Server Training

**For your server deployment, use this command:**

```bash
python train_video.py \
    --model ocr_gan_video \
    --dataset ucsd2 \
    --dataroot data/ucsd2 \
    --num_frames 16 \
    --isize 64 \
    --batchsize 8 \
    --niter 100 \
    --lr 0.0002 \
    --use_temporal_attention \
    --device gpu \
    --gpu_ids 0 \
    --ngpu 1 \
    --name ucsd2_server_training \
    --w_adv 1 \
    --w_con 50 \
    --w_lat 1 \
    --w_temporal_consistency 0.1 \
    --w_temporal_motion 0.05 \
    --w_temporal_reg 0.01 \
    --print_freq 25 \
    --save_image_freq 50 \
    --verbose \
    --manualseed 42
```

**This will:**
- ✅ Train for 100 epochs (~6-8 hours on good GPU)
- ✅ Save best weights automatically
- ✅ Use temporal attention for better performance
- ✅ Create organized output structure
- ✅ Provide detailed monitoring
- ✅ Handle UCSD Ped2 dataset properly

**Weights will be saved in:**
`output/ucsd2_server_training/ucsd2/train/weights/netG_best.pth`
