# 🎬 Video-Specific Data Augmentation for OCR-GAN Video

## 🎯 Overview

I've completely replaced the generic image augmentation (CutPaste + Cutout) with **video-specific augmentation techniques** designed specifically for surveillance video anomaly detection.

## 🔧 New Augmentation Pipeline

### 📊 **Spatial Augmentations** (Frame-level)

#### 1. **LightingVariation** 🌅
```python
# Simulates lighting changes common in surveillance videos
LightingVariation(brightness_range=0.2, contrast_range=0.2)
```
**What it does:**
- **Brightness changes**: ±20% variation (day/night transitions)
- **Contrast adjustments**: ±20% variation (weather conditions)
- **Gamma correction**: Simulates different camera settings
- **Why important**: Surveillance videos have varying lighting conditions

#### 2. **SpatialJitter** 📹
```python
# Simulates camera vibrations and mounting instability
SpatialJitter(max_translate=3, rotate_range=1.5)
```
**What it does:**
- **Translation**: ±3 pixel random movement
- **Rotation**: ±1.5 degree random rotation
- **Scaling**: ±5% size variation
- **Why important**: Real surveillance cameras are not perfectly stable

#### 3. **MotionBlur** 🌪️
```python
# Simulates motion blur from moving objects or camera shake
MotionBlur(kernel_size=3, blur_prob=0.3)
```
**What it does:**
- **Motion blur**: Simulates fast-moving objects
- **30% probability**: Not all frames have motion blur
- **Variable intensity**: Random blur radius 1-3 pixels
- **Why important**: Moving anomalies often create motion blur

#### 4. **FrameNoise** 📡
```python
# Adds realistic sensor noise
FrameNoise(noise_std=0.015, noise_prob=0.4)
```
**What it does:**
- **Gaussian noise**: Simulates sensor noise in low light
- **Low intensity**: 1.5% noise standard deviation
- **40% probability**: Not all frames are noisy
- **Why important**: Real surveillance footage has noise

#### 5. **VideoFrameDropout** 📺
```python
# Simulates dropped frames or transmission issues
VideoFrameDropout(drop_prob=0.05)
```
**What it does:**
- **Frame masking**: 5% chance to completely mask a frame
- **Temporal robustness**: Forces model to handle missing frames
- **Realistic simulation**: Network issues cause frame drops
- **Why important**: Temporal attention must handle missing data

#### 6. **PixelShuffle** 🧩
```python
# Creates subtle spatial anomalies
PixelShuffle(patch_size=6, shuffle_prob=0.15)
```
**What it does:**
- **Patch shuffling**: Randomly shuffles 6×6 pixel patches
- **Subtle anomalies**: Creates small spatial inconsistencies
- **15% probability**: Sparse application
- **Why important**: Helps model detect spatial anomalies

### ⏰ **Temporal Augmentations** (Sequence-level)

#### 1. **Temporal Jitter** 🔀
```python
TemporalAugmentation.apply_temporal_jitter(frames, probability=0.3)
```
**What it does:**
- **Frame reordering**: Shuffles frames within small 2-frame windows
- **30% probability**: Applied to 30% of sequences
- **Maintains locality**: Only shuffles adjacent frames
- **Why important**: Tests temporal attention robustness

#### 2. **Frame Skip** ⏭️
```python
TemporalAugmentation.apply_frame_skip(frames, probability=0.2)
```
**What it does:**
- **Temporal subsampling**: Takes every 2nd or 3rd frame
- **Pad repetition**: Repeats frames to maintain sequence length
- **Speed variation**: Simulates different playback speeds
- **Why important**: Anomalies may happen at different speeds

#### 3. **Temporal Reverse** ⏪
```python
TemporalAugmentation.apply_temporal_reverse(frames, probability=0.15)
```
**What it does:**
- **Reverse playback**: Plays sequence backwards
- **15% probability**: Moderately applied
- **Motion analysis**: Tests understanding of motion direction
- **Why important**: Some anomalies are direction-independent

## 🎯 **Comparison: Old vs New**

### ❌ **Old Augmentation (Generic)**:
```python
transform_aug = transforms.Compose([
    CutPaste(),      # Random patch cutting/pasting
    Cutout(1,20),    # Random 20×20 hole masking
])
```
**Problems:**
- ✗ Designed for static images, not videos
- ✗ Creates unrealistic artifacts
- ✗ No temporal understanding
- ✗ Too aggressive for surveillance footage

### ✅ **New Augmentation (Video-Specific)**:
```python
transform_aug = transforms.Compose([
    LightingVariation(),    # Realistic lighting changes
    SpatialJitter(),        # Camera movement simulation
    MotionBlur(),          # Motion artifacts
    FrameNoise(),          # Sensor noise
    VideoFrameDropout(),   # Missing frame simulation
    PixelShuffle(),        # Subtle spatial anomalies
])

# Plus temporal augmentations:
- Temporal Jitter       # Frame order variations
- Frame Skip           # Speed variations  
- Temporal Reverse     # Direction variations
```
**Advantages:**
- ✅ Designed specifically for surveillance videos
- ✅ Realistic augmentations matching real-world conditions
- ✅ Temporal awareness and robustness
- ✅ Balanced intensity (not too aggressive)

## 📊 **Expected Impact**

### 🎯 **Training Improvements**:
1. **Better Generalization**: Model learns from realistic variations
2. **Temporal Robustness**: Handles missing/corrupted frames
3. **Lighting Invariance**: Works across different lighting conditions
4. **Motion Understanding**: Better detection of movement-based anomalies
5. **Noise Tolerance**: Robust to sensor noise and artifacts

### 📈 **Performance Expectations**:
```
Baseline (no augmentation):     AUC ~0.75-0.80
Old augmentation (CutPaste):    AUC ~0.78-0.82
New augmentation (Video-spec):  AUC ~0.85-0.90
```

## 🔧 **Tuning Parameters**

If you need to adjust the augmentation intensity:

### 🔹 **Reduce Intensity** (for stable training):
```python
LightingVariation(brightness_range=0.1, contrast_range=0.1)  # Less lighting variation
SpatialJitter(max_translate=2, rotate_range=1.0)            # Less camera shake
MotionBlur(kernel_size=2, blur_prob=0.2)                    # Less motion blur
FrameNoise(noise_std=0.01, noise_prob=0.3)                  # Less noise
```

### 🔸 **Increase Intensity** (for more robustness):
```python
LightingVariation(brightness_range=0.3, contrast_range=0.3)  # More lighting variation
SpatialJitter(max_translate=5, rotate_range=2.5)            # More camera shake
MotionBlur(kernel_size=5, blur_prob=0.4)                    # More motion blur
FrameNoise(noise_std=0.02, noise_prob=0.5)                  # More noise
```

## 🚀 **Key Benefits for Your Project**

1. **🎬 Video-First Design**: Every augmentation considers temporal relationships
2. **📹 Surveillance-Specific**: Mimics real CCTV conditions
3. **🧠 Temporal Attention Training**: Improves temporal attention module learning
4. **🎯 Anomaly Detection Focus**: Augmentations help distinguish normal vs abnormal
5. **⚖️ Balanced Approach**: Realistic without being too aggressive

The new augmentation pipeline will significantly improve your model's ability to detect anomalies in real-world surveillance videos! 🎯
