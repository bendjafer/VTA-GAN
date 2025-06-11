#!/usr/bin/env python3
"""
Test script to verify video augmentation is working properly during training
"""

import torch
import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data.dataloader import load_video_data_FD_aug
from options import Options

def test_video_augmentation_effectiveness():
    """Test if video augmentation is working and producing expected variations"""
    print("🔍 Testing Video Augmentation Effectiveness")
    print("=" * 50)
    
    # Create options
    opt = Options().parse()
    opt.dataroot = "data/ucsd2"
    opt.isize = 64
    opt.num_frames = 16
    opt.batchsize = 1
    opt.aspect_method = "maintain_3_2"
    
    if not os.path.exists(opt.dataroot):
        print(f"❌ Dataset not found at: {opt.dataroot}")
        return
    
    # Load data with augmentation
    print(f"📂 Loading data from: {opt.dataroot}")
    classes = ['normal']
    data = load_video_data_FD_aug(opt, classes)
    
    # Get a sample from training data
    train_loader = data.train
    sample_count = 0
    
    print(f"\n🎬 Testing augmentation variations...")
    
    # Collect multiple samples to see variation
    variations = []
    for i, (lap_batch, res_batch, aug_batch, labels) in enumerate(train_loader):
        if i >= 5:  # Test 5 samples
            break
            
        # Extract first video from batch
        lap_video = lap_batch[0]  # (16, 3, 64, 64)
        res_video = res_batch[0]  # (16, 3, 64, 64)
        aug_video = aug_batch[0]  # (16, 3, 64, 64)
        
        # Convert to numpy for analysis
        lap_np = lap_video.cpu().numpy()
        res_np = res_video.cpu().numpy()
        aug_np = aug_video.cpu().numpy()
        
        # Compute statistics
        lap_mean = lap_np.mean()
        lap_std = lap_np.std()
        res_mean = res_np.mean()
        res_std = res_np.std()
        aug_mean = aug_np.mean()
        aug_std = aug_np.std()
        
        # Check for differences (augmentation should create variation)
        diff_mean = abs(aug_mean - (lap_mean + res_mean) / 2)
        diff_std = abs(aug_std - (lap_std + res_std) / 2)
        
        variations.append({
            'sample': i,
            'lap_mean': lap_mean,
            'lap_std': lap_std,
            'res_mean': res_mean,
            'res_std': res_std,
            'aug_mean': aug_mean,
            'aug_std': aug_std,
            'diff_mean': diff_mean,
            'diff_std': diff_std
        })
        
        print(f"   Sample {i+1}: Aug Mean={aug_mean:.3f}, Aug Std={aug_std:.3f}, "
              f"Diff={diff_mean:.3f}")
    
    # Analyze variations
    print(f"\n📊 Augmentation Analysis:")
    mean_diffs = [v['diff_mean'] for v in variations]
    std_diffs = [v['diff_std'] for v in variations]
    
    avg_mean_diff = np.mean(mean_diffs)
    avg_std_diff = np.mean(std_diffs)
    
    print(f"   Average mean difference: {avg_mean_diff:.4f}")
    print(f"   Average std difference: {avg_std_diff:.4f}")
    
    # Check if augmentation is working
    if avg_mean_diff > 0.01 or avg_std_diff > 0.01:
        print(f"   ✅ Augmentation is WORKING - creating variation!")
        augmentation_working = True
    else:
        print(f"   ❌ Augmentation may NOT be working - little variation detected")
        augmentation_working = False
    
    return augmentation_working, variations

def test_temporal_augmentation():
    """Test temporal augmentation methods"""
    print(f"\n🎬 Testing Temporal Augmentation Methods")
    print("-" * 40)
    
    from lib.data.video_datasets import TemporalAugmentation
    
    # Test data
    original_frames = [f"frame_{i:03d}.tif" for i in range(16)]
    print(f"   Original frames: {original_frames[:3]}...{original_frames[-3:]}")
    
    # Test temporal jitter
    jittered = TemporalAugmentation.apply_temporal_jitter(original_frames.copy(), probability=1.0)
    jitter_changed = jittered != original_frames
    print(f"   Temporal Jitter: {'✅ Working' if jitter_changed else '❌ Not applied'}")
    if jitter_changed:
        print(f"     Result: {jittered[:3]}...{jittered[-3:]}")
    
    # Test frame skip
    skipped = TemporalAugmentation.apply_frame_skip(original_frames.copy(), probability=1.0)
    skip_changed = skipped != original_frames
    print(f"   Frame Skip: {'✅ Working' if skip_changed else '❌ Not applied'}")
    if skip_changed:
        print(f"     Result: {skipped[:3]}...{skipped[-3:]}")
    
    # Test temporal reverse
    reversed_frames = TemporalAugmentation.apply_temporal_reverse(original_frames.copy(), probability=1.0)
    reverse_changed = reversed_frames != original_frames
    print(f"   Temporal Reverse: {'✅ Working' if reverse_changed else '❌ Not applied'}")
    if reverse_changed:
        print(f"     Result: {reversed_frames[:3]}...{reversed_frames[-3:]}")
    
    return jitter_changed and skip_changed and reverse_changed

def test_spatial_augmentation_transforms():
    """Test individual spatial augmentation transforms"""
    print(f"\n🎨 Testing Spatial Augmentation Transforms")
    print("-" * 40)
    
    from lib.data.dataloader import (VideoFrameDropout, MotionBlur, FrameNoise, 
                                    LightingVariation, SpatialJitter, PixelShuffle)
    
    # Create test image
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))  # Gray image
    test_tensor = torch.ones(3, 64, 64) * 0.5  # Gray tensor
    
    results = {}
    
    # Test VideoFrameDropout
    try:
        dropout = VideoFrameDropout(drop_prob=1.0, mask_value=0.0)  # Force dropout
        dropped = dropout(test_tensor)
        dropout_working = not torch.equal(dropped, test_tensor)
        results['VideoFrameDropout'] = dropout_working
        print(f"   VideoFrameDropout: {'✅ Working' if dropout_working else '❌ Failed'}")
    except Exception as e:
        results['VideoFrameDropout'] = False
        print(f"   VideoFrameDropout: ❌ Error - {e}")
    
    # Test MotionBlur
    try:
        blur = MotionBlur(kernel_size=3, blur_prob=1.0)  # Force blur
        blurred = blur(test_img)
        blur_working = blurred != test_img
        results['MotionBlur'] = blur_working
        print(f"   MotionBlur: {'✅ Working' if blur_working else '❌ Failed'}")
    except Exception as e:
        results['MotionBlur'] = False
        print(f"   MotionBlur: ❌ Error - {e}")
    
    # Test FrameNoise
    try:
        noise = FrameNoise(noise_std=0.1, noise_prob=1.0)  # Force noise
        noisy = noise(test_tensor)
        noise_working = not torch.equal(noisy, test_tensor)
        results['FrameNoise'] = noise_working
        print(f"   FrameNoise: {'✅ Working' if noise_working else '❌ Failed'}")
    except Exception as e:
        results['FrameNoise'] = False
        print(f"   FrameNoise: ❌ Error - {e}")
    
    # Test LightingVariation
    try:
        lighting = LightingVariation(brightness_range=0.2, contrast_range=0.2)
        lit = lighting(test_img)
        lighting_working = lit != test_img
        results['LightingVariation'] = lighting_working
        print(f"   LightingVariation: {'✅ Working' if lighting_working else '❌ Failed'}")
    except Exception as e:
        results['LightingVariation'] = False
        print(f"   LightingVariation: ❌ Error - {e}")
    
    # Test SpatialJitter
    try:
        jitter = SpatialJitter(max_translate=2, rotate_range=1.0)
        jittered = jitter(test_img)
        jitter_working = jittered != test_img
        results['SpatialJitter'] = jitter_working
        print(f"   SpatialJitter: {'✅ Working' if jitter_working else '❌ Failed'}")
    except Exception as e:
        results['SpatialJitter'] = False
        print(f"   SpatialJitter: ❌ Error - {e}")
    
    # Test PixelShuffle
    try:
        shuffle = PixelShuffle(patch_size=4, shuffle_prob=1.0)  # Force shuffle
        shuffled = shuffle(test_tensor)
        shuffle_working = not torch.equal(shuffled, test_tensor)
        results['PixelShuffle'] = shuffle_working
        print(f"   PixelShuffle: {'✅ Working' if shuffle_working else '❌ Failed'}")
    except Exception as e:
        results['PixelShuffle'] = False
        print(f"   PixelShuffle: ❌ Error - {e}")
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n   Summary: {working_count}/{total_count} spatial augmentations working")
    return results

def main():
    """Main testing function"""
    print("🧪 Video Augmentation Effectiveness Test")
    print("=" * 60)
    
    # Test spatial transforms
    spatial_results = test_spatial_augmentation_transforms()
    
    # Test temporal transforms
    temporal_working = test_temporal_augmentation()
    
    # Test augmentation during data loading
    try:
        augmentation_working, variations = test_video_augmentation_effectiveness()
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        augmentation_working = False
        variations = []
    
    # Final summary
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 60)
    
    spatial_working = sum(spatial_results.values())
    total_spatial = len(spatial_results)
    
    print(f"📊 Spatial Augmentation: {spatial_working}/{total_spatial} transforms working")
    print(f"🎬 Temporal Augmentation: {'✅ Working' if temporal_working else '❌ Not working'}")
    print(f"🔄 Data Pipeline: {'✅ Creating variation' if augmentation_working else '❌ Limited variation'}")
    
    if spatial_working >= total_spatial * 0.8 and temporal_working and augmentation_working:
        print(f"\n🎉 VIDEO AUGMENTATION IS WORKING EFFECTIVELY! 🎉")
        print(f"   Your model is receiving varied, augmented video data during training.")
    else:
        print(f"\n⚠️  VIDEO AUGMENTATION NEEDS ATTENTION")
        print(f"   Some augmentation components may not be working as expected.")
    
    return {
        'spatial_results': spatial_results,
        'temporal_working': temporal_working,
        'augmentation_working': augmentation_working,
        'variations': variations
    }

if __name__ == "__main__":
    results = main()
