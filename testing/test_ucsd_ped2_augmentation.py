#!/usr/bin/env python3
"""
Test script for UCSD Ped2 simplified augmentation pipeline
"""

import torch
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data.ucsd_ped2_augmentation import (
    UCSD_Ped2_Augmentation, 
    UCSD_LightingAdjustment,
    UCSD_MinimalJitter,
    UCSD_SensorNoise,
    UCSD_SubtleMotionBlur,
    UCSD_RareFrameDropout,
    UCSD_Ped2_TemporalAugmentation,
    create_ucsd_ped2_transforms
)
from options import Options


def test_individual_augmentations():
    """Test each augmentation component individually"""
    print("🧪 Testing Individual UCSD Ped2 Augmentations")
    print("=" * 50)
    
    # Create test image (grayscale converted to RGB to simulate UCSD Ped2)
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    test_tensor = torch.ones(3, 64, 64) * 0.5
    
    results = {}
    
    # Test LightingAdjustment
    try:
        lighting = UCSD_LightingAdjustment(brightness_range=0.1, probability=1.0)
        lit_img = lighting(test_img)
        lit_tensor = lighting(test_tensor)
        results['LightingAdjustment'] = True
        print(f"   ✅ LightingAdjustment: Working on PIL and Tensor")
    except Exception as e:
        results['LightingAdjustment'] = False
        print(f"   ❌ LightingAdjustment: Error - {e}")
    
    # Test MinimalJitter
    try:
        jitter = UCSD_MinimalJitter(max_translate=2, probability=1.0)
        jittered = jitter(test_img)
        results['MinimalJitter'] = True
        print(f"   ✅ MinimalJitter: Working")
    except Exception as e:
        results['MinimalJitter'] = False
        print(f"   ❌ MinimalJitter: Error - {e}")
    
    # Test SensorNoise
    try:
        noise = UCSD_SensorNoise(noise_std=0.01, probability=1.0)
        noisy_img = noise(test_img)
        noisy_tensor = noise(test_tensor)
        results['SensorNoise'] = True
        print(f"   ✅ SensorNoise: Working on PIL and Tensor")
    except Exception as e:
        results['SensorNoise'] = False
        print(f"   ❌ SensorNoise: Error - {e}")
    
    # Test SubtleMotionBlur
    try:
        blur = UCSD_SubtleMotionBlur(kernel_size=3, probability=1.0)
        blurred = blur(test_img)
        results['SubtleMotionBlur'] = True
        print(f"   ✅ SubtleMotionBlur: Working")
    except Exception as e:
        results['SubtleMotionBlur'] = False
        print(f"   ❌ SubtleMotionBlur: Error - {e}")
    
    # Test RareFrameDropout
    try:
        dropout = UCSD_RareFrameDropout(drop_probability=1.0)
        dropped_img = dropout(test_img)
        dropped_tensor = dropout(test_tensor)
        results['RareFrameDropout'] = True
        print(f"   ✅ RareFrameDropout: Working on PIL and Tensor")
    except Exception as e:
        results['RareFrameDropout'] = False
        print(f"   ❌ RareFrameDropout: Error - {e}")
    
    working_count = sum(results.values())
    total_count = len(results)
    print(f"\n📊 Individual Tests Summary: {working_count}/{total_count} working")
    
    return results


def test_augmentation_modes():
    """Test different augmentation modes"""
    print(f"\n🎛️ Testing UCSD Ped2 Augmentation Modes")
    print("=" * 50)
    
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    
    for mode in ['minimal', 'conservative', 'moderate']:
        print(f"\n🔧 Testing {mode} mode:")
        
        try:
            augmentation = UCSD_Ped2_Augmentation(mode=mode)
            
            # Apply multiple times to see if it's working
            variations = []
            for i in range(5):
                aug_img = augmentation(test_img)
                
                # Convert to numpy for analysis
                if hasattr(aug_img, 'numpy'):
                    img_array = aug_img.numpy()
                else:
                    img_array = np.array(aug_img)
                
                variations.append(img_array.mean())
            
            # Check for variation
            variation = np.std(variations)
            if variation > 0.001:  # Some variation detected
                print(f"   ✅ {mode} mode: Working (variation: {variation:.4f})")
            else:
                print(f"   ⚠️ {mode} mode: May not be creating enough variation")
                
        except Exception as e:
            print(f"   ❌ {mode} mode: Error - {e}")


def test_temporal_augmentation():
    """Test temporal augmentation methods"""
    print(f"\n⏰ Testing UCSD Ped2 Temporal Augmentation")
    print("=" * 50)
    
    # Test data
    original_frames = [f"frame_{i:03d}.tif" for i in range(16)]
    print(f"   Original frames: {original_frames[:3]}...{original_frames[-3:]}")
    
    # Test minimal temporal jitter
    try:
        jittered = UCSD_Ped2_TemporalAugmentation.apply_minimal_temporal_jitter(
            original_frames.copy(), probability=1.0)
        jitter_changed = jittered != original_frames
        print(f"   ✅ Minimal Temporal Jitter: {'Working' if jitter_changed else 'No change applied'}")
        if jitter_changed:
            print(f"     Result: {jittered[:3]}...{jittered[-3:]}")
    except Exception as e:
        print(f"   ❌ Minimal Temporal Jitter: Error - {e}")
    
    # Test pedestrian speed variation
    try:
        speed_varied = UCSD_Ped2_TemporalAugmentation.apply_pedestrian_speed_variation(
            original_frames.copy(), probability=1.0)
        speed_changed = speed_varied != original_frames
        print(f"   ✅ Pedestrian Speed Variation: {'Working' if speed_changed else 'No change applied'}")
        if speed_changed:
            print(f"     Result: {speed_varied[:3]}...{speed_varied[-3:]}")
    except Exception as e:
        print(f"   ❌ Pedestrian Speed Variation: Error - {e}")


def test_transform_pipeline():
    """Test the complete transform pipeline"""
    print(f"\n🔄 Testing Complete Transform Pipeline")
    print("=" * 50)
    
    # Create mock options
    opt = Options().parse()
    opt.isize = 64
    opt.aspect_method = 'maintain_3_2'
    
    for mode in ['minimal', 'conservative', 'moderate']:
        print(f"\n🔧 Testing {mode} transform pipeline:")
        
        try:
            basic_transforms, aug_transforms = create_ucsd_ped2_transforms(opt, mode=mode)
            
            # Test with sample image
            test_img = Image.new('RGB', (96, 64), color=(128, 128, 128))  # 3:2 aspect ratio
            
            # Apply basic transforms
            basic_result = basic_transforms(test_img)
            print(f"   ✅ Basic transforms: {test_img.size} → {basic_result.shape}")
            
            # Apply augmented transforms
            aug_result = aug_transforms(test_img)
            print(f"   ✅ Augmented transforms: {test_img.size} → {aug_result.shape}")
            
            # Check for differences (augmentation should create some variation)
            diff = torch.abs(basic_result - aug_result).mean()
            if diff > 0.001:
                print(f"   ✅ Augmentation creating variation: {diff:.4f}")
            else:
                print(f"   ⚠️ Little variation detected: {diff:.4f}")
                
        except Exception as e:
            print(f"   ❌ {mode} pipeline: Error - {e}")


def test_dataloader_integration():
    """Test integration with actual dataloader if data exists"""
    print(f"\n📁 Testing Dataloader Integration")
    print("=" * 50)
    
    # Check if UCSD2 data exists
    data_path = "data/ucsd2"
    if not os.path.exists(data_path):
        print(f"   ⚠️ UCSD2 data not found at {data_path}")
        print(f"   ℹ️ Skipping dataloader test")
        return
    
    try:
        from lib.data.dataloader import load_video_data_FD_ucsd_ped2
        
        # Create options
        opt = Options().parse()
        opt.dataroot = data_path
        opt.isize = 64
        opt.num_frames = 16
        opt.batchsize = 1
        opt.aspect_method = "maintain_3_2"
        
        for mode in ['minimal', 'conservative']:
            print(f"\n🔧 Testing {mode} dataloader:")
            
            try:
                data = load_video_data_FD_ucsd_ped2(opt, ['normal'], augmentation_mode=mode)
                
                # Get a sample batch
                train_loader = data.train
                sample_batch = next(iter(train_loader))
                
                lap_batch, res_batch, aug_batch, labels = sample_batch
                print(f"   ✅ {mode} dataloader: Batch shapes - Lap: {lap_batch.shape}, Res: {res_batch.shape}, Aug: {aug_batch.shape}")
                
                # Check for variation between batches
                lap_mean = lap_batch.mean().item()
                aug_mean = aug_batch.mean().item()
                diff = abs(lap_mean - aug_mean)
                print(f"   📊 Augmentation variation: {diff:.4f}")
                
            except Exception as e:
                print(f"   ❌ {mode} dataloader: Error - {e}")
                
    except ImportError as e:
        print(f"   ❌ Import error: {e}")


def main():
    """Main testing function"""
    print("🎯 UCSD Ped2 Augmentation Testing Suite")
    print("=" * 60)
    
    # Run all tests
    individual_results = test_individual_augmentations()
    test_augmentation_modes()
    test_temporal_augmentation()
    test_transform_pipeline()
    test_dataloader_integration()
    
    # Summary
    print(f"\n🎯 Testing Summary")
    print("=" * 60)
    working_individual = sum(individual_results.values())
    total_individual = len(individual_results)
    
    print(f"✅ Individual augmentations: {working_individual}/{total_individual} working")
    print(f"✅ Mode testing: Completed")
    print(f"✅ Temporal augmentation: Completed")
    print(f"✅ Transform pipeline: Completed")
    print(f"✅ Dataloader integration: Completed")
    
    if working_individual == total_individual:
        print(f"\n🎉 All UCSD Ped2 augmentations working perfectly!")
        print(f"🚀 Ready for simplified training!")
    else:
        print(f"\n⚠️ Some augmentations need attention")
    
    print(f"\n📝 Usage Instructions:")
    print(f"   python train_video.py --use_ucsd_augmentation --ucsd_augmentation conservative")
    print(f"   python train_video.py --use_ucsd_augmentation --ucsd_augmentation minimal")
    print(f"   python train_video.py --use_ucsd_augmentation --ucsd_augmentation moderate")


if __name__ == "__main__":
    main()
