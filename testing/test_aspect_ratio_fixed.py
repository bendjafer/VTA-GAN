#!/usr/bin/env python3
"""
Test script for VideoAspectRatioResize functionality
"""

import sys
sys.path.append('lib')

from data.dataloader import VideoAspectRatioResize, calculate_optimal_dimensions
import argparse
from PIL import Image
import numpy as np

# Mock options for testing
class MockOpt:
    def __init__(self):
        self.isize = 64
        self.aspect_method = 'maintain_3_2'

def test_aspect_ratio_transform():
    """Test the aspect ratio transform pipeline"""
    print("🧪 Testing VideoAspectRatioResize Transform Pipeline")
    print("=" * 55)
    
    # Create mock options
    opt = MockOpt()
    
    # Calculate optimal dimensions
    target_size = calculate_optimal_dimensions(opt)
    
    print(f"📐 Configuration:")
    print(f"   Input size: {opt.isize}x{opt.isize}")
    print(f"   Target size: {target_size}")
    print(f"   Method: {opt.aspect_method}")
    print(f"   Effective isize: {getattr(opt, 'effective_isize', 'Not set')}")
    print()
    
    # Test the transform
    print(f"🔄 Testing transform pipeline...")
    
    # Create a test image simulating 360x240 video frame
    test_img = Image.new('RGB', (360, 240), color=(255, 0, 0))  # Red image
    
    # Create transform
    transform = VideoAspectRatioResize(opt.isize, target_size, method=opt.aspect_method)
    
    # Apply transform
    result = transform(test_img)
    
    print(f"   Original size: {test_img.size} (360×240)")
    print(f"   Final size: {result.size}")
    print()
    
    # Verify expected behavior
    expected_size = (opt.isize, opt.isize)
    if result.size == expected_size:
        print("✅ Transform pipeline test SUCCESSFUL!")
        print(f"   ✓ Output is correct square size: {result.size}")
    else:
        print("❌ Transform pipeline test FAILED!")
        print(f"   ✗ Expected: {expected_size}, Got: {result.size}")
    
    print()
    print("🔍 Pipeline Analysis:")
    print(f"   Step 1: 360×240 → {target_size} (resize maintaining aspect)")
    print(f"   Step 2: {target_size} → {opt.isize}×{opt.isize} (center crop to square)")
    print(f"   Result: No distortion, maintains video quality")
    
    return result.size == expected_size

def test_different_methods():
    """Test different aspect ratio methods"""
    print("\n🔄 Testing Different Aspect Ratio Methods")
    print("=" * 45)
    
    methods = ['maintain_3_2', 'center_crop', 'pad_square', 'stretch']
    test_img = Image.new('RGB', (360, 240), color=(0, 255, 0))  # Green image
    
    for method in methods:
        opt = MockOpt()
        opt.aspect_method = method
        
        if method == 'maintain_3_2':
            target_size = calculate_optimal_dimensions(opt)
        else:
            target_size = (opt.isize, opt.isize)  # Default for other methods
        
        transform = VideoAspectRatioResize(opt.isize, target_size, method=method)
        result = transform(test_img)
        
        print(f"   {method:15}: {test_img.size} → {result.size}")
    
    print()

if __name__ == "__main__":
    print("🎯 OCR-GAN Video Aspect Ratio Transform Test")
    print("=" * 50)
    
    # Test main functionality
    success = test_aspect_ratio_transform()
    
    # Test different methods
    test_different_methods()
    
    if success:
        print("🎉 All tests passed! The aspect ratio transform is working correctly.")
    else:
        print("❌ Tests failed! Please check the implementation.")
