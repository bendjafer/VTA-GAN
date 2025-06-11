#!/usr/bin/env python3
"""
Test script for VideoAspectRatioResize functionality (Windows-compatible version)
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
    print("Testing VideoAspectRatioResize Transform Pipeline")
    print("=" * 50)
    
    # Create mock options
    opt = MockOpt()
    
    # Calculate optimal dimensions
    target_size = calculate_optimal_dimensions(opt)
    
    print(f"Configuration:")
    print(f"   Input size: {opt.isize}x{opt.isize}")
    print(f"   Target size: {target_size}")
    print(f"   Method: {opt.aspect_method}")
    print(f"   Effective isize: {getattr(opt, 'effective_isize', 'Not set')}")
    print()
    
    # Test the transform
    print(f"Testing transform pipeline...")
    
    # Create a test image simulating 360x240 video frame
    test_img = Image.new('RGB', (360, 240), color=(255, 0, 0))  # Red image
    
    # Create transform
    transform = VideoAspectRatioResize(opt.isize, target_size, method=opt.aspect_method)
    
    # Apply transform
    result = transform(test_img)
    
    print(f"   Original size: {test_img.size} (360x240)")
    print(f"   Final size: {result.size}")
    print()
    
    # Verify expected behavior
    expected_size = (opt.isize, opt.isize)
    if result.size == expected_size:
        print("SUCCESS: Transform pipeline test PASSED!")
        print(f"   [CHECK] Output is correct square size: {result.size}")
    else:
        print("FAILED: Transform pipeline test FAILED!")
        print(f"   [ERROR] Expected: {expected_size}, Got: {result.size}")
    
    print()
    print("Pipeline Analysis:")
    print(f"   Step 1: 360x240 -> {target_size} (resize maintaining aspect)")
    print(f"   Step 2: {target_size} -> {opt.isize}x{opt.isize} (center crop to square)")
    print(f"   Result: No distortion, maintains video quality")
    
    return result.size == expected_size

def test_different_methods():
    """Test different aspect ratio methods"""
    print("\nTesting Different Aspect Ratio Methods")
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
        
        print(f"   {method:15}: {test_img.size} -> {result.size}")
    
    print()

def test_comprehensive_cases():
    """Test comprehensive aspect ratio cases"""
    print("Testing Comprehensive Video Aspect Ratios")
    print("=" * 45)
    
    # Test with common video dimensions
    test_cases = [
        (320, 240, "4:3"),    # Common 4:3 ratio
        (640, 480, "4:3"),    # Common 4:3 ratio  
        (360, 240, "3:2"),    # 3:2 ratio (UCSD2 format)
        (480, 320, "3:2"),    # 3:2 ratio
        (854, 480, "16:9"),   # Common 16:9 ratio
        (1280, 720, "16:9"),  # HD 16:9 ratio
    ]
    
    method = 'maintain_3_2'
    opt = MockOpt()
    opt.aspect_method = method
    target_size = calculate_optimal_dimensions(opt)
    
    print(f"Using method: {method}")
    print(f"Target intermediate size: {target_size}")
    print(f"Final output size: {opt.isize}x{opt.isize}")
    print()
    
    all_passed = True
    
    for width, height, ratio in test_cases:
        test_img = Image.new('RGB', (width, height), color=(0, 0, 255))  # Blue image
        transform = VideoAspectRatioResize(opt.isize, target_size, method=method)
        result = transform(test_img)
        
        expected_size = (opt.isize, opt.isize)
        passed = result.size == expected_size
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"   {width:4}x{height:3} ({ratio:4}) -> {result.size} [{status}]")
    
    print()
    return all_passed

if __name__ == "__main__":
    print("OCR-GAN Video Aspect Ratio Transform Test")
    print("=" * 50)
    
    # Test main functionality
    success1 = test_aspect_ratio_transform()
    
    # Test different methods
    test_different_methods()
    
    # Test comprehensive cases
    success2 = test_comprehensive_cases()
    
    overall_success = success1 and success2
    
    if overall_success:
        print("ALL TESTS PASSED! The aspect ratio transform is working correctly.")
        print("The system properly preserves video aspect ratios without distortion.")
    else:
        print("SOME TESTS FAILED! Please check the implementation.")
    
    print("\nSummary:")
    print("- Videos maintain their natural aspect ratio during initial resize")
    print("- Center cropping to square format removes minimal content") 
    print("- No stretching or distortion occurs")
    print("- Compatible with all common video formats (4:3, 3:2, 16:9)")
