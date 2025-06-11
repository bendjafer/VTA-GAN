#!/usr/bin/env python3
"""
Test script for VideoAspectRatioResize and aspect ratio preservation functionality
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

def test_aspect_ratio_pipeline():
    """Test the complete aspect ratio preservation pipeline"""
    
    print("🧪 Testing Aspect Ratio Preservation Pipeline")
    print("=" * 50)
    
    # Create mock options
    opt = MockOpt()
    
    # Test calculate_optimal_dimensions function
    print(f"📊 Input configuration:")
    print(f"   - Base size: {opt.isize}x{opt.isize}")
    print(f"   - Aspect method: {opt.aspect_method}")
    
    target_size = calculate_optimal_dimensions(opt)
    print(f"\n📐 Calculated dimensions:")
    print(f"   - Target size: {target_size}")
    print(f"   - Effective isize: {getattr(opt, 'effective_isize', 'Not set')}")
    
    # Create test images with different aspect ratios
    test_cases = [
        ("UCSD Video Frame", (360, 240)),    # Standard UCSD Ped2 frame
        ("Square Image", (240, 240)),        # Square test
        ("Portrait", (240, 320)),            # Portrait orientation
        ("Wide", (480, 240)),                # Extra wide
    ]
    
    print(f"\n🖼️  Testing VideoAspectRatioResize transform:")
    
    for name, size in test_cases:
        print(f"\n   {name} ({size[0]}x{size[1]}):")
        
        # Create test image
        test_img = Image.new('RGB', size, color='red')
        
        # Apply transform
        transform = VideoAspectRatioResize(opt.isize, target_size, method=opt.aspect_method)
        result = transform(test_img)
        
        print(f"      Original: {test_img.size} → Final: {result.size}")
        
        # Verify result is correct size
        expected_size = (opt.isize, opt.isize)
        if result.size == expected_size:
            print(f"      ✅ Correct output size: {result.size}")
        else:
            print(f"      ❌ Wrong output size: {result.size}, expected: {expected_size}")
    
    print(f"\n🎯 Testing different aspect methods:")
    
    # Test different methods
    methods = ['maintain_3_2', 'center_crop', 'pad_square', 'stretch']
    test_img = Image.new('RGB', (360, 240), color='blue')
    
    for method in methods:
        opt.aspect_method = method
        target_size = calculate_optimal_dimensions(opt)
        transform = VideoAspectRatioResize(opt.isize, target_size, method=method)
        result = transform(test_img)
        print(f"   {method}: {test_img.size} → {result.size}")
    
    print(f"\n✅ Aspect ratio pipeline test completed!")

if __name__ == "__main__":
    test_aspect_ratio_pipeline()
