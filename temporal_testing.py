"""
Temporal Attention Testing and Validation Module
Tests temporal attention components and integration with OCR-GAN Video model
"""

import torch
import torch.nn as nn
import numpy as np
import time
from collections import OrderedDict

# Import temporal attention modules with proper error handling
try:
    # Try relative imports first (when imported as module)
    from .temporal_attention import TemporalAttention, TemporalFeatureFusion, ConvLSTM
    from .multiscale_temporal_attention import (
        MultiScaleTemporalAttention, 
        HierarchicalTemporalAttention, 
        AdaptiveTemporalPooling,
        EnhancedTemporalFusion
    )
    from .temporal_losses import (
        TemporalConsistencyLoss, 
        TemporalMotionLoss, 
        TemporalAttentionRegularization,
        CombinedTemporalLoss
    )
except ImportError:
    # Fallback to absolute imports (when run as main)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from temporal_attention import TemporalAttention, TemporalFeatureFusion, ConvLSTM
    from multiscale_temporal_attention import (
        MultiScaleTemporalAttention, 
        HierarchicalTemporalAttention, 
        AdaptiveTemporalPooling,
        EnhancedTemporalFusion
    )
    from temporal_losses import (
        TemporalConsistencyLoss, 
        TemporalMotionLoss, 
        TemporalAttentionRegularization,
        CombinedTemporalLoss
    )


class TemporalAttentionTester:
    """Comprehensive testing suite for temporal attention modules"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.test_results = OrderedDict()
        
    def create_test_data(self, batch_size=2, num_frames=16, channels=64, height=64, width=64):
        """Create synthetic test data for temporal attention modules"""
        # Video tensor: (batch, frames, channels, height, width)
        video_data = torch.randn(batch_size, num_frames, channels, height, width).to(self.device)
        
        # RGB video for input testing: (batch, frames, 3, height, width)
        rgb_video = torch.randn(batch_size, num_frames, 3, height, width).to(self.device)
        
        return video_data, rgb_video
    
    def test_basic_temporal_attention(self):
        """Test basic temporal attention module"""
        print("🔍 Testing Basic Temporal Attention...")
        
        try:
            # Create module
            temporal_attention = TemporalAttention(
                feature_dim=64, 
                num_frames=16, 
                num_heads=8
            ).to(self.device)
            
            # Test data
            video_data, _ = self.create_test_data(channels=64)
            
            # Forward pass
            start_time = time.time()
            output = temporal_attention(video_data)
            forward_time = time.time() - start_time
            
            # Validation checks
            assert output.shape == video_data.shape, f"Shape mismatch: {output.shape} vs {video_data.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            # Memory usage
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['basic_temporal_attention'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in temporal_attention.parameters())
            }
            
            print(f"✅ Basic Temporal Attention: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in temporal_attention.parameters()):,}")
            
        except Exception as e:
            self.test_results['basic_temporal_attention'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Basic Temporal Attention: FAILED - {e}")
    
    def test_multiscale_temporal_attention(self):
        """Test multi-scale temporal attention module"""
        print("\n🔍 Testing Multi-Scale Temporal Attention...")
        
        try:
            # Create module
            multiscale_attention = MultiScaleTemporalAttention(
                feature_dim=64, 
                num_frames=16, 
                num_heads=8
            ).to(self.device)
            
            # Test data
            video_data, _ = self.create_test_data(channels=64)
            
            # Forward pass
            start_time = time.time()
            output = multiscale_attention(video_data)
            forward_time = time.time() - start_time
            
            # Expected output shape: (batch, channels, height, width) - temporally aggregated
            expected_shape = (video_data.shape[0], video_data.shape[2], video_data.shape[3], video_data.shape[4])
            
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['multiscale_temporal_attention'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in multiscale_attention.parameters())
            }
            
            print(f"✅ Multi-Scale Temporal Attention: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in multiscale_attention.parameters()):,}")
            
        except Exception as e:
            self.test_results['multiscale_temporal_attention'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Multi-Scale Temporal Attention: FAILED - {e}")
    
    def test_hierarchical_temporal_attention(self):
        """Test hierarchical temporal attention module"""
        print("\n🔍 Testing Hierarchical Temporal Attention...")
        
        try:
            # Create module
            hierarchical_attention = HierarchicalTemporalAttention(
                feature_dim=64, 
                num_frames=16, 
                num_heads=8
            ).to(self.device)
            
            # Test data
            video_data, _ = self.create_test_data(channels=64)
            
            # Forward pass
            start_time = time.time()
            output = hierarchical_attention(video_data)
            forward_time = time.time() - start_time
            
            # Expected output shape: (batch, channels, height, width) - temporally aggregated
            expected_shape = (video_data.shape[0], video_data.shape[2], video_data.shape[3], video_data.shape[4])
            
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['hierarchical_temporal_attention'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in hierarchical_attention.parameters())
            }
            
            print(f"✅ Hierarchical Temporal Attention: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in hierarchical_attention.parameters()):,}")
            
        except Exception as e:
            self.test_results['hierarchical_temporal_attention'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Hierarchical Temporal Attention: FAILED - {e}")
    
    def test_adaptive_temporal_pooling(self):
        """Test adaptive temporal pooling module"""
        print("\n🔍 Testing Adaptive Temporal Pooling...")
        
        try:
            # Create module
            adaptive_pooling = AdaptiveTemporalPooling(
                feature_dim=64, 
                num_frames=16,
                pool_sizes=[2, 4, 8]
            ).to(self.device)
            
            # Test data
            video_data, _ = self.create_test_data(channels=64)
            
            # Forward pass
            start_time = time.time()
            output = adaptive_pooling(video_data)
            forward_time = time.time() - start_time
            
            # Expected output shape: (batch, channels, height, width) - temporally aggregated
            expected_shape = (video_data.shape[0], video_data.shape[2], video_data.shape[3], video_data.shape[4])
            
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['adaptive_temporal_pooling'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in adaptive_pooling.parameters())
            }
            
            print(f"✅ Adaptive Temporal Pooling: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in adaptive_pooling.parameters()):,}")
            
        except Exception as e:
            self.test_results['adaptive_temporal_pooling'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Adaptive Temporal Pooling: FAILED - {e}")
    
    def test_enhanced_temporal_fusion(self):
        """Test enhanced temporal fusion module"""
        print("\n🔍 Testing Enhanced Temporal Fusion...")
        
        try:
            # Create module
            enhanced_fusion = EnhancedTemporalFusion(
                feature_dim=64, 
                num_frames=16, 
                num_heads=8
            ).to(self.device)
            
            # Test data
            video_data, _ = self.create_test_data(channels=64)
            
            # Forward pass
            start_time = time.time()
            output = enhanced_fusion(video_data)
            forward_time = time.time() - start_time
            
            # Expected output shape: (batch, channels, height, width) - temporally aggregated
            expected_shape = (video_data.shape[0], video_data.shape[2], video_data.shape[3], video_data.shape[4])
            
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['enhanced_temporal_fusion'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in enhanced_fusion.parameters())
            }
            
            print(f"✅ Enhanced Temporal Fusion: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in enhanced_fusion.parameters()):,}")
            
        except Exception as e:
            self.test_results['enhanced_temporal_fusion'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Enhanced Temporal Fusion: FAILED - {e}")
    
    def test_temporal_losses(self):
        """Test temporal loss functions"""
        print("\n🔍 Testing Temporal Loss Functions...")
        
        try:
            # Create synthetic video data for testing losses
            batch_size, num_frames = 2, 16
            real_frames = torch.randn(batch_size, num_frames, 3, 64, 64).to(self.device)
            fake_frames = torch.randn(batch_size, num_frames, 3, 64, 64).to(self.device)
            features = torch.randn(batch_size, num_frames, 64, 16, 16).to(self.device)
              # Test individual loss components
            temporal_consistency = TemporalConsistencyLoss().to(self.device)
            temporal_motion = TemporalMotionLoss().to(self.device)
            temporal_reg = TemporalAttentionRegularization().to(self.device)
            
            # Create dummy attention weights for regularization test
            # Shape: (B*H*W, heads, T, T) where B=2, H=16, W=16, heads=8, T=16
            attention_weights = torch.softmax(
                torch.randn(batch_size * 16 * 16, 8, num_frames, num_frames), 
                dim=-1
            ).to(self.device)
            
            # Test loss computations
            consistency_loss = temporal_consistency(fake_frames, features)
            motion_loss = temporal_motion(real_frames, fake_frames)
            reg_loss = temporal_reg(attention_weights)            # Test combined loss
            combined_loss = CombinedTemporalLoss().to(self.device)
            total_losses = combined_loss(real_frames, fake_frames, features, attention_weights)
              # Validation checks
            assert isinstance(consistency_loss, torch.Tensor), "Consistency loss should be tensor"
            assert isinstance(motion_loss, torch.Tensor), "Motion loss should be tensor"
            assert isinstance(reg_loss, torch.Tensor), "Regularization loss should be tensor"
            assert isinstance(total_losses, dict), "Combined loss should return dictionary"
            
            assert not torch.isnan(consistency_loss), "Consistency loss contains NaN"
            assert not torch.isnan(motion_loss), "Motion loss contains NaN"
            assert not torch.isnan(reg_loss), "Regularization loss contains NaN"
            assert not torch.isnan(total_losses['total_temporal']), "Total temporal loss contains NaN"
            
            self.test_results['temporal_losses'] = {
                'status': 'PASSED',
                'consistency_loss': consistency_loss.item(),
                'motion_loss': motion_loss.item(),
                'regularization_loss': reg_loss.item(),
                'total_temporal_loss': total_losses['total_temporal'].item()
            }
            
            print(f"✅ Temporal Loss Functions: PASSED")
            print(f"   - Consistency Loss: {consistency_loss.item():.6f}")
            print(f"   - Motion Loss: {motion_loss.item():.6f}")
            print(f"   - Regularization Loss: {reg_loss.item():.6f}")
            print(f"   - Total Temporal Loss: {total_losses['total_temporal'].item():.6f}")
            
        except Exception as e:
            self.test_results['temporal_losses'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Temporal Loss Functions: FAILED - {e}")
    
    def test_temporal_feature_fusion(self):
        """Test temporal feature fusion module"""
        print("\n🔍 Testing Temporal Feature Fusion...")
        
        try:
            # Create module for RGB input (3 channels)
            temporal_fusion = TemporalFeatureFusion(
                feature_dim=3, 
                num_frames=16
            ).to(self.device)
            
            # Test data
            _, rgb_video = self.create_test_data(channels=3)
            
            # Forward pass
            start_time = time.time()
            output = temporal_fusion(rgb_video)
            forward_time = time.time() - start_time
            
            # Expected output shape: (batch, channels, height, width) - temporally aggregated
            expected_shape = (rgb_video.shape[0], rgb_video.shape[2], rgb_video.shape[3], rgb_video.shape[4])
            
            assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.test_results['temporal_feature_fusion'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(output.shape),
                'parameters': sum(p.numel() for p in temporal_fusion.parameters())
            }
            
            print(f"✅ Temporal Feature Fusion: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Parameters: {sum(p.numel() for p in temporal_fusion.parameters()):,}")
            
        except Exception as e:
            self.test_results['temporal_feature_fusion'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Temporal Feature Fusion: FAILED - {e}")
    
    def run_all_tests(self):
        """Run comprehensive temporal attention testing suite"""
        print("🧪 Starting Comprehensive Temporal Attention Testing Suite")
        print("=" * 60)
        
        # Clear GPU memory before testing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run all tests
        self.test_basic_temporal_attention()
        self.test_multiscale_temporal_attention()
        self.test_hierarchical_temporal_attention()
        self.test_adaptive_temporal_pooling()
        self.test_enhanced_temporal_fusion()
        self.test_temporal_losses()
        self.test_temporal_feature_fusion()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🧪 TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            if status == 'PASSED':
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed_tests += 1
                print(f"❌ {test_name}: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        total_tests = passed_tests + failed_tests
        success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        if passed_tests > 0:
            print(f"\n⚡ Performance Summary:")
            total_params = sum(result.get('parameters', 0) for result in self.test_results.values() if result['status'] == 'PASSED')
            avg_forward_time = np.mean([result.get('forward_time', 0) for result in self.test_results.values() if result['status'] == 'PASSED' and 'forward_time' in result])
            
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Average Forward Time: {avg_forward_time:.4f}s")
            
            if torch.cuda.is_available():
                max_memory = max([result.get('memory_usage_mb', 0) for result in self.test_results.values() if result['status'] == 'PASSED'])
                print(f"   Peak Memory Usage: {max_memory:.2f}MB")
        
        return self.test_results


def main():
    """Main testing function"""
    print("🚀 Temporal Attention Testing Module")
    print("Testing temporal attention components for OCR-GAN Video model")
    
    # Initialize tester
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tester = TemporalAttentionTester(device=device)
    
    # Run all tests
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    main()
