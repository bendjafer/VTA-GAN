"""
Comprehensive Training Pipeline Validation for OCR-GAN Video with Temporal Attention
Tests the complete training workflow with temporal enhancements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from collections import OrderedDict
from types import SimpleNamespace

# Import model components
from ocr_gan_video import Ocr_Gan_Video
from temporal_testing import TemporalAttentionTester
from temporal_unet_generator import define_temporal_G


class TrainingPipelineValidator:
    """Validates the complete training pipeline with temporal attention"""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.validation_results = OrderedDict()
        
    def create_mock_options(self):
        """Create mock options for testing"""
        opt = SimpleNamespace()
        
        # Basic model parameters
        opt.nc = 3  # Number of channels
        opt.nz = 100  # Latent dimension
        opt.ngf = 64  # Generator filters
        opt.ndf = 64  # Discriminator filters
        opt.isize = 64  # Image size
        opt.batchsize = 2  # Small batch for testing
        opt.lr = 0.0002  # Learning rate
        opt.beta1 = 0.5  # Adam beta1
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        opt.ngpu = 1 if torch.cuda.is_available() else 0
        opt.manualseed = 42
        opt.verbose = False
        opt.extralayers = 0
        
        # Video-specific parameters
        opt.num_frames = 16
        
        # Temporal attention parameters
        opt.use_temporal_attention = True
        opt.use_temporal_loss = True
        opt.w_temporal_consistency = 0.1
        opt.w_temporal_motion = 0.05
        opt.w_temporal_reg = 0.01
        
        # Loss weights
        opt.w_adv = 1.0
        opt.w_con = 50.0
        opt.w_lat = 1.0
        
        # Training parameters
        opt.niter = 2  # Small number for testing
        opt.iter = 0
        opt.print_freq = 1
        opt.save_image_freq = 1
        opt.isTrain = True
        opt.name = 'temporal_test'
        opt.outf = './test_output'
        opt.dataset = 'test_dataset'
        opt.note = 'testing'
        opt.resume = ''
        opt.load_weights = False
        opt.lr_policy = 'constant'
        opt.display = False
        opt.display_id = 0
        opt.save_test_images = False
        
        return opt
    
    def create_mock_data_loader(self, opt):
        """Create mock data loader for testing"""
        class MockDataset:
            def __init__(self, num_samples=10):
                self.num_samples = num_samples
                
            def __len__(self):
                return self.num_samples
                
            def __getitem__(self, idx):
                # Return video data: (lap_tensor, res_tensor, target)
                lap_tensor = torch.randn(opt.num_frames, opt.nc, opt.isize, opt.isize)
                res_tensor = torch.randn(opt.num_frames, opt.nc, opt.isize, opt.isize)
                target = torch.randint(0, 2, (1,)).float()  # Binary label
                
                return lap_tensor, res_tensor, target
        
        class MockDataLoader:
            def __init__(self, dataset, batch_size):
                self.dataset = dataset
                self.batch_size = batch_size
                
            def __len__(self):
                return len(self.dataset) // self.batch_size
                
            def __iter__(self):
                for i in range(len(self)):
                    batch_data = []
                    for j in range(self.batch_size):
                        idx = (i * self.batch_size + j) % len(self.dataset)
                        batch_data.append(self.dataset[idx])
                    
                    # Collate batch
                    lap_batch = torch.stack([item[0] for item in batch_data])
                    res_batch = torch.stack([item[1] for item in batch_data])
                    target_batch = torch.stack([item[2] for item in batch_data])
                    
                    yield lap_batch, res_batch, target_batch
        
        class MockData:
            def __init__(self, opt):
                train_dataset = MockDataset(20)  # Small training set
                valid_dataset = MockDataset(10)  # Small validation set
                
                self.train = MockDataLoader(train_dataset, opt.batchsize)
                self.valid = MockDataLoader(valid_dataset, opt.batchsize)
        
        return MockData(opt)
    
    def test_model_initialization(self):
        """Test OCR-GAN Video model initialization with temporal attention"""
        print("🔍 Testing Model Initialization...")
        
        try:
            opt = self.create_mock_options()
            data = self.create_mock_data_loader(opt)
            classes = ['normal', 'anomaly']
            
            # Initialize model
            start_time = time.time()
            model = Ocr_Gan_Video(opt, data, classes)
            init_time = time.time() - start_time
            
            # Check if temporal components are properly initialized
            assert hasattr(model, 'use_temporal_attention'), "Model should have temporal attention flag"
            assert hasattr(model, 'temporal_loss'), "Model should have temporal loss"
            
            if model.use_temporal_attention:
                assert hasattr(model, 'temporal_attention_gen'), "Model should have generator temporal attention"
                assert hasattr(model, 'temporal_attention_disc'), "Model should have discriminator temporal attention"
                assert hasattr(model, 'temporal_fusion'), "Model should have temporal fusion"
            
            # Check parameter counts
            total_params = sum(p.numel() for p in model.netg.parameters())
            temporal_params = 0
            if hasattr(model, 'temporal_attention_gen'):
                temporal_params += sum(p.numel() for p in model.temporal_attention_gen.parameters())
            if hasattr(model, 'temporal_fusion'):
                temporal_params += sum(p.numel() for p in model.temporal_fusion.parameters())
            
            self.validation_results['model_initialization'] = {
                'status': 'PASSED',
                'init_time': init_time,
                'total_params': total_params,
                'temporal_params': temporal_params,
                'temporal_ratio': temporal_params / total_params if total_params > 0 else 0
            }
            
            print(f"✅ Model Initialization: PASSED")
            print(f"   - Initialization time: {init_time:.4f}s")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Temporal parameters: {temporal_params:,}")
            print(f"   - Temporal ratio: {temporal_params / total_params * 100:.2f}%")
            
            return model, opt, data, classes
            
        except Exception as e:
            self.validation_results['model_initialization'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Model Initialization: FAILED - {e}")
            return None, None, None, None
    
    def test_forward_pass(self, model, opt):
        """Test forward pass with temporal attention"""
        print("\n🔍 Testing Forward Pass...")
        
        try:
            model.netg.eval()
            model.netd.eval()
            
            # Create test input
            batch_size, num_frames = opt.batchsize, opt.num_frames
            input_lap = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            input_res = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            target = torch.randint(0, 2, (batch_size,)).float().to(self.device)
            
            # Set input
            test_data = (input_lap, input_res, target)
            model.set_input(test_data, noise=True)
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                model.forward()
            forward_time = time.time() - start_time
            
            # Check outputs
            assert hasattr(model, 'fake'), "Model should have fake output"
            assert hasattr(model, 'fake_lap'), "Model should have fake_lap output"
            assert hasattr(model, 'fake_res'), "Model should have fake_res output"
            
            # Check output shapes
            expected_shape = (batch_size, num_frames, opt.nc, opt.isize, opt.isize)
            assert model.fake.shape == expected_shape, f"Wrong fake shape: {model.fake.shape} vs {expected_shape}"
            
            # Check for NaN values
            assert not torch.isnan(model.fake).any(), "Fake output contains NaN"
            assert not torch.isnan(model.fake_lap).any(), "Fake lap output contains NaN"
            assert not torch.isnan(model.fake_res).any(), "Fake res output contains NaN"
            
            memory_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            self.validation_results['forward_pass'] = {
                'status': 'PASSED',
                'forward_time': forward_time,
                'memory_usage_mb': memory_usage,
                'output_shape': list(model.fake.shape)
            }
            
            print(f"✅ Forward Pass: PASSED")
            print(f"   - Forward time: {forward_time:.4f}s")
            print(f"   - Memory usage: {memory_usage:.2f}MB")
            print(f"   - Output shape: {model.fake.shape}")
            
        except Exception as e:
            self.validation_results['forward_pass'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Forward Pass: FAILED - {e}")
    
    def test_loss_computation(self, model, opt):
        """Test loss computation with temporal losses"""
        print("\n🔍 Testing Loss Computation...")
        
        try:
            model.netg.train()
            model.netd.train()
            
            # Create test input
            batch_size, num_frames = opt.batchsize, opt.num_frames
            input_lap = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            input_res = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            target = torch.randint(0, 2, (batch_size,)).float().to(self.device)
            
            # Set input and forward pass
            test_data = (input_lap, input_res, target)
            model.set_input(test_data, noise=True)
            model.forward()
            
            # Test loss computation
            start_time = time.time()
            
            # Generator losses
            model.optimizer_g.zero_grad()
            model.backward_g()
            g_loss_time = time.time() - start_time
            
            # Discriminator losses
            start_time = time.time()
            model.optimizer_d.zero_grad()
            model.backward_d()
            d_loss_time = time.time() - start_time
            
            # Get errors
            errors = model.get_errors()
            
            # Check required losses
            required_losses = ['err_d', 'err_g', 'err_g_adv', 'err_g_con', 'err_g_lat']
            for loss_name in required_losses:
                assert loss_name in errors, f"Missing required loss: {loss_name}"
                assert not np.isnan(errors[loss_name]), f"Loss {loss_name} is NaN"
            
            # Check temporal loss if enabled
            if model.use_temporal_attention and 'err_g_temporal' in errors:
                assert not np.isnan(errors['err_g_temporal']), "Temporal loss is NaN"
            
            self.validation_results['loss_computation'] = {
                'status': 'PASSED',
                'g_loss_time': g_loss_time,
                'd_loss_time': d_loss_time,
                'losses': dict(errors),
                'has_temporal_loss': 'err_g_temporal' in errors
            }
            
            print(f"✅ Loss Computation: PASSED")
            print(f"   - Generator loss time: {g_loss_time:.4f}s")
            print(f"   - Discriminator loss time: {d_loss_time:.4f}s")
            print(f"   - Total losses computed: {len(errors)}")
            print(f"   - Has temporal loss: {'err_g_temporal' in errors}")
            
            for loss_name, loss_value in errors.items():
                print(f"     {loss_name}: {loss_value:.6f}")
            
        except Exception as e:
            self.validation_results['loss_computation'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Loss Computation: FAILED - {e}")
    
    def test_training_step(self, model, opt):
        """Test complete training step"""
        print("\n🔍 Testing Training Step...")
        
        try:
            model.netg.train()
            model.netd.train()
            
            # Create test input
            batch_size, num_frames = opt.batchsize, opt.num_frames
            input_lap = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            input_res = torch.randn(batch_size, num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
            target = torch.randint(0, 2, (batch_size,)).float().to(self.device)
            
            # Set input
            test_data = (input_lap, input_res, target)
            model.set_input(test_data, noise=True)
            
            # Store initial parameters for comparison
            initial_g_params = [p.clone() for p in model.netg.parameters()]
            initial_d_params = [p.clone() for p in model.netd.parameters()]
            
            # Perform training step
            start_time = time.time()
            model.optimize_params()
            training_step_time = time.time() - start_time
            
            # Check if parameters updated
            g_params_updated = any(not torch.equal(p1, p2) for p1, p2 in 
                                 zip(initial_g_params, model.netg.parameters()))
            d_params_updated = any(not torch.equal(p1, p2) for p1, p2 in 
                                 zip(initial_d_params, model.netd.parameters()))
            
            assert g_params_updated, "Generator parameters not updated"
            assert d_params_updated, "Discriminator parameters not updated"
            
            # Get final errors
            errors = model.get_errors()
            
            self.validation_results['training_step'] = {
                'status': 'PASSED',
                'training_step_time': training_step_time,
                'g_params_updated': g_params_updated,
                'd_params_updated': d_params_updated,
                'final_losses': dict(errors)
            }
            
            print(f"✅ Training Step: PASSED")
            print(f"   - Training step time: {training_step_time:.4f}s")
            print(f"   - Generator params updated: {g_params_updated}")
            print(f"   - Discriminator params updated: {d_params_updated}")
            
        except Exception as e:
            self.validation_results['training_step'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Training Step: FAILED - {e}")
    
    def test_multi_epoch_training(self, model, opt, data):
        """Test multi-epoch training"""
        print("\n🔍 Testing Multi-Epoch Training...")
        
        try:
            initial_auc = 0.5  # Random baseline
            
            # Track training metrics
            epoch_losses = []
            
            start_time = time.time()
            for epoch in range(opt.niter):
                epoch_start = time.time()
                
                # Training epoch
                model.netg.train()
                model.netd.train()
                
                batch_losses = []
                for i, batch_data in enumerate(data.train):
                    # Set input and optimize
                    model.set_input(batch_data, noise=True)
                    model.optimize_params()
                    
                    # Collect losses
                    errors = model.get_errors()
                    batch_losses.append(errors['err_g'])
                    
                    if i >= 2:  # Limit batches for testing
                        break
                
                epoch_time = time.time() - epoch_start
                avg_loss = np.mean(batch_losses)
                epoch_losses.append(avg_loss)
                
                print(f"     Epoch {epoch + 1}/{opt.niter}: Loss={avg_loss:.4f}, Time={epoch_time:.2f}s")
            
            total_training_time = time.time() - start_time
            
            # Check for convergence indicators
            loss_trend = np.diff(epoch_losses[-3:]) if len(epoch_losses) >= 3 else [0]
            loss_stable = np.abs(np.mean(loss_trend)) < 1.0  # Reasonable stability
            
            self.validation_results['multi_epoch_training'] = {
                'status': 'PASSED',
                'total_training_time': total_training_time,
                'epoch_losses': epoch_losses,
                'final_loss': epoch_losses[-1] if epoch_losses else 0,
                'loss_stable': loss_stable,
                'epochs_completed': len(epoch_losses)
            }
            
            print(f"✅ Multi-Epoch Training: PASSED")
            print(f"   - Total training time: {total_training_time:.4f}s")
            print(f"   - Epochs completed: {len(epoch_losses)}")
            print(f"   - Final loss: {epoch_losses[-1]:.6f}")
            print(f"   - Training stable: {loss_stable}")
            
        except Exception as e:
            self.validation_results['multi_epoch_training'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Multi-Epoch Training: FAILED - {e}")
    
    def test_memory_efficiency(self, model, opt):
        """Test memory efficiency and performance"""
        print("\n🔍 Testing Memory Efficiency...")
        
        try:
            if not torch.cuda.is_available():
                print("   Skipping memory test (CUDA not available)")
                self.validation_results['memory_efficiency'] = {'status': 'SKIPPED', 'reason': 'CUDA not available'}
                return
            
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Test with different batch sizes
            batch_sizes = [1, 2, 4]
            memory_usage = {}
            forward_times = {}
            
            for batch_size in batch_sizes:
                try:
                    # Create input
                    input_lap = torch.randn(batch_size, opt.num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
                    input_res = torch.randn(batch_size, opt.num_frames, opt.nc, opt.isize, opt.isize).to(self.device)
                    target = torch.randint(0, 2, (batch_size,)).float().to(self.device)
                    
                    # Set input
                    test_data = (input_lap, input_res, target)
                    model.set_input(test_data, noise=True)
                    
                    # Forward pass with timing
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    with torch.no_grad():
                        model.forward()
                    
                    torch.cuda.synchronize()
                    forward_time = time.time() - start_time
                    
                    # Memory usage
                    current_memory = torch.cuda.memory_allocated()
                    memory_usage[batch_size] = (current_memory - initial_memory) / 1024**2  # MB
                    forward_times[batch_size] = forward_time
                    
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        memory_usage[batch_size] = float('inf')
                        forward_times[batch_size] = float('inf')
                        break
                    else:
                        raise e
            
            # Calculate efficiency metrics
            valid_batches = [bs for bs in batch_sizes if memory_usage.get(bs, float('inf')) != float('inf')]
            max_batch_size = max(valid_batches) if valid_batches else 0
            
            # Memory scaling efficiency
            if len(valid_batches) >= 2:
                mem_scaling = memory_usage[valid_batches[-1]] / memory_usage[valid_batches[0]]
                time_scaling = forward_times[valid_batches[-1]] / forward_times[valid_batches[0]]
                batch_scaling = valid_batches[-1] / valid_batches[0]
                
                memory_efficiency = batch_scaling / mem_scaling
                time_efficiency = batch_scaling / time_scaling
            else:
                memory_efficiency = 1.0
                time_efficiency = 1.0
            
            self.validation_results['memory_efficiency'] = {
                'status': 'PASSED',
                'max_batch_size': max_batch_size,
                'memory_usage': memory_usage,
                'forward_times': forward_times,
                'memory_efficiency': memory_efficiency,
                'time_efficiency': time_efficiency
            }
            
            print(f"✅ Memory Efficiency: PASSED")
            print(f"   - Max batch size: {max_batch_size}")
            print(f"   - Memory efficiency: {memory_efficiency:.2f}")
            print(f"   - Time efficiency: {time_efficiency:.2f}")
            
            for bs in batch_sizes:
                if bs in memory_usage and memory_usage[bs] != float('inf'):
                    print(f"     Batch {bs}: {memory_usage[bs]:.1f}MB, {forward_times[bs]:.4f}s")
            
        except Exception as e:
            self.validation_results['memory_efficiency'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Memory Efficiency: FAILED - {e}")
    
    def run_full_validation(self):
        """Run complete training pipeline validation"""
        print("🧪 Starting Comprehensive Training Pipeline Validation")
        print("=" * 70)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Test model initialization
        model, opt, data, classes = self.test_model_initialization()
        if model is None:
            print("❌ Cannot continue validation due to initialization failure")
            return self.validation_results
        
        # Run validation tests
        self.test_forward_pass(model, opt)
        self.test_loss_computation(model, opt)
        self.test_training_step(model, opt)
        self.test_multi_epoch_training(model, opt, data)
        self.test_memory_efficiency(model, opt)
        
        # Print summary
        print("\n" + "=" * 70)
        print("🧪 TRAINING PIPELINE VALIDATION SUMMARY")
        print("=" * 70)
        
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for test_name, result in self.validation_results.items():
            status = result['status']
            if status == 'PASSED':
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            elif status == 'SKIPPED':
                skipped_tests += 1
                print(f"⏭️ {test_name}: SKIPPED")
            else:
                failed_tests += 1
                print(f"❌ {test_name}: FAILED")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        total_tests = passed_tests + failed_tests + skipped_tests
        success_rate = passed_tests / (total_tests - skipped_tests) * 100 if (total_tests - skipped_tests) > 0 else 0
        
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Skipped: {skipped_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        if passed_tests > 0:
            print(f"\n⚡ Performance Summary:")
            if 'model_initialization' in self.validation_results:
                init_result = self.validation_results['model_initialization']
                if init_result['status'] == 'PASSED':
                    print(f"   Model Parameters: {init_result['total_params']:,}")
                    print(f"   Temporal Parameters: {init_result['temporal_params']:,}")
            
            if 'forward_pass' in self.validation_results:
                forward_result = self.validation_results['forward_pass']
                if forward_result['status'] == 'PASSED':
                    print(f"   Forward Pass Time: {forward_result['forward_time']:.4f}s")
            
            if 'memory_efficiency' in self.validation_results:
                memory_result = self.validation_results['memory_efficiency']
                if memory_result['status'] == 'PASSED':
                    print(f"   Max Batch Size: {memory_result['max_batch_size']}")
        
        return self.validation_results


def main():
    """Main validation function"""
    print("🚀 OCR-GAN Video Training Pipeline Validation")
    print("Testing complete training workflow with temporal attention enhancements")
    
    # Initialize validator
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    validator = TrainingPipelineValidator(device=device)
    
    # Run complete validation
    results = validator.run_full_validation()
    
    # Additional component testing
    print("\n" + "=" * 70)
    print("🔧 ADDITIONAL COMPONENT TESTING")
    print("=" * 70)
    
    # Test temporal attention components
    print("\n🧪 Testing Temporal Attention Components...")
    temporal_tester = TemporalAttentionTester(device=device)
    temporal_results = temporal_tester.run_all_tests()
    
    # Combine results
    all_results = {
        'training_pipeline': results,
        'temporal_components': temporal_results
    }
    
    return all_results


if __name__ == "__main__":
    main()
