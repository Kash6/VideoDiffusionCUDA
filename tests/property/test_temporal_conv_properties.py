"""
Property-Based Tests for Temporal 3D Convolution CUDA Kernel

These tests verify correctness properties of the custom temporal convolution
kernel across randomized input configurations.
"""

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings, strategies as st, assume
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Try to import custom CUDA operations
try:
    import video_diffusion_cuda_ops
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    video_diffusion_cuda_ops = None


# Hypothesis strategies for generating test inputs
@st.composite
def conv3d_config(draw):
    """Generate valid conv3d configuration."""
    batch_size = draw(st.integers(1, 4))
    in_channels = draw(st.integers(1, 16))
    out_channels = draw(st.integers(1, 16))
    
    # Input dimensions
    in_frames = draw(st.integers(4, 16))
    in_height = draw(st.integers(8, 32))
    in_width = draw(st.integers(8, 32))
    
    # Kernel dimensions
    kernel_t = draw(st.integers(1, min(3, in_frames)))
    kernel_h = draw(st.integers(1, min(5, in_height)))
    kernel_w = draw(st.integers(1, min(5, in_width)))
    
    # Stride
    stride_t = draw(st.integers(1, 2))
    stride_h = draw(st.integers(1, 2))
    stride_w = draw(st.integers(1, 2))
    
    # Padding
    pad_t = draw(st.integers(0, kernel_t // 2))
    pad_h = draw(st.integers(0, kernel_h // 2))
    pad_w = draw(st.integers(0, kernel_w // 2))
    
    # Calculate output dimensions
    out_frames = (in_frames + 2 * pad_t - kernel_t) // stride_t + 1
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Ensure valid output dimensions
    assume(out_frames > 0 and out_height > 0 and out_width > 0)
    
    return {
        'batch_size': batch_size,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'in_frames': in_frames,
        'in_height': in_height,
        'in_width': in_width,
        'kernel_t': kernel_t,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'stride_t': stride_t,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'pad_t': pad_t,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'out_frames': out_frames,
        'out_height': out_height,
        'out_width': out_width
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.cuda
@pytest.mark.slow
class TestTemporalConvProperties:
    """Property-based tests for temporal 3D convolution kernel."""
    
    @given(config=conv3d_config())
    @settings(max_examples=100, deadline=None)
    def test_numerical_equivalence_basic_kernel(self, config):
        """
        Property 2: Numerical Equivalence of Custom Kernels (Temporal Conv)
        Validates: Requirements 4.5
        
        Test that basic temporal conv3d kernel produces numerically equivalent
        results to PyTorch conv3d across random input configurations.
        """
        # Generate random inputs
        input_tensor = torch.randn(
            config['batch_size'],
            config['in_channels'],
            config['in_frames'],
            config['in_height'],
            config['in_width'],
            device='cuda',
            dtype=torch.float32
        )
        
        weight = torch.randn(
            config['out_channels'],
            config['in_channels'],
            config['kernel_t'],
            config['kernel_h'],
            config['kernel_w'],
            device='cuda',
            dtype=torch.float32
        )
        
        bias = torch.randn(config['out_channels'], device='cuda', dtype=torch.float32)
        
        # Compute baseline output using PyTorch
        baseline_output = F.conv3d(
            input_tensor,
            weight,
            bias,
            stride=(config['stride_t'], config['stride_h'], config['stride_w']),
            padding=(config['pad_t'], config['pad_h'], config['pad_w'])
        )
        
        # Compute custom kernel output (basic version)
        custom_output = video_diffusion_cuda_ops.temporal_conv3d(
            input_tensor,
            weight,
            bias,
            stride=[config['stride_t'], config['stride_h'], config['stride_w']],
            padding=[config['pad_t'], config['pad_h'], config['pad_w']],
            use_optimized=False
        )
        
        # Verify numerical equivalence
        assert custom_output.shape == baseline_output.shape, \
            f"Shape mismatch: custom {custom_output.shape} vs baseline {baseline_output.shape}"
        
        assert torch.allclose(custom_output, baseline_output, rtol=1e-4, atol=1e-4), \
            f"Numerical mismatch: max diff = {(custom_output - baseline_output).abs().max().item()}"
    
    @given(config=conv3d_config())
    @settings(max_examples=100, deadline=None)
    def test_numerical_equivalence_optimized_kernel(self, config):
        """
        Property 2: Numerical Equivalence of Custom Kernels (Temporal Conv - Optimized)
        Validates: Requirements 4.5
        
        Test that optimized temporal conv3d kernel with shared memory produces
        numerically equivalent results to PyTorch conv3d.
        """
        # Generate random inputs
        input_tensor = torch.randn(
            config['batch_size'],
            config['in_channels'],
            config['in_frames'],
            config['in_height'],
            config['in_width'],
            device='cuda',
            dtype=torch.float32
        )
        
        weight = torch.randn(
            config['out_channels'],
            config['in_channels'],
            config['kernel_t'],
            config['kernel_h'],
            config['kernel_w'],
            device='cuda',
            dtype=torch.float32
        )
        
        bias = torch.randn(config['out_channels'], device='cuda', dtype=torch.float32)
        
        # Compute baseline output using PyTorch
        baseline_output = F.conv3d(
            input_tensor,
            weight,
            bias,
            stride=(config['stride_t'], config['stride_h'], config['stride_w']),
            padding=(config['pad_t'], config['pad_h'], config['pad_w'])
        )
        
        # Compute custom kernel output (optimized version)
        custom_output = video_diffusion_cuda_ops.temporal_conv3d(
            input_tensor,
            weight,
            bias,
            stride=[config['stride_t'], config['stride_h'], config['stride_w']],
            padding=[config['pad_t'], config['pad_h'], config['pad_w']],
            use_optimized=True
        )
        
        # Verify numerical equivalence
        assert custom_output.shape == baseline_output.shape, \
            f"Shape mismatch: custom {custom_output.shape} vs baseline {baseline_output.shape}"
        
        assert torch.allclose(custom_output, baseline_output, rtol=1e-4, atol=1e-4), \
            f"Numerical mismatch: max diff = {(custom_output - baseline_output).abs().max().item()}"
    
    @given(config=conv3d_config())
    @settings(max_examples=50, deadline=None)
    def test_no_bias_equivalence(self, config):
        """
        Test temporal conv3d without bias produces correct results.
        """
        # Generate random inputs
        input_tensor = torch.randn(
            config['batch_size'],
            config['in_channels'],
            config['in_frames'],
            config['in_height'],
            config['in_width'],
            device='cuda',
            dtype=torch.float32
        )
        
        weight = torch.randn(
            config['out_channels'],
            config['in_channels'],
            config['kernel_t'],
            config['kernel_h'],
            config['kernel_w'],
            device='cuda',
            dtype=torch.float32
        )
        
        # Compute baseline output using PyTorch (no bias)
        baseline_output = F.conv3d(
            input_tensor,
            weight,
            None,
            stride=(config['stride_t'], config['stride_h'], config['stride_w']),
            padding=(config['pad_t'], config['pad_h'], config['pad_w'])
        )
        
        # Compute custom kernel output (no bias)
        custom_output = video_diffusion_cuda_ops.temporal_conv3d(
            input_tensor,
            weight,
            None,
            stride=[config['stride_t'], config['stride_h'], config['stride_w']],
            padding=[config['pad_t'], config['pad_h'], config['pad_w']],
            use_optimized=True
        )
        
        # Verify numerical equivalence
        assert torch.allclose(custom_output, baseline_output, rtol=1e-4, atol=1e-4)
    
    @given(
        batch_size=st.integers(1, 4),
        channels=st.integers(8, 32),
        frames=st.integers(8, 24),
        spatial_size=st.integers(16, 64)
    )
    @settings(max_examples=50, deadline=None)
    def test_configuration_support(self, batch_size, channels, spatial_size, frames):
        """
        Property 4: Temporal Convolution Configuration Support
        Validates: Requirements 4.4
        
        Test various kernel sizes and strides to verify kernel executes correctly
        and produces correct output dimensions.
        """
        # Test different kernel configurations
        configs = [
            {'kernel': (3, 3, 3), 'stride': (1, 1, 1), 'padding': (1, 1, 1)},
            {'kernel': (3, 3, 3), 'stride': (2, 2, 2), 'padding': (1, 1, 1)},
            {'kernel': (1, 3, 3), 'stride': (1, 1, 1), 'padding': (0, 1, 1)},
            {'kernel': (3, 1, 1), 'stride': (1, 1, 1), 'padding': (1, 0, 0)},
        ]
        
        for cfg in configs:
            kt, kh, kw = cfg['kernel']
            st_t, st_h, st_w = cfg['stride']
            pt, ph, pw = cfg['padding']
            
            # Skip if kernel larger than input
            if kt > frames or kh > spatial_size or kw > spatial_size:
                continue
            
            # Calculate expected output dimensions
            out_frames = (frames + 2 * pt - kt) // st_t + 1
            out_height = (spatial_size + 2 * ph - kh) // st_h + 1
            out_width = (spatial_size + 2 * pw - kw) // st_w + 1
            
            if out_frames <= 0 or out_height <= 0 or out_width <= 0:
                continue
            
            # Generate inputs
            input_tensor = torch.randn(
                batch_size, channels, frames, spatial_size, spatial_size,
                device='cuda', dtype=torch.float32
            )
            
            weight = torch.randn(
                channels, channels, kt, kh, kw,
                device='cuda', dtype=torch.float32
            )
            
            # Run custom kernel
            output = video_diffusion_cuda_ops.temporal_conv3d(
                input_tensor,
                weight,
                None,
                stride=[st_t, st_h, st_w],
                padding=[pt, ph, pw],
                use_optimized=True
            )
            
            # Verify output dimensions
            expected_shape = (batch_size, channels, out_frames, out_height, out_width)
            assert output.shape == expected_shape, \
                f"Shape mismatch for config {cfg}: got {output.shape}, expected {expected_shape}"
            
            # Verify no NaN or Inf
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
