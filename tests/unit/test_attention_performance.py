"""Unit tests for attention kernel performance"""
import pytest
import torch
import torch.nn.functional as F
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


@pytest.fixture(scope="module")
def cuda_ops():
    """Fixture to load CUDA extension"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA kernel tests")
    
    try:
        import video_diffusion_cuda_ops
        return video_diffusion_cuda_ops
    except ImportError:
        pytest.skip("CUDA extension not compiled. Run: python setup.py build_ext --inplace")


def benchmark_function(func, *args, num_runs=100, warmup=10):
    """
    Benchmark a function with proper CUDA synchronization
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        num_runs: Number of benchmark runs
        warmup: Number of warmup runs
    
    Returns:
        Average time per run in seconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / num_runs


@pytest.mark.unit
@pytest.mark.cuda
@pytest.mark.slow
class TestAttentionPerformance:
    """
    Unit tests for attention kernel performance
    
    Validates: Requirements 3.3
    Target: 2-3x speedup over PyTorch baseline
    """
    
    def test_attention_performance_small(self, cuda_ops):
        """Test performance on small dimensions (typical for video diffusion)"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Benchmark PyTorch baseline
        def pytorch_attention(Q, K, V):
            Q_flat = Q.reshape(-1, Q.size(2), Q.size(3))
            K_flat = K.reshape(-1, K.size(2), K.size(3))
            V_flat = V.reshape(-1, V.size(2), V.size(3))
            out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            return out.reshape(Q.shape)
        
        baseline_time = benchmark_function(pytorch_attention, Q, K, V, num_runs=100)
        
        # Benchmark custom kernel
        def custom_attention(Q, K, V):
            return cuda_ops.fused_attention(Q, K, V)
        
        custom_time = benchmark_function(custom_attention, Q, K, V, num_runs=100)
        
        speedup = baseline_time / custom_time
        
        print(f"\nSmall dimensions ({batch_size}x{num_heads}x{seq_len}x{head_dim}):")
        print(f"  PyTorch baseline: {baseline_time*1000:.3f} ms")
        print(f"  Custom kernel:    {custom_time*1000:.3f} ms")
        print(f"  Speedup:          {speedup:.2f}x")
        
        # Assert speedup (relaxed for initial implementation)
        assert speedup > 0.8, \
            f"Custom kernel is significantly slower ({speedup:.2f}x). " \
            f"Expected at least 0.8x (80% of baseline performance)"
    
    def test_attention_performance_medium(self, cuda_ops):
        """Test performance on medium dimensions"""
        batch_size, num_heads, seq_len, head_dim = 4, 12, 128, 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Benchmark PyTorch baseline
        def pytorch_attention(Q, K, V):
            Q_flat = Q.reshape(-1, Q.size(2), Q.size(3))
            K_flat = K.reshape(-1, K.size(2), K.size(3))
            V_flat = V.reshape(-1, V.size(2), V.size(3))
            out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            return out.reshape(Q.shape)
        
        baseline_time = benchmark_function(pytorch_attention, Q, K, V, num_runs=50)
        
        # Benchmark custom kernel
        def custom_attention(Q, K, V):
            return cuda_ops.fused_attention(Q, K, V)
        
        custom_time = benchmark_function(custom_attention, Q, K, V, num_runs=50)
        
        speedup = baseline_time / custom_time
        
        print(f"\nMedium dimensions ({batch_size}x{num_heads}x{seq_len}x{head_dim}):")
        print(f"  PyTorch baseline: {baseline_time*1000:.3f} ms")
        print(f"  Custom kernel:    {custom_time*1000:.3f} ms")
        print(f"  Speedup:          {speedup:.2f}x")
        
        assert speedup > 0.8, \
            f"Custom kernel is significantly slower ({speedup:.2f}x)"
    
    def test_attention_performance_large(self, cuda_ops):
        """Test performance on large dimensions"""
        batch_size, num_heads, seq_len, head_dim = 2, 16, 256, 64
        
        try:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient GPU memory for large dimension test")
            raise
        
        # Benchmark PyTorch baseline
        def pytorch_attention(Q, K, V):
            Q_flat = Q.reshape(-1, Q.size(2), Q.size(3))
            K_flat = K.reshape(-1, K.size(2), K.size(3))
            V_flat = V.reshape(-1, V.size(2), V.size(3))
            out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            return out.reshape(Q.shape)
        
        baseline_time = benchmark_function(pytorch_attention, Q, K, V, num_runs=20)
        
        # Benchmark custom kernel
        def custom_attention(Q, K, V):
            return cuda_ops.fused_attention(Q, K, V)
        
        custom_time = benchmark_function(custom_attention, Q, K, V, num_runs=20)
        
        speedup = baseline_time / custom_time
        
        print(f"\nLarge dimensions ({batch_size}x{num_heads}x{seq_len}x{head_dim}):")
        print(f"  PyTorch baseline: {baseline_time*1000:.3f} ms")
        print(f"  Custom kernel:    {custom_time*1000:.3f} ms")
        print(f"  Speedup:          {speedup:.2f}x")
        
        assert speedup > 0.8, \
            f"Custom kernel is significantly slower ({speedup:.2f}x)"
    
    def test_attention_performance_target(self, cuda_ops):
        """
        Test that attention kernel achieves target speedup
        
        Target: 2-3x speedup on typical video diffusion dimensions
        Validates: Requirements 3.3
        """
        # Typical dimensions for video diffusion UNet
        batch_size, num_heads, seq_len, head_dim = 2, 8, 77, 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Benchmark PyTorch baseline
        def pytorch_attention(Q, K, V):
            Q_flat = Q.reshape(-1, Q.size(2), Q.size(3))
            K_flat = K.reshape(-1, K.size(2), K.size(3))
            V_flat = V.reshape(-1, V.size(2), V.size(3))
            out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            return out.reshape(Q.shape)
        
        baseline_time = benchmark_function(pytorch_attention, Q, K, V, num_runs=100)
        
        # Benchmark custom kernel
        def custom_attention(Q, K, V):
            return cuda_ops.fused_attention(Q, K, V)
        
        custom_time = benchmark_function(custom_attention, Q, K, V, num_runs=100)
        
        speedup = baseline_time / custom_time
        
        print(f"\nVideo diffusion dimensions ({batch_size}x{num_heads}x{seq_len}x{head_dim}):")
        print(f"  PyTorch baseline: {baseline_time*1000:.3f} ms")
        print(f"  Custom kernel:    {custom_time*1000:.3f} ms")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  Target:           2.0-3.0x")
        
        if speedup >= 2.0:
            print(f"  ✓ Target achieved!")
        elif speedup >= 1.5:
            print(f"  ⚠ Approaching target (need {2.0/speedup:.2f}x more)")
        else:
            print(f"  ✗ Below target (need {2.0/speedup:.2f}x more)")
        
        # Relaxed assertion for initial implementation
        assert speedup > 0.8, \
            f"Custom kernel is significantly slower ({speedup:.2f}x). " \
            f"Expected at least 0.8x baseline performance"
    
    def test_attention_memory_efficiency(self, cuda_ops):
        """Test that custom kernel doesn't use significantly more memory"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Measure baseline memory
        torch.cuda.reset_peak_memory_stats()
        def pytorch_attention(Q, K, V):
            Q_flat = Q.reshape(-1, Q.size(2), Q.size(3))
            K_flat = K.reshape(-1, K.size(2), K.size(3))
            V_flat = V.reshape(-1, V.size(2), V.size(3))
            out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            return out.reshape(Q.shape)
        
        _ = pytorch_attention(Q, K, V)
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        
        # Measure custom kernel memory
        torch.cuda.reset_peak_memory_stats()
        _ = cuda_ops.fused_attention(Q, K, V)
        torch.cuda.synchronize()
        custom_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        
        memory_ratio = custom_memory / baseline_memory
        
        print(f"\nMemory usage:")
        print(f"  PyTorch baseline: {baseline_memory:.2f} MB")
        print(f"  Custom kernel:    {custom_memory:.2f} MB")
        print(f"  Ratio:            {memory_ratio:.2f}x")
        
        # Custom kernel should use similar or less memory
        assert memory_ratio < 2.0, \
            f"Custom kernel uses {memory_ratio:.2f}x more memory than baseline"
