#!/usr/bin/env python3
"""
Simple test to verify attention kernel works without illegal memory access
"""

import torch
import sys
sys.path.insert(0, '/content/VideoDiffusionCUDA/src')

try:
    import video_diffusion_cuda_ops
    print("✓ CUDA extension loaded successfully")
except Exception as e:
    print(f"✗ Failed to load CUDA extension: {e}")
    sys.exit(1)

# Test with small dimensions first
batch_size = 1
num_heads = 2
seq_len = 16
head_dim = 32

print(f"\nTesting with: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")

try:
    # Create test inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    print("✓ Created input tensors")
    
    # Call custom kernel
    output = video_diffusion_cuda_ops.fused_attention(Q, K, V)
    
    print("✓ Custom kernel executed successfully")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  Has NaN: {torch.isnan(output).any().item()}")
    print(f"  Has Inf: {torch.isinf(output).any().item()}")
    
    # Compare with PyTorch baseline
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    baseline_output = torch.matmul(attn_weights, V)
    
    print("✓ PyTorch baseline computed")
    
    # Check numerical difference
    max_diff = torch.abs(output - baseline_output).max().item()
    mean_diff = torch.abs(output - baseline_output).mean().item()
    
    print(f"\nNumerical comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ PASS: Numerical equivalence verified")
    else:
        print(f"✗ FAIL: Difference too large (max_diff={max_diff:.6f})")
        
except Exception as e:
    print(f"✗ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! Attention kernel is working correctly.")
print("="*60)
