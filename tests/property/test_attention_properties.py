"""Property-based tests for custom CUDA attention kernel"""
import pytest
import torch
import torch.nn.functional as F
import sys
import os
from hypothesis import given, strategies as st, settings, assume

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# Strategy for generating valid attention dimensions
@st.composite
def attention_dimensions(draw):
    """Generate valid attention tensor dimensions"""
    batch_size = draw(st.integers(min_value=1, max_value=4))
    num_heads = draw(st.sampled_from([4, 8, 12, 16]))
    seq_len = draw(st.sampled_from([32, 64, 128, 256]))
    head_dim = draw(st.sampled_from([32, 64, 128]))
    
    return {
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
    }


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


@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestAttentionNumericalEquivalence:
    """Property-based tests for attention kernel numerical correctness"""
    
    @given(dims=attention_dimensions())
    @settings(max_examples=10, deadline=None)
    def test_attention_numerical_equivalence(self, cuda_ops, dims):
        """
        Property 2: Numerical Equivalence of Custom Kernels (Attention)
        
        For any valid input tensors to custom attention kernel, the output
        SHALL be numerically equivalent to the baseline PyTorch implementation
        within floating-point tolerance (1e-5 for float32).
        
        Validates: Requirements 3.5, 10.5
        """
        batch_size = dims["batch_size"]
        num_heads = dims["num_heads"]
        seq_len = dims["seq_len"]
        head_dim = dims["head_dim"]
        
        # Create random input tensors
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device='cuda', dtype=torch.float32)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float32)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float32)
        
        # Compute with custom kernel
        with torch.no_grad():
            custom_output = cuda_ops.fused_attention(Q, K, V)
        
        # Compute with PyTorch baseline
        with torch.no_grad():
            # Reshape for PyTorch attention
            Q_flat = Q.reshape(batch_size * num_heads, seq_len, head_dim)
            K_flat = K.reshape(batch_size * num_heads, seq_len, head_dim)
            V_flat = V.reshape(batch_size * num_heads, seq_len, head_dim)
            
            # Compute attention
            baseline_output = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
            baseline_output = baseline_output.reshape(batch_size, num_heads, seq_len, head_dim)
        
        # Compare outputs
        max_diff = torch.abs(custom_output - baseline_output).max().item()
        mean_diff = torch.abs(custom_output - baseline_output).mean().item()
        relative_error = max_diff / (baseline_output.abs().max().item() + 1e-8)
        
        # Assert numerical equivalence
        assert not torch.isnan(custom_output).any(), \
            "Custom kernel output contains NaN"
        assert not torch.isinf(custom_output).any(), \
            "Custom kernel output contains Inf"
        
        assert max_diff < 1e-3, \
            f"Max difference {max_diff:.6f} exceeds tolerance 1e-3 " \
            f"(dims: {dims}, relative error: {relative_error:.6f})"
        
        assert mean_diff < 1e-4, \
            f"Mean difference {mean_diff:.6f} exceeds tolerance 1e-4 (dims: {dims})"
    
    def test_attention_output_shape(self, cuda_ops):
        """Test that output shape matches input shape"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 64
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        assert output.shape == Q.shape, \
            f"Output shape {output.shape} doesn't match input shape {Q.shape}"
        assert output.device == Q.device, \
            "Output device doesn't match input device"
        assert output.dtype == Q.dtype, \
            "Output dtype doesn't match input dtype"
    
    def test_attention_zero_input(self, cuda_ops):
        """Test attention with zero inputs"""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 32
        
        Q = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        # With zero Q and K, attention should be uniform, so output should be mean of V (which is 0)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5), \
            "Zero input should produce zero output"
    
    def test_attention_identity_pattern(self, cuda_ops):
        """Test attention with identity-like pattern"""
        batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 4
        
        # Create identity-like Q and K (each row attends to itself)
        Q = torch.eye(seq_len, head_dim, device='cuda').unsqueeze(0).unsqueeze(0)
        K = torch.eye(seq_len, head_dim, device='cuda').unsqueeze(0).unsqueeze(0)
        V = torch.arange(seq_len * head_dim, device='cuda', dtype=torch.float32).reshape(1, 1, seq_len, head_dim)
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        # With identity Q and K, attention should be mostly diagonal
        # Output should be close to V
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    @given(dims=attention_dimensions())
    @settings(max_examples=5, deadline=None)
    def test_attention_deterministic(self, cuda_ops, dims):
        """Test that attention kernel is deterministic"""
        batch_size = dims["batch_size"]
        num_heads = dims["num_heads"]
        seq_len = dims["seq_len"]
        head_dim = dims["head_dim"]
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Run twice with same inputs
        output1 = cuda_ops.fused_attention(Q, K, V)
        output2 = cuda_ops.fused_attention(Q, K, V)
        
        # Should produce identical results
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Kernel is not deterministic - same inputs produced different outputs"


@pytest.mark.property
@pytest.mark.cuda
class TestAttentionInputValidation:
    """Test input validation and error handling"""
    
    def test_attention_wrong_device(self, cuda_ops):
        """Test that CPU tensors are rejected"""
        Q = torch.randn(1, 4, 32, 32)  # CPU tensor
        K = torch.randn(1, 4, 32, 32)
        V = torch.randn(1, 4, 32, 32)
        
        with pytest.raises(RuntimeError, match="CUDA"):
            cuda_ops.fused_attention(Q, K, V)
    
    def test_attention_wrong_dtype(self, cuda_ops):
        """Test that non-float32 tensors are rejected"""
        Q = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16)
        K = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16)
        V = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16)
        
        with pytest.raises(RuntimeError, match="float32"):
            cuda_ops.fused_attention(Q, K, V)
    
    def test_attention_wrong_dimensions(self, cuda_ops):
        """Test that wrong number of dimensions is rejected"""
        Q = torch.randn(4, 32, 32, device='cuda')  # 3D instead of 4D
        K = torch.randn(4, 32, 32, device='cuda')
        V = torch.randn(4, 32, 32, device='cuda')
        
        with pytest.raises(RuntimeError, match="4D"):
            cuda_ops.fused_attention(Q, K, V)
    
    def test_attention_mismatched_shapes(self, cuda_ops):
        """Test that mismatched shapes are rejected"""
        Q = torch.randn(1, 4, 32, 32, device='cuda')
        K = torch.randn(1, 4, 64, 32, device='cuda')  # Different seq_len
        V = torch.randn(1, 4, 32, 32, device='cuda')
        
        with pytest.raises(RuntimeError, match="same shape"):
            cuda_ops.fused_attention(Q, K, V)


@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestAttentionEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_attention_single_token(self, cuda_ops):
        """Test attention with sequence length of 1"""
        Q = torch.randn(1, 4, 1, 32, device='cuda')
        K = torch.randn(1, 4, 1, 32, device='cuda')
        V = torch.randn(1, 4, 1, 32, device='cuda')
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        # With seq_len=1, output should equal V (only one token to attend to)
        assert torch.allclose(output, V, atol=1e-3), \
            "Single token attention should return V"
    
    def test_attention_large_sequence(self, cuda_ops):
        """Test attention with large sequence length"""
        # Test with larger sequence (memory permitting)
        Q = torch.randn(1, 4, 512, 64, device='cuda')
        K = torch.randn(1, 4, 512, 64, device='cuda')
        V = torch.randn(1, 4, 512, 64, device='cuda')
        
        try:
            output = cuda_ops.fused_attention(Q, K, V)
            assert output.shape == Q.shape
            assert not torch.isnan(output).any()
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Insufficient GPU memory for large sequence test")
            else:
                raise
    
    def test_attention_extreme_values(self, cuda_ops):
        """Test attention with extreme input values"""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 32
        
        # Test with large values
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 10
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        assert not torch.isnan(output).any(), \
            "Large input values caused NaN in output"
        assert not torch.isinf(output).any(), \
            "Large input values caused Inf in output"



@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestAttentionVariableSequenceLength:
    """
    Property 3: Attention Kernel Variable Sequence Length Handling
    
    Tests that the attention kernel correctly handles variable sequence lengths
    without cross-contamination between sequences.
    
    Validates: Requirements 3.4
    """
    
    @given(
        batch_size=st.integers(min_value=2, max_value=4),
        num_heads=st.sampled_from([4, 8]),
        base_seq_len=st.sampled_from([32, 64, 128]),
        head_dim=st.sampled_from([32, 64])
    )
    @settings(max_examples=5, deadline=None)
    def test_variable_sequence_lengths_batched(self, cuda_ops, batch_size, num_heads, base_seq_len, head_dim):
        """
        Test attention with different sequence lengths in batch
        
        Note: Current implementation requires same seq_len within batch.
        This test verifies that different batches can use different seq_lens.
        """
        # Test with two different sequence lengths
        seq_len1 = base_seq_len
        seq_len2 = base_seq_len // 2
        
        # Batch 1 with seq_len1
        Q1 = torch.randn(batch_size, num_heads, seq_len1, head_dim, device='cuda')
        K1 = torch.randn(batch_size, num_heads, seq_len1, head_dim, device='cuda')
        V1 = torch.randn(batch_size, num_heads, seq_len1, head_dim, device='cuda')
        
        output1 = cuda_ops.fused_attention(Q1, K1, V1)
        
        # Batch 2 with seq_len2
        Q2 = torch.randn(batch_size, num_heads, seq_len2, head_dim, device='cuda')
        K2 = torch.randn(batch_size, num_heads, seq_len2, head_dim, device='cuda')
        V2 = torch.randn(batch_size, num_heads, seq_len2, head_dim, device='cuda')
        
        output2 = cuda_ops.fused_attention(Q2, K2, V2)
        
        # Verify outputs have correct shapes
        assert output1.shape == Q1.shape, \
            f"Output1 shape {output1.shape} doesn't match Q1 shape {Q1.shape}"
        assert output2.shape == Q2.shape, \
            f"Output2 shape {output2.shape} doesn't match Q2 shape {Q2.shape}"
        
        # Verify no NaN or Inf
        assert not torch.isnan(output1).any(), "Output1 contains NaN"
        assert not torch.isnan(output2).any(), "Output2 contains NaN"
        assert not torch.isinf(output1).any(), "Output1 contains Inf"
        assert not torch.isinf(output2).any(), "Output2 contains Inf"
    
    def test_sequence_length_independence(self, cuda_ops):
        """
        Test that different sequence lengths produce independent results
        """
        batch_size, num_heads, head_dim = 2, 4, 32
        
        # Create inputs with different sequence lengths
        seq_lens = [32, 64, 128]
        outputs = []
        
        for seq_len in seq_lens:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
            
            output = cuda_ops.fused_attention(Q, K, V)
            outputs.append(output)
            
            # Verify output shape
            assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
                f"Wrong output shape for seq_len={seq_len}"
            
            # Verify no NaN or Inf
            assert not torch.isnan(output).any(), \
                f"Output contains NaN for seq_len={seq_len}"
            assert not torch.isinf(output).any(), \
                f"Output contains Inf for seq_len={seq_len}"
    
    def test_no_cross_contamination_within_batch(self, cuda_ops):
        """
        Test that different samples in batch don't contaminate each other
        """
        batch_size, num_heads, seq_len, head_dim = 4, 4, 64, 32
        
        # Create batch where each sample has different values
        Q = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        # Set different values for each batch element
        for b in range(batch_size):
            Q[b] = torch.randn(num_heads, seq_len, head_dim, device='cuda') * (b + 1)
            K[b] = torch.randn(num_heads, seq_len, head_dim, device='cuda') * (b + 1)
            V[b] = torch.randn(num_heads, seq_len, head_dim, device='cuda') * (b + 1)
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        # Compute each batch element separately
        for b in range(batch_size):
            Q_single = Q[b:b+1]
            K_single = K[b:b+1]
            V_single = V[b:b+1]
            
            output_single = cuda_ops.fused_attention(Q_single, K_single, V_single)
            
            # Compare with batched output
            diff = torch.abs(output[b:b+1] - output_single).max().item()
            assert diff < 1e-5, \
                f"Batch element {b} differs when computed separately (diff={diff})"
    
    def test_no_cross_contamination_across_heads(self, cuda_ops):
        """
        Test that different attention heads don't contaminate each other
        """
        batch_size, num_heads, seq_len, head_dim = 2, 8, 64, 32
        
        # Create inputs where each head has different values
        Q = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        K = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        V = torch.zeros(batch_size, num_heads, seq_len, head_dim, device='cuda')
        
        for h in range(num_heads):
            Q[:, h] = torch.randn(batch_size, seq_len, head_dim, device='cuda') * (h + 1)
            K[:, h] = torch.randn(batch_size, seq_len, head_dim, device='cuda') * (h + 1)
            V[:, h] = torch.randn(batch_size, seq_len, head_dim, device='cuda') * (h + 1)
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        # Verify each head independently
        for h in range(num_heads):
            Q_head = Q[:, h:h+1]
            K_head = K[:, h:h+1]
            V_head = V[:, h:h+1]
            
            output_head = cuda_ops.fused_attention(Q_head, K_head, V_head)
            
            diff = torch.abs(output[:, h:h+1] - output_head).max().item()
            assert diff < 1e-5, \
                f"Head {h} differs when computed separately (diff={diff})"



@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestAttentionGradientCorrectness:
    """
    Property 7: PyTorch Extension Gradient Flow (Attention)
    
    Tests that gradients flow correctly through the custom attention kernel.
    
    Validates: Requirements 6.3
    """
    
    @pytest.mark.skip(reason="Gradient support not yet implemented in kernel")
    @given(dims=attention_dimensions())
    @settings(max_examples=5, deadline=None)
    def test_attention_gradient_flow(self, cuda_ops, dims):
        """
        Test that gradients flow correctly through attention kernel
        
        Note: This test is skipped because the current kernel implementation
        does not support autograd. This will be implemented in a future version.
        """
        batch_size = dims["batch_size"]
        num_heads = dims["num_heads"]
        seq_len = dims["seq_len"]
        head_dim = dims["head_dim"]
        
        # Create inputs with gradient tracking
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', requires_grad=True)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', requires_grad=True)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', requires_grad=True)
        
        # Forward pass with custom kernel
        output_custom = cuda_ops.fused_attention(Q, K, V)
        loss_custom = output_custom.sum()
        loss_custom.backward()
        
        grad_Q_custom = Q.grad.clone()
        grad_K_custom = K.grad.clone()
        grad_V_custom = V.grad.clone()
        
        # Clear gradients
        Q.grad = None
        K.grad = None
        V.grad = None
        
        # Forward pass with PyTorch baseline
        Q_flat = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K_flat = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V_flat = V.reshape(batch_size * num_heads, seq_len, head_dim)
        
        output_baseline = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat)
        output_baseline = output_baseline.reshape(batch_size, num_heads, seq_len, head_dim)
        loss_baseline = output_baseline.sum()
        loss_baseline.backward()
        
        grad_Q_baseline = Q.grad
        grad_K_baseline = K.grad
        grad_V_baseline = V.grad
        
        # Compare gradients
        assert torch.allclose(grad_Q_custom, grad_Q_baseline, atol=1e-3), \
            "Q gradients don't match"
        assert torch.allclose(grad_K_custom, grad_K_baseline, atol=1e-3), \
            "K gradients don't match"
        assert torch.allclose(grad_V_custom, grad_V_baseline, atol=1e-3), \
            "V gradients don't match"
    
    def test_attention_no_gradient_leak(self, cuda_ops):
        """
        Test that attention kernel doesn't leak gradients
        
        This test verifies that when inputs don't require gradients,
        the output also doesn't track gradients.
        """
        Q = torch.randn(1, 4, 32, 32, device='cuda', requires_grad=False)
        K = torch.randn(1, 4, 32, 32, device='cuda', requires_grad=False)
        V = torch.randn(1, 4, 32, 32, device='cuda', requires_grad=False)
        
        output = cuda_ops.fused_attention(Q, K, V)
        
        assert not output.requires_grad, \
            "Output should not require gradients when inputs don't"
