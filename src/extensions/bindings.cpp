/**
 * PyTorch C++ Extension Bindings for Custom CUDA Kernels
 * 
 * This file provides Python bindings for custom CUDA kernels using pybind11.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
extern "C" {
    void launch_fused_attention(
        const float* Q,
        const float* K,
        const float* V,
        float* O,
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        cudaStream_t stream
    );
}

/**
 * Fused Attention Forward Pass
 * 
 * Args:
 *   Q: Query tensor [batch, heads, seq_len, head_dim]
 *   K: Key tensor [batch, heads, seq_len, head_dim]
 *   V: Value tensor [batch, heads, seq_len, head_dim]
 * 
 * Returns:
 *   O: Output tensor [batch, heads, seq_len, head_dim]
 */
torch::Tensor fused_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (batch, heads, seq_len, head_dim)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D (batch, heads, seq_len, head_dim)");
    TORCH_CHECK(V.dim() == 4, "V must be 4D (batch, heads, seq_len, head_dim)");
    
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be float32");
    
    // Check shapes match
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");
    
    // Extract dimensions
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);
    
    // Allocate output tensor
    auto O = torch::empty_like(Q);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    launch_fused_attention(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        stream
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, 
                "CUDA kernel launch failed: ", cudaGetErrorString(error));
    
    return O;
}

/**
 * Fused Attention with optional mask support
 * 
 * Args:
 *   Q: Query tensor [batch, heads, seq_len, head_dim]
 *   K: Key tensor [batch, heads, seq_len, head_dim]
 *   V: Value tensor [batch, heads, seq_len, head_dim]
 *   mask: Optional attention mask [batch, heads, seq_len, seq_len] or None
 * 
 * Returns:
 *   O: Output tensor [batch, heads, seq_len, head_dim]
 */
torch::Tensor fused_attention_forward_with_mask(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    c10::optional<torch::Tensor> mask
) {
    if (mask.has_value()) {
        // TODO: Implement masked attention
        TORCH_CHECK(false, "Masked attention not yet implemented");
    }
    
    return fused_attention_forward(Q, K, V);
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_attention", 
          &fused_attention_forward, 
          "Fused attention forward pass (CUDA)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"));
    
    m.def("fused_attention_with_mask",
          &fused_attention_forward_with_mask,
          "Fused attention forward pass with optional mask (CUDA)",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("mask") = py::none());
}
