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
    
    void launch_temporal_conv3d(
        const float* input,
        const float* weight,
        const float* bias,
        float* output,
        int batch_size,
        int in_channels,
        int out_channels,
        int in_frames, int in_height, int in_width,
        int out_frames, int out_height, int out_width,
        int kernel_t, int kernel_h, int kernel_w,
        int stride_t, int stride_h, int stride_w,
        int pad_t, int pad_h, int pad_w,
        bool use_optimized
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
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

/**
 * Helper function to calculate output dimensions for conv3d
 */
std::vector<int> calculate_conv3d_output_dims(
    int in_dim, int kernel_size, int stride, int padding
) {
    int out_dim = (in_dim + 2 * padding - kernel_size) / stride + 1;
    return {out_dim};
}

/**
 * Temporal 3D Convolution Forward Pass
 * 
 * Args:
 *   input: Input tensor [batch, in_channels, frames, height, width]
 *   weight: Weight tensor [out_channels, in_channels, kernel_t, kernel_h, kernel_w]
 *   bias: Optional bias tensor [out_channels] or None
 *   stride: Tuple of (stride_t, stride_h, stride_w)
 *   padding: Tuple of (pad_t, pad_h, pad_w)
 *   use_optimized: Whether to use optimized kernel with shared memory
 * 
 * Returns:
 *   output: Output tensor [batch, out_channels, frames', height', width']
 */
torch::Tensor temporal_conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    bool use_optimized
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    
    TORCH_CHECK(input.dim() == 5, 
                "input must be 5D (batch, in_channels, frames, height, width), got ", 
                input.dim(), "D");
    TORCH_CHECK(weight.dim() == 5, 
                "weight must be 5D (out_channels, in_channels, kernel_t, kernel_h, kernel_w), got ",
                weight.dim(), "D");
    
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.value().size(0) == weight.size(0), 
                    "bias size must match out_channels");
    }
    
    // Check stride and padding
    TORCH_CHECK(stride.size() == 3, "stride must have 3 elements (t, h, w)");
    TORCH_CHECK(padding.size() == 3, "padding must have 3 elements (t, h, w)");
    
    // Extract dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_frames = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_t = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    TORCH_CHECK(weight.size(1) == in_channels, 
                "weight in_channels must match input in_channels");
    
    int stride_t = stride[0];
    int stride_h = stride[1];
    int stride_w = stride[2];
    
    int pad_t = padding[0];
    int pad_h = padding[1];
    int pad_w = padding[2];
    
    // Calculate output dimensions
    int out_frames = (in_frames + 2 * pad_t - kernel_t) / stride_t + 1;
    int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    TORCH_CHECK(out_frames > 0 && out_height > 0 && out_width > 0,
                "Invalid output dimensions. Check kernel size, stride, and padding.");
    
    // Allocate output tensor
    auto output = torch::empty(
        {batch_size, out_channels, out_frames, out_height, out_width},
        input.options()
    );
    
    // Get bias pointer (nullptr if not provided)
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    // Launch kernel
    launch_temporal_conv3d(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_frames, in_height, in_width,
        out_frames, out_height, out_width,
        kernel_t, kernel_h, kernel_w,
        stride_t, stride_h, stride_w,
        pad_t, pad_h, pad_w,
        use_optimized
    );
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, 
                "CUDA kernel launch failed in temporal_conv3d: ", 
                cudaGetErrorString(error));
    
    return output;
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
    
    m.def("temporal_conv3d",
          &temporal_conv3d_forward,
          "Temporal 3D convolution forward pass (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = std::vector<int64_t>{1, 1, 1},
          py::arg("padding") = std::vector<int64_t>{0, 0, 0},
          py::arg("use_optimized") = true);
}
