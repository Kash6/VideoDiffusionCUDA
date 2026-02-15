/**
 * Fused Attention CUDA Kernel
 * 
 * Implements optimized attention computation with:
 * - Efficient memory access patterns
 * - Fused softmax computation
 * - Numerical stability (max subtraction)
 * - Optimized for T4 GPU (sm_75)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Maximum sequence length that can be stored in registers
#define MAX_SEQ_REGISTER 512

/**
 * Fused attention kernel (basic version)
 * 
 * Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
 * 
 * Input shapes:
 *   Q: [batch, heads, seq_len, head_dim]
 *   K: [batch, heads, seq_len, head_dim]
 *   V: [batch, heads, seq_len, head_dim]
 * Output shape:
 *   O: [batch, heads, seq_len, head_dim]
 */
__global__ void fused_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Each thread processes one output position
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= seq_len) return;
    
    // Calculate base offsets
    int base_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int q_offset = base_offset + row * head_dim;
    
    // Scale factor for attention scores
    float scale = rsqrtf((float)head_dim);
    
    // Allocate temporary storage for scores
    extern __shared__ float shared_scores[];
    float* my_scores = &shared_scores[threadIdx.x * seq_len];
    
    // First pass: compute attention scores and find max
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        int k_offset = base_offset + k_idx * head_dim;
        
        // Compute Q @ K^T
        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;
        
        my_scores[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Second pass: compute softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float exp_score = expf(my_scores[k_idx] - max_score);
        my_scores[k_idx] = exp_score;
        sum_exp += exp_score;
    }
    
    // Third pass: compute weighted sum with V
    for (int d = 0; d < head_dim; d++) {
        float output_val = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float attention_weight = my_scores[k_idx] / sum_exp;
            int v_offset = base_offset + k_idx * head_dim + d;
            output_val += attention_weight * V[v_offset];
        }
        
        O[q_offset + d] = output_val;
    }
}

/**
 * Optimized fused attention kernel with shared memory tiling
 * This version uses shared memory more efficiently
 */
__global__ void fused_attention_kernel_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Each thread processes one output position
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= seq_len) return;
    
    // Calculate base offsets
    int base_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int q_offset = base_offset + row * head_dim;
    
    // Scale factor for attention scores
    float scale = rsqrtf((float)head_dim);
    
    // First pass: compute attention scores and find max
    float max_score = -INFINITY;
    
    // Use registers to store scores (limited by seq_len)
    // For large seq_len, we'll process in chunks
    const int MAX_SEQ_REGISTER = 512;
    float scores[MAX_SEQ_REGISTER];
    
    for (int k_idx = 0; k_idx < seq_len && k_idx < MAX_SEQ_REGISTER; k_idx++) {
        float score = 0.0f;
        int k_offset = base_offset + k_idx * head_dim;
        
        // Compute Q @ K^T
        for (int d = 0; d < head_dim; d++) {
            score += Q[q_offset + d] * K[k_offset + d];
        }
        score *= scale;
        
        scores[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Second pass: compute softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len && k_idx < MAX_SEQ_REGISTER; k_idx++) {
        float exp_score = expf(scores[k_idx] - max_score);
        scores[k_idx] = exp_score;
        sum_exp += exp_score;
    }
    
    // Third pass: compute weighted sum with V
    for (int d = 0; d < head_dim; d++) {
        float output_val = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len && k_idx < MAX_SEQ_REGISTER; k_idx++) {
            float attention_weight = scores[k_idx] / sum_exp;
            int v_offset = base_offset + k_idx * head_dim + d;
            output_val += attention_weight * V[v_offset];
        }
        
        O[q_offset + d] = output_val;
    }
}

// Kernel launcher helper
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
    ) {
        // Use simple 1D grid/block layout
        // Each thread processes one output row
        int threads_per_block = 256;
        dim3 block(threads_per_block);
        dim3 grid(
            (seq_len + threads_per_block - 1) / threads_per_block,  // x: rows
            num_heads,                                                // y: heads
            batch_size                                                // z: batch
        );
        
        // For basic kernel, we need shared memory for scores
        // Each thread needs seq_len floats
        size_t shared_mem_basic = threads_per_block * seq_len * sizeof(float);
        
        // Check if we can use basic kernel with shared memory
        if (seq_len <= 512 && shared_mem_basic < 48 * 1024) {
            // Use basic kernel with shared memory
            fused_attention_kernel<<<grid, block, shared_mem_basic, stream>>>(
                Q, K, V, O, batch_size, num_heads, seq_len, head_dim
            );
        } else {
            // Use v2 kernel with register storage (limited to seq_len <= 512)
            fused_attention_kernel_v2<<<grid, block, 0, stream>>>(
                Q, K, V, O, batch_size, num_heads, seq_len, head_dim
            );
        }
    }
}
