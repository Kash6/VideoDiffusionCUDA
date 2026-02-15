/**
 * Fused Attention CUDA Kernel
 * 
 * Implements optimized attention computation with:
 * - Shared memory tiling for Q, K, V matrices
 * - Fused softmax computation
 * - Coalesced memory access patterns
 * - Optimized for T4 GPU (sm_75)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Tile size for shared memory (tuned for T4)
#define TILE_SIZE 64
#define WARP_SIZE 32

/**
 * Fused attention kernel
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
    // Block indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row_block = blockIdx.x;
    
    // Thread indices
    int tid = threadIdx.x;
    int row_in_block = tid / WARP_SIZE;
    int col_in_warp = tid % WARP_SIZE;
    
    // Calculate global row index
    int row = row_block * (blockDim.x / WARP_SIZE) + row_in_block;
    
    if (row >= seq_len) return;
    
    // Shared memory for tiles
    __shared__ float Q_tile[TILE_SIZE];
    __shared__ float K_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float V_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float scores[TILE_SIZE];
    
    // Calculate base offsets
    int qkv_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int q_row_offset = qkv_offset + row * head_dim;
    
    // Scale factor for attention scores
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Initialize output accumulator
    float output_acc = 0.0f;
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // First pass: compute max score for numerical stability
    for (int k_block = 0; k_block < (seq_len + TILE_SIZE - 1) / TILE_SIZE; k_block++) {
        int k_col = k_block * TILE_SIZE + col_in_warp;
        
        if (k_col < seq_len) {
            // Load Q and K tiles
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = Q[q_row_offset + d];
                float k_val = K[qkv_offset + k_col * head_dim + d];
                score += q_val * k_val;
            }
            score *= scale;
            
            // Update max score
            max_score = fmaxf(max_score, score);
            
            // Store score in shared memory
            if (col_in_warp < TILE_SIZE) {
                scores[k_block * WARP_SIZE + col_in_warp] = score;
            }
        }
    }
    
    // Reduce max across warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    }
    
    // Broadcast max to all threads in warp
    max_score = __shfl_sync(0xffffffff, max_score, 0);
    
    // Second pass: compute softmax and weighted sum
    for (int k_block = 0; k_block < (seq_len + TILE_SIZE - 1) / TILE_SIZE; k_block++) {
        int k_col = k_block * TILE_SIZE + col_in_warp;
        
        if (k_col < seq_len) {
            // Load score from shared memory
            float score = scores[k_block * WARP_SIZE + col_in_warp];
            
            // Compute exp(score - max) for numerical stability
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;
            
            // Load V tile and accumulate weighted sum
            for (int d = col_in_warp; d < head_dim; d += WARP_SIZE) {
                float v_val = V[qkv_offset + k_col * head_dim + d];
                atomicAdd(&output_acc, exp_score * v_val);
            }
        }
    }
    
    // Reduce sum_exp across warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    
    // Broadcast sum to all threads
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    
    // Normalize and write output
    if (sum_exp > 0.0f) {
        for (int d = col_in_warp; d < head_dim; d += WARP_SIZE) {
            O[q_row_offset + d] = output_acc / sum_exp;
        }
    }
}

/**
 * Optimized fused attention kernel with better memory coalescing
 * This version processes multiple rows per block for better occupancy
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
    // Shared memory for Q, K tiles
    extern __shared__ float shared_mem[];
    float* Q_shared = shared_mem;
    float* K_shared = shared_mem + TILE_SIZE * head_dim;
    float* scores_shared = K_shared + TILE_SIZE * head_dim;
    
    // Block and thread indices
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int row_block = blockIdx.x;
    int tid = threadIdx.x;
    
    // Calculate row index for this thread
    int row = row_block * TILE_SIZE + tid;
    if (row >= seq_len) return;
    
    // Base offset for this batch and head
    int base_offset = ((batch_idx * num_heads + head_idx) * seq_len) * head_dim;
    int q_offset = base_offset + row * head_dim;
    
    // Load Q row into shared memory
    for (int d = 0; d < head_dim; d++) {
        Q_shared[tid * head_dim + d] = Q[q_offset + d];
    }
    __syncthreads();
    
    // Scale factor
    float scale = rsqrtf((float)head_dim);
    
    // Compute attention scores and apply softmax
    float max_score = -INFINITY;
    
    // First pass: compute scores and find max
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = 0.0f;
        int k_offset = base_offset + k_idx * head_dim;
        
        // Compute dot product Q @ K^T
        for (int d = 0; d < head_dim; d++) {
            score += Q_shared[tid * head_dim + d] * K[k_offset + d];
        }
        score *= scale;
        
        scores_shared[tid * seq_len + k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Second pass: compute exp and sum
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float score = scores_shared[tid * seq_len + k_idx];
        float exp_score = expf(score - max_score);
        scores_shared[tid * seq_len + k_idx] = exp_score;
        sum_exp += exp_score;
    }
    
    // Third pass: compute weighted sum with V
    for (int d = 0; d < head_dim; d++) {
        float output_val = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float attention_weight = scores_shared[tid * seq_len + k_idx] / sum_exp;
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
        // Calculate grid and block dimensions
        dim3 block(256);  // 256 threads per block
        dim3 grid(
            (seq_len + TILE_SIZE - 1) / TILE_SIZE,  // x: row blocks
            num_heads,                               // y: heads
            batch_size                               // z: batch
        );
        
        // Calculate shared memory size
        size_t shared_mem_size = (2 * TILE_SIZE * head_dim + TILE_SIZE * seq_len) * sizeof(float);
        
        // Launch kernel
        if (shared_mem_size < 48 * 1024) {  // 48KB shared memory limit on T4
            fused_attention_kernel_v2<<<grid, block, shared_mem_size, stream>>>(
                Q, K, V, O, batch_size, num_heads, seq_len, head_dim
            );
        } else {
            // Fallback to simpler kernel if shared memory is insufficient
            fused_attention_kernel<<<grid, block, 0, stream>>>(
                Q, K, V, O, batch_size, num_heads, seq_len, head_dim
            );
        }
    }
}
