# Video Diffusion CUDA Optimization

High-performance video generation pipeline with custom CUDA kernel optimizations for Stable Video Diffusion.

## Project Overview

This project implements custom CUDA kernels to optimize video diffusion models, achieving:
- 2-3x speedup on attention operations
- 30-40% overall latency reduction
- 8-12 FPS on NVIDIA T4 GPU (Google Colab free tier)

## Hardware Requirements

- **GPU**: NVIDIA T4 (16GB) - available free on Google Colab
- **CUDA**: 12.2+ (pre-installed on Colab)
- **Python**: 3.10+

## Project Structure

```
video_diffusion_cuda/
├── src/                    # Source code
│   ├── baseline/          # Baseline PyTorch implementation
│   ├── cuda_kernels/      # Custom CUDA kernels
│   ├── extensions/        # PyTorch C++ extensions
│   ├── optimized/         # Optimized pipeline
│   └── utils/             # Utilities and profiling
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── property/         # Property-based tests
│   └── integration/      # Integration tests
├── notebooks/            # Colab notebooks
├── docs/                 # Documentation
└── benchmarks/           # Benchmark scripts
```

## Quick Start (Google Colab)

See `notebooks/setup_colab.ipynb` for complete setup instructions.

## Development Status

🚧 Under active development
