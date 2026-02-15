from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Determine CUDA architecture based on GPU
# T4 = sm_75, V100 = sm_70, A100 = sm_80
cuda_arch = os.environ.get('CUDA_ARCH', 'sm_75')  # Default to T4

setup(
    name='video_diffusion_cuda',
    version='0.1.0',
    description='Optimized Video Diffusion with Custom CUDA Kernels',
    author='Akash',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='video_diffusion_cuda_ops',
            sources=[
                'src/cuda_kernels/fused_attention.cu',
                'src/cuda_kernels/temporal_conv3d.cu',
                'src/cuda_kernels/fused_sampler.cu',
                'src/extensions/bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    f'-arch={cuda_arch}',
                    '--ptxas-options=-v',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
    ],
    python_requires='>=3.8',
)
