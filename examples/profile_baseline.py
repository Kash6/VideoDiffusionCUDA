"""Script to profile baseline video diffusion and document bottlenecks"""
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baseline.profiled_pipeline import ProfiledVideoDiffusion
import torch


def main():
    """Run baseline profiling and save results"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Profiling requires GPU.")
        print("  Please run on Google Colab with GPU runtime.")
        return
    
    print("="*80)
    print("BASELINE VIDEO DIFFUSION PROFILING")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Initialize profiled pipeline
    print("Loading model...")
    model = ProfiledVideoDiffusion(
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        device="cuda",
        dtype=torch.float16,
        enable_profiling=True,
    )
    print("✓ Model loaded\n")
    
    # Use sample image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    
    print("="*80)
    print("PROFILING RUN 1: Standard Configuration")
    print("="*80)
    print(f"Image: {image_url}")
    print(f"Frames: 25")
    print(f"Inference steps: 25")
    print()
    
    # Run profiling
    result1 = model.generate(
        image=image_url,
        num_frames=25,
        num_inference_steps=25,
        fps=6,
        seed=42,
        return_dict=True,
        profile=True,
    )
    
    print("\n" + "="*80)
    print("PROFILING RUN 2: Reduced Configuration (for comparison)")
    print("="*80)
    print(f"Frames: 14")
    print(f"Inference steps: 10")
    print()
    
    # Reset profiler
    model.reset_profiler()
    
    # Run with reduced config
    result2 = model.generate(
        image=image_url,
        num_frames=14,
        num_inference_steps=10,
        fps=6,
        seed=42,
        return_dict=True,
        profile=True,
    )
    
    # Document bottlenecks
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    print("\nRun 1 (25 frames, 25 steps):")
    print("-" * 40)
    for i, bottleneck in enumerate(result1["bottlenecks"], 1):
        print(f"{i}. {bottleneck['operation']}")
        print(f"   Time: {bottleneck['total_time_s']:.3f}s ({bottleneck['percentage']:.1f}%)")
        print(f"   Calls: {bottleneck['count']}")
        print(f"   Avg: {bottleneck['mean_time_ms']:.2f}ms per call")
    
    print("\nRun 2 (14 frames, 10 steps):")
    print("-" * 40)
    for i, bottleneck in enumerate(result2["bottlenecks"], 1):
        print(f"{i}. {bottleneck['operation']}")
        print(f"   Time: {bottleneck['total_time_s']:.3f}s ({bottleneck['percentage']:.1f}%)")
        print(f"   Calls: {bottleneck['count']}")
        print(f"   Avg: {bottleneck['mean_time_ms']:.2f}ms per call")
    
    # Save profiling results
    output_dir = "profiling_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results = {
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "run1": {
            "config": {
                "num_frames": 25,
                "num_inference_steps": 25,
            },
            "execution_time": result1["execution_time"],
            "fps": result1["fps"],
            "profiling_summary": result1["profiling_summary"],
            "bottlenecks": result1["bottlenecks"],
        },
        "run2": {
            "config": {
                "num_frames": 14,
                "num_inference_steps": 10,
            },
            "execution_time": result2["execution_time"],
            "fps": result2["fps"],
            "profiling_summary": result2["profiling_summary"],
            "bottlenecks": result2["bottlenecks"],
        },
    }
    
    output_file = os.path.join(output_dir, f"baseline_profiling_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"✓ Profiling results saved to: {output_file}")
    print("="*80)
    
    # Generate markdown report
    report_file = os.path.join(output_dir, f"baseline_profiling_{timestamp}.md")
    with open(report_file, 'w') as f:
        f.write("# Baseline Video Diffusion Profiling Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**GPU:** {torch.cuda.get_device_name(0)}\n\n")
        f.write(f"**CUDA Version:** {torch.version.cuda}\n\n")
        f.write(f"**PyTorch Version:** {torch.__version__}\n\n")
        
        f.write("## Run 1: Standard Configuration (25 frames, 25 steps)\n\n")
        f.write(f"- **Execution Time:** {result1['execution_time']:.2f}s\n")
        f.write(f"- **FPS:** {result1['fps']:.2f}\n")
        f.write(f"- **Time per frame:** {result1['time_per_frame']*1000:.1f}ms\n\n")
        
        f.write("### Top Bottlenecks\n\n")
        for i, b in enumerate(result1["bottlenecks"], 1):
            f.write(f"{i}. **{b['operation']}**: {b['total_time_s']:.3f}s ({b['percentage']:.1f}%)\n")
        
        f.write("\n## Run 2: Reduced Configuration (14 frames, 10 steps)\n\n")
        f.write(f"- **Execution Time:** {result2['execution_time']:.2f}s\n")
        f.write(f"- **FPS:** {result2['fps']:.2f}\n")
        f.write(f"- **Time per frame:** {result2['time_per_frame']*1000:.1f}ms\n\n")
        
        f.write("### Top Bottlenecks\n\n")
        for i, b in enumerate(result2["bottlenecks"], 1):
            f.write(f"{i}. **{b['operation']}**: {b['total_time_s']:.3f}s ({b['percentage']:.1f}%)\n")
        
        f.write("\n## Optimization Targets\n\n")
        f.write("Based on profiling, the following operations should be optimized:\n\n")
        
        # Combine bottlenecks from both runs
        all_bottlenecks = {}
        for b in result1["bottlenecks"] + result2["bottlenecks"]:
            op = b["operation"]
            if op not in all_bottlenecks:
                all_bottlenecks[op] = []
            all_bottlenecks[op].append(b["percentage"])
        
        # Sort by average percentage
        sorted_targets = sorted(
            all_bottlenecks.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )
        
        for i, (op, percentages) in enumerate(sorted_targets[:5], 1):
            avg_pct = sum(percentages) / len(percentages)
            f.write(f"{i}. **{op}** (avg {avg_pct:.1f}% of total time)\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Implement custom CUDA kernel for attention operations\n")
        f.write("2. Optimize temporal convolutions with 3D conv kernel\n")
        f.write("3. Fuse denoising sampler operations\n")
        f.write("4. Benchmark optimizations against this baseline\n")
    
    print(f"✓ Markdown report saved to: {report_file}")
    print()
    
    print("="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print(f"- Baseline FPS (25 frames): {result1['fps']:.2f}")
    print(f"- Baseline FPS (14 frames): {result2['fps']:.2f}")
    print(f"- Top bottleneck: {result1['bottlenecks'][0]['operation']} "
          f"({result1['bottlenecks'][0]['percentage']:.1f}%)")
    print("\nOptimization targets identified. Ready for CUDA kernel development!")


if __name__ == "__main__":
    main()
