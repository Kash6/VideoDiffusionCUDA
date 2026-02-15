"""Example usage of baseline video diffusion pipeline"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baseline.video_diffusion import BaselineVideoDiffusion
import torch


def main():
    """Run baseline video generation example"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. This will be very slow on CPU.")
        print("  Please run on Google Colab with GPU runtime.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize baseline pipeline
    print("\n" + "="*60)
    print("Initializing Baseline Video Diffusion Pipeline")
    print("="*60)
    
    model = BaselineVideoDiffusion(
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Example: Generate video from a sample image
    # You can replace this with your own image URL or local path
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
    
    print("\n" + "="*60)
    print("Generating Video")
    print("="*60)
    print(f"Input image: {image_url}")
    print(f"Frames: 14 (reduced for T4 memory)")
    print(f"Inference steps: 10")
    
    # Generate video (reduced parameters for T4 GPU memory)
    result = model.generate(
        image=image_url,
        num_frames=14,
        num_inference_steps=10,
        decode_chunk_size=2,  # Reduce memory usage
        fps=6,
        seed=42,
        return_dict=True,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Generation Complete")
    print("="*60)
    print(f"Video shape: {result['video'].shape}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    print(f"FPS: {result['fps']:.2f}")
    print(f"Time per frame: {result['time_per_frame']*1000:.1f}ms")
    
    # Print memory stats
    print("\n" + "="*60)
    print("Memory Statistics")
    print("="*60)
    mem_stats = model.get_memory_stats()
    for key, value in mem_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Save video
    output_path = "output_baseline.mp4"
    model.save_video(result['video'], output_path, fps=6)
    
    print("\n" + "="*60)
    print("✓ Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
