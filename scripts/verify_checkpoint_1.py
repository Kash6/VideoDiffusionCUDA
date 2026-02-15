"""Checkpoint 1: Verify baseline implementation is working correctly"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from baseline.video_diffusion import BaselineVideoDiffusion
from baseline.profiled_pipeline import ProfiledVideoDiffusion
from utils.profiler import PerformanceProfiler


def check_cuda():
    """Check CUDA availability"""
    print("1. Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("   ✗ CUDA not available")
        print("   → Please run on Google Colab with GPU runtime")
        return False
    
    print(f"   ✓ CUDA available")
    print(f"   → GPU: {torch.cuda.get_device_name(0)}")
    print(f"   → Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


def check_baseline_model():
    """Check baseline model loading"""
    print("\n2. Checking baseline model loading...")
    try:
        model = BaselineVideoDiffusion(
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            device="cuda",
            dtype=torch.float16,
        )
        print("   ✓ Baseline model loaded successfully")
        
        # Check components
        assert model.vae is not None, "VAE not initialized"
        assert model.unet is not None, "UNet not initialized"
        assert model.scheduler is not None, "Scheduler not initialized"
        print("   ✓ All model components initialized")
        
        return model
    except Exception as e:
        print(f"   ✗ Failed to load baseline model: {e}")
        return None


def check_video_generation(model):
    """Check video generation works"""
    print("\n3. Checking video generation...")
    try:
        from PIL import Image
        import numpy as np
        
        # Create test image
        img_array = np.zeros((576, 1024, 3), dtype=np.uint8)
        for i in range(576):
            img_array[i, :, :] = int(255 * i / 576)
        test_image = Image.fromarray(img_array)
        
        # Generate short video
        print("   → Generating 14 frames (this may take 30-60s)...")
        video = model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            decode_chunk_size=2,  # Reduce memory usage
            seed=42,
            return_dict=False,
        )
        
        # Verify output
        assert video.shape == (14, 3, 576, 1024), f"Wrong shape: {video.shape}"
        assert not torch.isnan(video).any(), "Output contains NaN"
        assert video.min() >= 0 and video.max() <= 1, "Values out of range [0,1]"
        
        print("   ✓ Video generation successful")
        print(f"   → Output shape: {video.shape}")
        print(f"   → Value range: [{video.min():.3f}, {video.max():.3f}]")
        return True
    except Exception as e:
        print(f"   ✗ Video generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_profiler():
    """Check profiler functionality"""
    print("\n4. Checking profiler...")
    try:
        profiler = PerformanceProfiler(device="cuda")
        
        # Test profiling
        with profiler.profile("test_op"):
            _ = torch.randn(1000, 1000, device="cuda")
            torch.cuda.synchronize()
        
        summary = profiler.get_summary("test_op")
        assert summary is not None, "No profiling data"
        assert summary["count"] == 1, "Wrong call count"
        assert summary["mean_time_s"] > 0, "No timing data"
        
        print("   ✓ Profiler working correctly")
        return True
    except Exception as e:
        print(f"   ✗ Profiler check failed: {e}")
        return False


def check_profiled_pipeline():
    """Check profiled pipeline"""
    print("\n5. Checking profiled pipeline...")
    try:
        # Clear memory from previous tests
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        model = ProfiledVideoDiffusion(
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            device="cuda",
            dtype=torch.float16,
            enable_profiling=True,
        )
        print("   ✓ Profiled pipeline loaded")
        
        # Generate with profiling
        from PIL import Image
        import numpy as np
        
        img_array = np.zeros((576, 1024, 3), dtype=np.uint8)
        for i in range(576):
            img_array[i, :, :] = int(255 * i / 576)
        test_image = Image.fromarray(img_array)
        
        print("   → Generating with profiling (this may take 30-60s)...")
        result = model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            decode_chunk_size=2,  # Reduce from 8 to 2 to save memory
            seed=42,
            return_dict=True,
            profile=True,
        )
        
        # Verify profiling data
        assert "profiling_summary" in result, "No profiling summary"
        assert "bottlenecks" in result, "No bottlenecks identified"
        assert len(result["bottlenecks"]) > 0, "No bottlenecks found"
        
        print("   ✓ Profiled pipeline working")
        print(f"   → Bottlenecks identified: {len(result['bottlenecks'])}")
        print(f"   → Top bottleneck: {result['bottlenecks'][0]['operation']}")
        return True
    except Exception as e:
        print(f"   ✗ Profiled pipeline check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checkpoint verifications"""
    print("="*80)
    print("CHECKPOINT 1: BASELINE IMPLEMENTATION VERIFICATION")
    print("="*80)
    print()
    
    results = []
    
    # Run checks
    results.append(("CUDA", check_cuda()))
    
    if results[0][1]:  # Only continue if CUDA is available
        model = check_baseline_model()
        results.append(("Baseline Model", model is not None))
        
        if model:
            results.append(("Video Generation", check_video_generation(model)))
        else:
            results.append(("Video Generation", False))
        
        results.append(("Profiler", check_profiler()))
        results.append(("Profiled Pipeline", check_profiled_pipeline()))
    else:
        results.extend([
            ("Baseline Model", False),
            ("Video Generation", False),
            ("Profiler", False),
            ("Profiled Pipeline", False),
        ])
    
    # Summary
    print("\n" + "="*80)
    print("CHECKPOINT SUMMARY")
    print("="*80)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("="*80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues before proceeding")
        print("="*80)
        failed = [name for name, passed in results if not passed]
        print("\nFailed checks:")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    exit(main())
