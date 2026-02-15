"""Property-based tests for baseline video diffusion"""
import pytest
import torch
import sys
import os
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from baseline.video_diffusion import BaselineVideoDiffusion


# Strategy for valid video generation parameters
@st.composite
def video_generation_config(draw):
    """Generate valid video generation configurations"""
    # Num frames: reasonable range for testing
    num_frames = draw(st.integers(min_value=14, max_value=25))
    
    # Inference steps: lower for faster testing
    num_inference_steps = draw(st.integers(min_value=10, max_value=25))
    
    # FPS: standard video frame rates
    fps = draw(st.sampled_from([6, 12, 24]))
    
    # Motion bucket: valid range
    motion_bucket_id = draw(st.integers(min_value=50, max_value=200))
    
    # Noise augmentation: small values
    noise_aug_strength = draw(st.floats(min_value=0.0, max_value=0.1))
    
    return {
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "motion_bucket_id": motion_bucket_id,
        "noise_aug_strength": noise_aug_strength,
    }


@pytest.fixture(scope="module")
def baseline_model():
    """Fixture to load model once for all property tests"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU property tests")
    
    return BaselineVideoDiffusion(
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        device="cuda",
        dtype=torch.float16,
    )


@pytest.fixture(scope="module")
def test_image():
    """Create a simple test image"""
    # Create a simple gradient image
    img_array = np.zeros((576, 1024, 3), dtype=np.uint8)
    for i in range(576):
        img_array[i, :, :] = int(255 * i / 576)
    
    return Image.fromarray(img_array)


@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestVideoGenerationProperties:
    """Property-based tests for video generation"""
    
    @given(config=video_generation_config())
    @settings(max_examples=5, deadline=None)  # Reduced for speed, video generation is slow
    def test_output_shape_consistency(self, baseline_model, test_image, config):
        """
        Property 1: Video Generation Output Shape Consistency
        
        For any valid generation configuration, the generated video tensor
        shape SHALL match the requested dimensions exactly.
        
        Validates: Requirements 1.3
        """
        # Generate video with random configuration
        video_tensor = baseline_model.generate(
            image=test_image,
            num_frames=config["num_frames"],
            num_inference_steps=config["num_inference_steps"],
            fps=config["fps"],
            motion_bucket_id=config["motion_bucket_id"],
            noise_aug_strength=config["noise_aug_strength"],
            decode_chunk_size=8,
            seed=42,  # Fixed seed for reproducibility
            return_dict=False,
        )
        
        # Verify output shape
        assert video_tensor.shape[0] == config["num_frames"], \
            f"Expected {config['num_frames']} frames, got {video_tensor.shape[0]}"
        
        assert video_tensor.shape[1] == 3, \
            f"Expected 3 channels (RGB), got {video_tensor.shape[1]}"
        
        assert video_tensor.shape[2] == 576, \
            f"Expected height 576, got {video_tensor.shape[2]}"
        
        assert video_tensor.shape[3] == 1024, \
            f"Expected width 1024, got {video_tensor.shape[3]}"
        
        # Verify tensor is on CPU (after generation)
        assert video_tensor.device.type == "cpu", \
            "Output tensor should be on CPU"
        
        # Verify dtype
        assert video_tensor.dtype == torch.float32, \
            "Output tensor should be float32"
        
        # Verify value range [0, 1]
        assert video_tensor.min() >= 0.0, \
            f"Minimum value should be >= 0, got {video_tensor.min()}"
        assert video_tensor.max() <= 1.0, \
            f"Maximum value should be <= 1, got {video_tensor.max()}"
    
    def test_output_no_nan_or_inf(self, baseline_model, test_image):
        """
        Property: Generated video should not contain NaN or Inf values
        
        Validates: Requirements 1.5
        """
        video_tensor = baseline_model.generate(
            image=test_image,
            num_frames=14,  # Minimum for faster test
            num_inference_steps=10,  # Minimum for faster test
            seed=42,
            return_dict=False,
        )
        
        # Check for NaN
        assert not torch.isnan(video_tensor).any(), \
            "Output contains NaN values"
        
        # Check for Inf
        assert not torch.isinf(video_tensor).any(), \
            "Output contains Inf values"
    
    def test_deterministic_with_seed(self, baseline_model, test_image):
        """
        Property: Same seed should produce same output
        
        Validates: Requirements 1.5
        """
        # Generate twice with same seed
        video1 = baseline_model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            return_dict=False,
        )
        
        video2 = baseline_model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            return_dict=False,
        )
        
        # Should be very similar (allowing for minor floating point differences)
        diff = torch.abs(video1 - video2).max()
        assert diff < 1e-3, \
            f"Videos with same seed differ by {diff}, expected < 1e-3"
