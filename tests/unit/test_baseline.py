"""Unit tests for baseline video diffusion pipeline"""
import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from baseline.video_diffusion import BaselineVideoDiffusion


@pytest.mark.unit
@pytest.mark.cuda
class TestBaselineVideoDiffusion:
    """Test suite for baseline video diffusion implementation"""
    
    @pytest.fixture(scope="class")
    def model(self, device):
        """Fixture to load model once for all tests"""
        if device.type != "cuda":
            pytest.skip("CUDA not available, skipping GPU tests")
        
        return BaselineVideoDiffusion(
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            device=str(device),
            dtype=torch.float16,
        )
    
    def test_model_loading(self, model):
        """Test that model loads successfully with all components"""
        # Verify model is loaded
        assert model.pipe is not None, "Pipeline should be loaded"
        
        # Verify all components are initialized
        assert model.vae is not None, "VAE should be initialized"
        assert model.unet is not None, "UNet should be initialized"
        assert model.scheduler is not None, "Scheduler should be initialized"
        assert model.image_encoder is not None, "Image encoder should be initialized"
        
        # Verify device placement
        assert next(model.unet.parameters()).device.type == "cuda", "UNet should be on CUDA"
        assert next(model.vae.parameters()).device.type == "cuda", "VAE should be on CUDA"
        
        # Verify dtype
        assert next(model.unet.parameters()).dtype == torch.float16, "UNet should use float16"
    
    def test_model_info(self, model):
        """Test that model info is accessible"""
        mem_stats = model.get_memory_stats()
        
        assert "device" in mem_stats
        assert "allocated_gb" in mem_stats
        assert "reserved_gb" in mem_stats
        assert mem_stats["allocated_gb"] > 0, "Should have allocated GPU memory"
    
    def test_memory_reset(self, model):
        """Test memory statistics reset"""
        # Get initial stats
        initial_stats = model.get_memory_stats()
        
        # Reset
        model.reset_memory_stats()
        
        # Get new stats
        new_stats = model.get_memory_stats()
        
        # Max allocated should be reset
        assert new_stats["max_allocated_gb"] <= initial_stats["max_allocated_gb"]


@pytest.mark.unit
def test_model_loading_cpu():
    """Test that model can be loaded on CPU (for CI/CD without GPU)"""
    model = BaselineVideoDiffusion(
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        device="cpu",
        dtype=torch.float32,
    )
    
    assert model.pipe is not None
    assert model.device.type == "cpu"



@pytest.mark.unit
@pytest.mark.cuda
@pytest.mark.slow
class TestBaselineVideoGeneration:
    """Test suite for video generation functionality"""
    
    @pytest.fixture(scope="class")
    def model(self, device):
        """Fixture to load model once for all tests"""
        if device.type != "cuda":
            pytest.skip("CUDA not available, skipping GPU tests")
        
        return BaselineVideoDiffusion(
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            device=str(device),
            dtype=torch.float16,
        )
    
    @pytest.fixture(scope="class")
    def test_image(self):
        """Create a simple test image"""
        from PIL import Image
        import numpy as np
        
        # Create a simple gradient image
        img_array = np.zeros((576, 1024, 3), dtype=np.uint8)
        for i in range(576):
            img_array[i, :, :] = int(255 * i / 576)
        
        return Image.fromarray(img_array)
    
    def test_baseline_output_quality(self, model, test_image):
        """
        Test baseline output quality with fixed seed
        
        Validates: Requirements 1.5
        """
        # Generate video with fixed seed
        result = model.generate(
            image=test_image,
            num_frames=14,  # Minimum for faster test
            num_inference_steps=10,  # Minimum for faster test
            seed=42,
            return_dict=True,
        )
        
        video = result["video"]
        
        # Verify output validity
        assert video.shape == (14, 3, 576, 1024), \
            f"Expected shape (14, 3, 576, 1024), got {video.shape}"
        
        # Check no NaN values
        assert not torch.isnan(video).any(), "Output contains NaN values"
        
        # Check proper range [0, 1]
        assert video.min() >= 0.0, f"Min value {video.min()} < 0"
        assert video.max() <= 1.0, f"Max value {video.max()} > 1"
        
        # Check that video has variation (not all zeros or all ones)
        assert video.std() > 0.01, "Video has no variation (might be all zeros/ones)"
        
        # Check execution time is reasonable
        assert result["execution_time"] > 0, "Execution time should be positive"
        assert result["execution_time"] < 300, "Execution time too long (>5 min)"
        
        # Check FPS calculation
        expected_fps = 14 / result["execution_time"]
        assert abs(result["fps"] - expected_fps) < 0.01, "FPS calculation incorrect"
    
    def test_video_save(self, model, test_image, tmp_path):
        """Test video saving functionality"""
        # Generate a short video
        video = model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            return_dict=False,
        )
        
        # Save to temporary file
        output_path = tmp_path / "test_video.mp4"
        model.save_video(video, str(output_path), fps=6)
        
        # Verify file exists
        assert output_path.exists(), "Video file was not created"
        assert output_path.stat().st_size > 0, "Video file is empty"
    
    def test_execution_time_logging(self, model, test_image):
        """Test that execution time is logged correctly"""
        result = model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            return_dict=True,
        )
        
        # Verify timing information
        assert "execution_time" in result
        assert "fps" in result
        assert "time_per_frame" in result
        
        # Verify calculations
        assert result["execution_time"] > 0
        assert result["fps"] == 14 / result["execution_time"]
        assert result["time_per_frame"] == result["execution_time"] / 14
