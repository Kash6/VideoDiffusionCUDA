"""Baseline Video Diffusion Pipeline using PyTorch"""
import time
from typing import Optional, Union
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image


class BaselineVideoDiffusion:
    """
    Baseline video diffusion implementation using Stable Video Diffusion.
    
    This serves as the reference implementation for correctness and performance
    benchmarking. All operations use standard PyTorch without custom optimizations.
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize baseline video diffusion pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type for model weights (float16 for memory efficiency)
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device}, dtype: {dtype}")
        
        # Load the pipeline
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        self.pipe = self.pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            # Enable attention slicing to reduce memory usage
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            # Enable VAE slicing for lower memory usage (if available)
            if hasattr(self.pipe, 'enable_vae_slicing'):
                self.pipe.enable_vae_slicing()
        
        # Store model components for profiling
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler
        self.image_encoder = self.pipe.image_encoder
        
        print("✓ Model loaded successfully")
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information and memory usage"""
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Encode input image to latent space.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Image embeddings tensor
        """
        if isinstance(image, str):
            image = load_image(image)
        
        # Preprocess image - use video_processor if available
        processor = getattr(self.pipe, 'video_processor', getattr(self.pipe, 'image_processor', None))
        if processor:
            image = processor.preprocess(image, height=576, width=1024)
            image = image.to(self.device, dtype=self.dtype)
        
        # Encode with image encoder (CLIP)
        with torch.no_grad():
            image_embeddings = self.pipe._encode_image(
                image, 
                self.device, 
                num_videos_per_prompt=1,
                do_classifier_free_guidance=False
            )
        
        return image_embeddings
    
    def generate(
        self,
        image: Union[str, Image.Image],
        num_frames: int = 25,
        num_inference_steps: int = 25,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        seed: Optional[int] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Generate video from input image using baseline PyTorch operations.
        
        Args:
            image: Input image (PIL Image or path)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            fps: Frames per second for output video
            motion_bucket_id: Motion bucket for conditioning (higher = more motion)
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Chunk size for VAE decoding (lower = less memory)
            seed: Random seed for reproducibility
            return_dict: If True, return dict with timing info
            
        Returns:
            Generated video tensor of shape (num_frames, 3, height, width)
            or dict with video and timing information
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Load and preprocess image
        if isinstance(image, str):
            image = load_image(image)
        
        # Resize to standard resolution
        image = image.resize((1024, 576))
        
        # Track execution time
        start_time = time.perf_counter()
        
        # Generate video using the pipeline
        with torch.no_grad():
            frames = self.pipe(
                image=image,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
            ).frames[0]
        
        # Synchronize CUDA for accurate timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Convert frames to tensor
        import numpy as np
        video_tensor = torch.stack([
            torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ])
        
        print(f"✓ Generated {num_frames} frames in {execution_time:.2f}s")
        print(f"  FPS: {num_frames / execution_time:.2f}")
        print(f"  Time per frame: {execution_time / num_frames * 1000:.1f}ms")
        
        if return_dict:
            return {
                "video": video_tensor,
                "frames": frames,
                "execution_time": execution_time,
                "fps": num_frames / execution_time,
                "time_per_frame": execution_time / num_frames,
            }
        
        return video_tensor
    
    def save_video(
        self,
        video_tensor: torch.Tensor,
        output_path: str,
        fps: int = 6,
    ):
        """
        Save video tensor to file.
        
        Args:
            video_tensor: Video tensor of shape (num_frames, 3, height, width)
            output_path: Path to save video file
            fps: Frames per second
        """
        # Convert tensor to list of numpy arrays
        frames = [
            (frame.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            for frame in video_tensor
        ]
        
        # Export to video file
        export_to_video(frames, output_path, fps=fps)
        print(f"✓ Video saved to {output_path}")
    
    def get_memory_stats(self) -> dict:
        """
        Get current GPU memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.device.type != "cuda":
            return {"device": "cpu", "memory_gb": 0}
        
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    
    def reset_memory_stats(self):
        """Reset GPU memory statistics"""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
