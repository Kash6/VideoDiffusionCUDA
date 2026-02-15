"""Profiled video diffusion pipeline for performance analysis"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Optional, Union
import torch
from PIL import Image
from baseline.video_diffusion import BaselineVideoDiffusion
from utils.profiler import PerformanceProfiler


class ProfiledVideoDiffusion(BaselineVideoDiffusion):
    """
    Video diffusion pipeline with integrated performance profiling.
    
    Extends BaselineVideoDiffusion to add profiling points for:
    - Attention operations
    - Temporal convolutions  
    - Denoising sampler steps
    - VAE encode/decode
    - Overall pipeline execution
    """
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_profiling: bool = True,
    ):
        """
        Initialize profiled pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            dtype: Data type for model weights
            enable_profiling: Whether to enable profiling
        """
        super().__init__(model_name, device, dtype)
        
        self.profiler = PerformanceProfiler(device=device)
        if not enable_profiling:
            self.profiler.disable()
        
        # Wrap model components with profiling hooks
        self._wrap_unet_forward()
        self._wrap_vae_decode()
    
    def _wrap_unet_forward(self):
        """Wrap UNet forward pass to profile attention and convolutions"""
        original_forward = self.unet.forward
        
        def profiled_forward(*args, **kwargs):
            with self.profiler.profile("unet_forward"):
                return original_forward(*args, **kwargs)
        
        self.unet.forward = profiled_forward
    
    def _wrap_vae_decode(self):
        """Wrap VAE decode to profile decoding"""
        original_decode = self.vae.decode
        
        def profiled_decode(*args, **kwargs):
            with self.profiler.profile("vae_decode"):
                return original_decode(*args, **kwargs)
        
        self.vae.decode = profiled_decode
    
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
        profile: bool = True,
    ) -> Union[torch.Tensor, dict]:
        """
        Generate video with profiling.
        
        Args:
            Same as BaselineVideoDiffusion.generate()
            profile: Whether to profile this generation
            
        Returns:
            Video tensor or dict with video and profiling information
        """
        # Temporarily enable/disable profiling
        was_enabled = self.profiler.enabled
        if profile:
            self.profiler.enable()
        else:
            self.profiler.disable()
        
        # Reset profiler for this generation
        self.profiler.reset()
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Load and preprocess image
        if isinstance(image, str):
            from diffusers.utils import load_image
            image = load_image(image)
        
        image = image.resize((1024, 576))
        
        # Profile complete generation
        with self.profiler.profile("total_generation"):
            # Profile image encoding
            with self.profiler.profile("image_encoding"):
                # Use video_processor if available, fallback to image_processor
                processor = getattr(self.pipe, 'video_processor', getattr(self.pipe, 'image_processor', None))
                if processor:
                    image_tensor = processor.preprocess(
                        image, height=576, width=1024
                    )
                    image_tensor = image_tensor.to(self.device, dtype=self.dtype)
            
            # Profile the pipeline call (includes UNet, VAE, etc.)
            with self.profiler.profile("pipeline_call"):
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
            
            # Profile tensor conversion
            with self.profiler.profile("tensor_conversion"):
                import numpy as np
                video_tensor = torch.stack([
                    torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
                    for frame in frames
                ])
        
        # Restore profiler state
        if not was_enabled:
            self.profiler.disable()
        
        # Get profiling summary
        summary = self.profiler.get_summary()
        bottlenecks = self.profiler.identify_bottlenecks(top_n=3)
        
        # Print profiling report
        if profile:
            print("\n" + self.profiler.report(top_n=10))
            
            print("\nTop 3 Bottlenecks:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"{i}. {bottleneck['operation']}: "
                      f"{bottleneck['total_time_s']:.3f}s "
                      f"({bottleneck['percentage']:.1f}%)")
        
        if return_dict:
            total_time = summary.get("total_generation", {}).get("total_time_s", 0)
            return {
                "video": video_tensor,
                "frames": frames,
                "execution_time": total_time,
                "fps": num_frames / total_time if total_time > 0 else 0,
                "time_per_frame": total_time / num_frames if num_frames > 0 else 0,
                "profiling_summary": summary,
                "bottlenecks": bottlenecks,
            }
        
        return video_tensor
    
    def get_profiling_report(self) -> str:
        """Get formatted profiling report"""
        return self.profiler.report()
    
    def get_bottlenecks(self, top_n: int = 3) -> list:
        """Get top N performance bottlenecks"""
        return self.profiler.identify_bottlenecks(top_n=top_n)
    
    def reset_profiler(self):
        """Reset profiling data"""
        self.profiler.reset()
    
    def export_profiling_data(self) -> dict:
        """Export profiling data for analysis"""
        return self.profiler.export_to_dict()
