"""Property-based tests for performance profiler"""
import pytest
import torch
import sys
import os
from hypothesis import given, strategies as st, settings
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.profiler import PerformanceProfiler
from baseline.profiled_pipeline import ProfiledVideoDiffusion


@pytest.fixture(scope="module")
def profiled_model():
    """Fixture to load profiled model once"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU property tests")
    
    return ProfiledVideoDiffusion(
        model_name="stabilityai/stable-video-diffusion-img2vid-xt",
        device="cuda",
        dtype=torch.float16,
        enable_profiling=True,
    )


@pytest.fixture(scope="module")
def test_image():
    """Create a simple test image"""
    img_array = np.zeros((576, 1024, 3), dtype=np.uint8)
    for i in range(576):
        img_array[i, :, :] = int(255 * i / 576)
    
    return Image.fromarray(img_array)


@pytest.mark.property
@pytest.mark.cuda
class TestProfilerProperties:
    """Property-based tests for profiler functionality"""
    
    def test_profiler_context_manager(self):
        """Test that profiler context manager works correctly"""
        profiler = PerformanceProfiler(device="cuda")
        
        # Profile a simple operation
        with profiler.profile("test_op"):
            torch.cuda.synchronize()
            result = torch.randn(1000, 1000, device="cuda")
            torch.cuda.synchronize()
        
        # Verify profiling data was collected
        summary = profiler.get_summary("test_op")
        assert summary is not None
        assert "count" in summary
        assert summary["count"] == 1
        assert "mean_time_s" in summary
        assert summary["mean_time_s"] > 0
    
    def test_profiler_multiple_calls(self):
        """Test profiler with multiple calls to same operation"""
        profiler = PerformanceProfiler(device="cuda")
        
        # Profile same operation multiple times
        for i in range(5):
            with profiler.profile("repeated_op"):
                torch.cuda.synchronize()
                _ = torch.randn(1000, 1000, device="cuda")
                torch.cuda.synchronize()
        
        summary = profiler.get_summary("repeated_op")
        assert summary["count"] == 5
        assert summary["total_time_s"] > 0
        assert summary["mean_time_s"] > 0
    
    def test_profiler_memory_tracking(self):
        """Test that profiler tracks memory correctly"""
        profiler = PerformanceProfiler(device="cuda")
        
        # Profile operation that allocates memory
        with profiler.profile("memory_op"):
            large_tensor = torch.randn(10000, 10000, device="cuda")
        
        summary = profiler.get_summary("memory_op")
        assert "mean_allocated_mb" in summary
        # Should have allocated some memory
        assert summary["mean_allocated_mb"] != 0 or summary["peak_allocated_mb"] > 0
    
    def test_profiler_reset(self):
        """Test that profiler reset clears data"""
        profiler = PerformanceProfiler(device="cuda")
        
        # Collect some data
        with profiler.profile("test_op"):
            _ = torch.randn(100, 100, device="cuda")
        
        # Verify data exists
        assert len(profiler.timings) > 0
        
        # Reset
        profiler.reset()
        
        # Verify data is cleared
        assert len(profiler.timings) == 0
        assert len(profiler.memory_stats) == 0
    
    def test_profiler_enable_disable(self):
        """Test profiler enable/disable functionality"""
        profiler = PerformanceProfiler(device="cuda")
        
        # Disable profiling
        profiler.disable()
        
        # Profile operation (should be no-op)
        with profiler.profile("disabled_op"):
            _ = torch.randn(100, 100, device="cuda")
        
        # Should have no data
        assert "disabled_op" not in profiler.timings
        
        # Re-enable
        profiler.enable()
        
        # Profile operation (should work)
        with profiler.profile("enabled_op"):
            _ = torch.randn(100, 100, device="cuda")
        
        # Should have data
        assert "enabled_op" in profiler.timings


@pytest.mark.property
@pytest.mark.cuda
@pytest.mark.slow
class TestProfilingCompleteness:
    """Test profiling completeness in video generation pipeline"""
    
    def test_profiling_completeness(self, profiled_model, test_image):
        """
        Property 9: Profiling Completeness
        
        When profiling is enabled, the profiler SHALL capture timing, memory,
        and bandwidth metrics for all major operations and identify bottlenecks.
        
        Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
        """
        # Generate video with profiling
        result = profiled_model.generate(
            image=test_image,
            num_frames=14,  # Minimum for faster test
            num_inference_steps=10,  # Minimum for faster test
            seed=42,
            return_dict=True,
            profile=True,
        )
        
        # Verify profiling data was collected
        assert "profiling_summary" in result
        assert "bottlenecks" in result
        
        summary = result["profiling_summary"]
        bottlenecks = result["bottlenecks"]
        
        # Verify major operations were profiled
        expected_operations = [
            "total_generation",
            "image_encoding",
            "pipeline_call",
            "unet_forward",
            "vae_decode",
            "tensor_conversion",
        ]
        
        for op in expected_operations:
            assert op in summary, f"Operation '{op}' not profiled"
            
            # Verify timing metrics exist
            op_summary = summary[op]
            assert "count" in op_summary
            assert "total_time_s" in op_summary
            assert "mean_time_s" in op_summary
            assert "mean_time_ms" in op_summary
            
            # Verify timing values are reasonable
            assert op_summary["count"] > 0
            assert op_summary["total_time_s"] > 0
            assert op_summary["mean_time_s"] > 0
        
        # Verify memory metrics exist (for CUDA)
        for op in expected_operations:
            op_summary = summary[op]
            assert "mean_allocated_mb" in op_summary or "peak_allocated_mb" in op_summary
        
        # Verify bottlenecks were identified
        assert len(bottlenecks) > 0
        assert len(bottlenecks) <= 3  # Top 3
        
        # Verify bottleneck structure
        for bottleneck in bottlenecks:
            assert "operation" in bottleneck
            assert "total_time_s" in bottleneck
            assert "percentage" in bottleneck
            assert bottleneck["total_time_s"] > 0
            assert 0 <= bottleneck["percentage"] <= 100
        
        # Verify bottlenecks are sorted by time (descending)
        times = [b["total_time_s"] for b in bottlenecks]
        assert times == sorted(times, reverse=True)
    
    def test_profiling_report_generation(self, profiled_model, test_image):
        """Test that profiling report can be generated"""
        # Generate video
        profiled_model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            profile=True,
        )
        
        # Get report
        report = profiled_model.get_profiling_report()
        
        # Verify report is not empty
        assert len(report) > 0
        assert "PERFORMANCE PROFILING REPORT" in report
        assert "Total profiled time" in report
        assert "Top" in report and "operations" in report
    
    def test_bottleneck_identification(self, profiled_model, test_image):
        """Test that bottlenecks are correctly identified"""
        # Generate video
        profiled_model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            profile=True,
        )
        
        # Get bottlenecks
        bottlenecks = profiled_model.get_bottlenecks(top_n=3)
        
        # Verify structure
        assert len(bottlenecks) <= 3
        assert all("operation" in b for b in bottlenecks)
        assert all("total_time_s" in b for b in bottlenecks)
        assert all("percentage" in b for b in bottlenecks)
        
        # Verify percentages sum to <= 100
        total_percentage = sum(b["percentage"] for b in bottlenecks)
        assert total_percentage <= 100
    
    def test_profiling_export(self, profiled_model, test_image):
        """Test that profiling data can be exported"""
        # Generate video
        profiled_model.generate(
            image=test_image,
            num_frames=14,
            num_inference_steps=10,
            seed=42,
            profile=True,
        )
        
        # Export data
        export_data = profiled_model.export_profiling_data()
        
        # Verify export structure
        assert "device" in export_data
        assert "summary" in export_data
        assert "bottlenecks" in export_data
        
        # Verify data is serializable (can be converted to JSON)
        import json
        try:
            json.dumps(export_data)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Profiling data is not JSON serializable: {e}")
