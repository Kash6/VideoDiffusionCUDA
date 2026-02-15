"""Performance profiling utilities for video diffusion pipeline"""
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Optional, Any
import torch


class PerformanceProfiler:
    """
    Performance profiler for measuring execution time and memory usage.
    
    Provides context manager for profiling code blocks with CUDA synchronization
    for accurate GPU timing.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize profiler.
        
        Args:
            device: Device to profile ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.enabled = True
        
        # CUDA events for more accurate GPU timing
        if self.device.type == "cuda":
            self.cuda_events: Dict[str, List[tuple]] = defaultdict(list)
    
    @contextmanager
    def profile(self, operation_name: str, use_cuda_events: bool = True):
        """
        Context manager for profiling a code block.
        
        Args:
            operation_name: Name of the operation being profiled
            use_cuda_events: Use CUDA events for GPU timing (more accurate)
            
        Example:
            with profiler.profile("attention"):
                output = attention_layer(input)
        """
        if not self.enabled:
            yield
            return
        
        # Record memory before
        mem_before = self._get_memory_stats()
        
        # Start timing
        if self.device.type == "cuda" and use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            yield
            
            end_event.record()
            torch.cuda.synchronize()
            
            # Get elapsed time in seconds
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            # CPU timing or fallback
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            start_time = time.perf_counter()
            
            yield
            
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
        
        # Record memory after
        mem_after = self._get_memory_stats()
        
        # Store results
        self.timings[operation_name].append(elapsed_time)
        self.call_counts[operation_name] += 1
        
        # Calculate memory delta
        mem_delta = {
            "allocated_mb": mem_after["allocated_mb"] - mem_before["allocated_mb"],
            "reserved_mb": mem_after["reserved_mb"] - mem_before["reserved_mb"],
            "peak_allocated_mb": mem_after["peak_allocated_mb"],
        }
        self.memory_stats[operation_name].append(mem_delta)
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device.type != "cuda":
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "peak_allocated_mb": 0.0,
            }
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1e6,
            "reserved_mb": torch.cuda.memory_reserved() / 1e6,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
    
    def get_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get profiling summary for an operation or all operations.
        
        Args:
            operation_name: Specific operation to summarize, or None for all
            
        Returns:
            Dictionary with timing and memory statistics
        """
        if operation_name:
            return self._summarize_operation(operation_name)
        
        # Summarize all operations
        summary = {}
        for op_name in self.timings.keys():
            summary[op_name] = self._summarize_operation(op_name)
        
        return summary
    
    def _summarize_operation(self, operation_name: str) -> Dict[str, Any]:
        """Summarize statistics for a single operation"""
        if operation_name not in self.timings:
            return {}
        
        times = self.timings[operation_name]
        mem_stats = self.memory_stats[operation_name]
        
        import numpy as np
        
        summary = {
            "count": len(times),
            "total_time_s": sum(times),
            "mean_time_s": np.mean(times),
            "std_time_s": np.std(times),
            "min_time_s": min(times),
            "max_time_s": max(times),
            "mean_time_ms": np.mean(times) * 1000,
        }
        
        # Add memory statistics
        if mem_stats:
            allocated = [m["allocated_mb"] for m in mem_stats]
            reserved = [m["reserved_mb"] for m in mem_stats]
            peak = [m["peak_allocated_mb"] for m in mem_stats]
            
            summary.update({
                "mean_allocated_mb": np.mean(allocated),
                "mean_reserved_mb": np.mean(reserved),
                "peak_allocated_mb": max(peak),
            })
        
        return summary
    
    def report(self, top_n: int = 10) -> str:
        """
        Generate a formatted profiling report.
        
        Args:
            top_n: Number of top operations to show by total time
            
        Returns:
            Formatted report string
        """
        summary = self.get_summary()
        
        if not summary:
            return "No profiling data collected."
        
        # Calculate total time across all operations
        total_time = sum(s["total_time_s"] for s in summary.values())
        
        # Sort by total time
        sorted_ops = sorted(
            summary.items(),
            key=lambda x: x[1]["total_time_s"],
            reverse=True
        )[:top_n]
        
        # Build report
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE PROFILING REPORT")
        lines.append("=" * 80)
        lines.append(f"Total profiled time: {total_time:.3f}s")
        lines.append(f"Number of operations: {len(summary)}")
        lines.append("")
        lines.append(f"Top {min(top_n, len(sorted_ops))} operations by total time:")
        lines.append("-" * 80)
        
        # Header
        lines.append(f"{'Operation':<30} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'%':>8}")
        lines.append("-" * 80)
        
        # Operations
        for op_name, stats in sorted_ops:
            percentage = (stats["total_time_s"] / total_time * 100) if total_time > 0 else 0
            lines.append(
                f"{op_name:<30} "
                f"{stats['count']:>8} "
                f"{stats['total_time_s']:>10.3f} "
                f"{stats['mean_time_ms']:>10.2f} "
                f"{percentage:>7.1f}%"
            )
        
        lines.append("-" * 80)
        
        # Memory statistics (if CUDA)
        if self.device.type == "cuda":
            lines.append("")
            lines.append("Memory Statistics:")
            lines.append("-" * 80)
            lines.append(f"{'Operation':<30} {'Allocated(MB)':>15} {'Peak(MB)':>15}")
            lines.append("-" * 80)
            
            for op_name, stats in sorted_ops:
                if "mean_allocated_mb" in stats:
                    lines.append(
                        f"{op_name:<30} "
                        f"{stats['mean_allocated_mb']:>15.2f} "
                        f"{stats['peak_allocated_mb']:>15.2f}"
                    )
            
            lines.append("-" * 80)
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def identify_bottlenecks(self, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Identify top N performance bottlenecks by execution time.
        
        Args:
            top_n: Number of bottlenecks to identify
            
        Returns:
            List of bottleneck information dictionaries
        """
        summary = self.get_summary()
        
        if not summary:
            return []
        
        # Calculate total time
        total_time = sum(s["total_time_s"] for s in summary.values())
        
        # Sort by total time
        sorted_ops = sorted(
            summary.items(),
            key=lambda x: x[1]["total_time_s"],
            reverse=True
        )[:top_n]
        
        bottlenecks = []
        for op_name, stats in sorted_ops:
            percentage = (stats["total_time_s"] / total_time * 100) if total_time > 0 else 0
            bottlenecks.append({
                "operation": op_name,
                "total_time_s": stats["total_time_s"],
                "mean_time_ms": stats["mean_time_ms"],
                "percentage": percentage,
                "count": stats["count"],
            })
        
        return bottlenecks
    
    def reset(self):
        """Reset all profiling data"""
        self.timings.clear()
        self.memory_stats.clear()
        self.call_counts.clear()
        if self.device.type == "cuda":
            self.cuda_events.clear()
            torch.cuda.reset_peak_memory_stats()
    
    def enable(self):
        """Enable profiling"""
        self.enabled = True
    
    def disable(self):
        """Disable profiling (profiling context managers become no-ops)"""
        self.enabled = False
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export profiling data to dictionary for serialization.
        
        Returns:
            Dictionary with all profiling data
        """
        return {
            "device": str(self.device),
            "summary": self.get_summary(),
            "bottlenecks": self.identify_bottlenecks(top_n=10),
        }
