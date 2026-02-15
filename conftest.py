"""Pytest configuration and fixtures"""
import pytest
import torch
from hypothesis import settings, Verbosity

# Configure Hypothesis profiles
settings.register_profile("dev", max_examples=10, verbosity=Verbosity.verbose)
settings.register_profile("thorough", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=50, verbosity=Verbosity.normal)

# Load profile from environment or use default
settings.load_profile("thorough")


@pytest.fixture(scope="session")
def device():
    """Fixture providing CUDA device if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def gpu_info():
    """Fixture providing GPU information"""
    if torch.cuda.is_available():
        return {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "compute_capability": torch.cuda.get_device_capability(0),
        }
    return None


@pytest.fixture(autouse=True)
def reset_cuda_memory():
    """Reset CUDA memory between tests"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "cuda: Tests requiring CUDA GPU")
