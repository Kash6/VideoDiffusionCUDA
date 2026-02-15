# Testing Guide

## Running Tests

### Prerequisites

Make sure you have installed dependencies in a virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd video_diffusion_cuda
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -m unit

# Property-based tests only
pytest tests/property/ -m property

# Integration tests only
pytest tests/integration/ -m integration

# Skip slow tests
pytest tests/ -m "not slow"

# Run only CUDA tests (requires GPU)
pytest tests/ -m cuda
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run Specific Test File

```bash
pytest tests/unit/test_baseline.py -v
```

### Run Specific Test Function

```bash
pytest tests/unit/test_baseline.py::TestBaselineVideoDiffusion::test_model_loading -v
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.property` - Property-based tests (Hypothesis)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (video generation, full pipeline)
- `@pytest.mark.cuda` - Tests requiring CUDA GPU

## Google Colab Testing

To run tests in Colab:

```python
# In Colab notebook
!cd video_diffusion_cuda && pytest tests/unit/test_baseline.py -v -m cuda
```

## Notes

- **GPU Required**: Most tests require CUDA GPU (T4 on Colab)
- **Slow Tests**: Video generation tests take 30s-2min each
- **Memory**: Tests use ~12-15GB GPU memory
- **Property Tests**: Use fewer examples (5-10) for speed
- **CI/CD**: Use `-m "not slow and not cuda"` for CPU-only CI

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Reduce `num_frames` in tests
- Reduce `num_inference_steps` in tests
- Use `decode_chunk_size=4` instead of 8
- Clear CUDA cache between tests (done automatically)

### Model Download Issues

If model download fails:
- Check internet connection
- Try again (HuggingFace can be slow)
- Use cached model if available

### CUDA Not Available

If CUDA tests are skipped:
- Verify GPU runtime in Colab (Runtime → Change runtime type → GPU)
- Check `torch.cuda.is_available()` returns True
- Verify NVIDIA driver with `!nvidia-smi`
