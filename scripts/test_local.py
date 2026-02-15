"""Local testing script (CPU-compatible checks)"""
import sys
import os
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_imports():
    """Check that all modules can be imported"""
    print("1. Checking module imports...")
    
    modules_to_check = [
        ('baseline.video_diffusion', 'BaselineVideoDiffusion'),
        ('baseline.profiled_pipeline', 'ProfiledVideoDiffusion'),
        ('utils.profiler', 'PerformanceProfiler'),
    ]
    
    all_passed = True
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ✗ {module_name}.{class_name}: {e}")
            all_passed = False
    
    return all_passed


def check_dependencies():
    """Check that required dependencies are available"""
    print("\n2. Checking dependencies...")
    
    dependencies = [
        'torch',
        'numpy',
        'PIL',
        'pytest',
        'hypothesis',
    ]
    
    all_passed = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✓ {dep}")
        except ImportError:
            print(f"   ✗ {dep} - NOT INSTALLED")
            all_passed = False
    
    return all_passed


def check_file_structure():
    """Check that all expected files exist"""
    print("\n3. Checking file structure...")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    expected_files = [
        'src/baseline/video_diffusion.py',
        'src/baseline/profiled_pipeline.py',
        'src/utils/profiler.py',
        'tests/unit/test_baseline.py',
        'tests/property/test_baseline_properties.py',
        'tests/property/test_profiler_properties.py',
        'examples/baseline_example.py',
        'examples/profile_baseline.py',
        'setup.py',
        'requirements.txt',
        'pytest.ini',
        'conftest.py',
    ]
    
    all_passed = True
    for file_path in expected_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} - MISSING")
            all_passed = False
    
    return all_passed


def check_profiler_basic():
    """Test basic profiler functionality (CPU)"""
    print("\n4. Testing profiler (CPU mode)...")
    
    try:
        import torch
        from utils.profiler import PerformanceProfiler
        
        profiler = PerformanceProfiler(device="cpu")
        
        # Test profiling
        with profiler.profile("test_op"):
            _ = torch.randn(100, 100)
        
        summary = profiler.get_summary("test_op")
        assert summary is not None, "No profiling data"
        assert summary["count"] == 1, "Wrong call count"
        assert summary["mean_time_s"] > 0, "No timing data"
        
        # Test report generation
        report = profiler.report()
        assert len(report) > 0, "Empty report"
        assert "PERFORMANCE PROFILING REPORT" in report
        
        # Test bottleneck identification
        bottlenecks = profiler.identify_bottlenecks(top_n=1)
        assert len(bottlenecks) == 1
        assert bottlenecks[0]["operation"] == "test_op"
        
        print("   ✓ Profiler basic functionality works")
        return True
    except Exception as e:
        print(f"   ✗ Profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_code_quality():
    """Check for basic code quality issues"""
    print("\n5. Checking code quality...")
    
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    
    issues = []
    
    # Check for syntax errors by trying to compile
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), file_path, 'exec')
                except SyntaxError as e:
                    issues.append(f"Syntax error in {file_path}: {e}")
    
    if issues:
        for issue in issues:
            print(f"   ✗ {issue}")
        return False
    else:
        print("   ✓ No syntax errors found")
        return True


def main():
    """Run all local tests"""
    print("="*80)
    print("LOCAL TESTING (CPU-compatible checks)")
    print("="*80)
    print()
    
    results = []
    
    results.append(("Dependencies", check_dependencies()))
    results.append(("Module Imports", check_imports()))
    results.append(("File Structure", check_file_structure()))
    results.append(("Profiler Basic", check_profiler_basic()))
    results.append(("Code Quality", check_code_quality()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*80)
    if all_passed:
        print("✓ ALL LOCAL TESTS PASSED")
        print("="*80)
        print("\nNext steps:")
        print("1. Upload code to Google Colab")
        print("2. Run full GPU tests with: python scripts/verify_checkpoint_1.py")
        print("3. Run pytest: pytest tests/ -m cuda")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*80)
        failed = [name for name, passed in results if not passed]
        print("\nFailed checks:")
        for name in failed:
            print(f"  - {name}")
        return 1


if __name__ == "__main__":
    exit(main())
