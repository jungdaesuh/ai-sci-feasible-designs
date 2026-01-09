"""Geometry Performance Benchmark Harness.

Run with:
    pytest tests/optim/benchmark_geometry.py -v --benchmark-only

Or standalone:
    python tests/optim/benchmark_geometry.py

This measures:
- Execution time for key geometry functions
- Memory allocation tracking
- Comparison of batch vs sequential processing
"""

import time
import tracemalloc
import torch
import pytest

from ai_scientist.optim import geometry


def _create_test_surfaces(batch_size: int, mpol: int = 5, ntor: int = 3):
    """Create random but valid stellarator surfaces for benchmarking."""
    r_cos = torch.randn(batch_size, mpol + 1, 2 * ntor + 1) * 0.1
    z_sin = torch.randn(batch_size, mpol + 1, 2 * ntor + 1) * 0.1
    # Set major radius (m=0, n=0) for stability
    r_cos[:, 0, ntor] = 10.0
    # Set minor radius (m=1, n=0)
    r_cos[:, 1, ntor] = 1.0
    z_sin[:, 1, ntor] = 1.0
    return r_cos, z_sin


# =============================================================================
# PYTEST-BENCHMARK TESTS (requires pytest-benchmark)
# Run: pip install pytest-benchmark && pytest tests/optim/benchmark_geometry.py -v
# =============================================================================


class TestGeometryBenchmarks:
    """Performance benchmarks for geometry module.

    These tests require pytest-benchmark:
        pip install pytest-benchmark

    Run with:
        pytest tests/optim/benchmark_geometry.py -v --benchmark-only
    """

    @pytest.mark.benchmark(group="aspect_ratio")
    def test_aspect_ratio_batch_100(self, benchmark):
        """Benchmark batch aspect ratio computation (100 samples)."""
        r_cos, z_sin = _create_test_surfaces(100)
        benchmark(lambda: geometry.aspect_ratio(r_cos, z_sin, 3))

    @pytest.mark.benchmark(group="elongation")
    def test_elongation_isoperimetric_batch_100(self, benchmark):
        """Benchmark isoperimetric elongation computation (100 samples)."""
        r_cos, z_sin = _create_test_surfaces(100)
        benchmark(lambda: geometry.elongation_isoperimetric(r_cos, z_sin, 3))

    @pytest.mark.benchmark(group="surface_area")
    def test_surface_area_batch_100(self, benchmark):
        """Benchmark surface area computation (100 samples)."""
        r_cos, z_sin = _create_test_surfaces(100)
        benchmark(lambda: geometry.surface_area(r_cos, z_sin, 3))

    @pytest.mark.benchmark(group="aspect_ratio")
    def test_aspect_ratio_batch_500(self, benchmark):
        """Benchmark batch aspect ratio computation (500 samples)."""
        r_cos, z_sin = _create_test_surfaces(500)
        benchmark(lambda: geometry.aspect_ratio(r_cos, z_sin, 3))


# =============================================================================
# MANUAL TIMING TESTS (no pytest-benchmark dependency)
# =============================================================================


class TestGeometryTimingManual:
    """Manual timing tests that don't require pytest-benchmark."""

    def test_memory_usage_batch_vs_sequential(self):
        """Verify batch processing memory usage patterns."""
        r_cos, z_sin = _create_test_surfaces(50)

        # Measure batch memory
        tracemalloc.start()
        _ = geometry.aspect_ratio(r_cos, z_sin, 3)
        batch_current, batch_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure sequential memory
        tracemalloc.start()
        for i in range(50):
            _ = geometry.aspect_ratio(r_cos[i : i + 1], z_sin[i : i + 1], 3)
        seq_current, seq_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n=== MEMORY USAGE COMPARISON ===")
        print(f"Batch (50 samples):      peak = {batch_peak / 1024:.1f} KB")
        print(f"Sequential (50 samples): peak = {seq_peak / 1024:.1f} KB")

        # The test passes if memory is tracked correctly (no assertion on values)
        assert batch_peak > 0
        assert seq_peak > 0

    def test_cpu_timing_report(self):
        """Generate timing report for documentation."""
        batch_sizes = [1, 10, 50, 100]
        functions = [
            ("aspect_ratio", geometry.aspect_ratio),
            ("elongation_isoperimetric", geometry.elongation_isoperimetric),
            ("surface_area", geometry.surface_area),
            ("mean_curvature", geometry.mean_curvature),
        ]

        results = []
        for batch_size in batch_sizes:
            r_cos, z_sin = _create_test_surfaces(batch_size)
            for name, func in functions:
                # Warmup
                func(r_cos, z_sin, 3)

                # Time (average of 5 runs)
                start = time.perf_counter()
                for _ in range(5):
                    func(r_cos, z_sin, 3)
                elapsed = (time.perf_counter() - start) / 5

                results.append(
                    {
                        "function": name,
                        "batch_size": batch_size,
                        "time_ms": elapsed * 1000,
                        "per_sample_us": (elapsed * 1e6) / batch_size,
                    }
                )

        print("\n=== GEOMETRY BENCHMARK RESULTS (CPU) ===")
        print(
            f"{'Function':<28} {'Batch':>6} {'Time (ms)':>10} {'Per Sample (μs)':>16}"
        )
        print("-" * 64)
        for r in results:
            print(
                f"{r['function']:<28} {r['batch_size']:>6} "
                f"{r['time_ms']:>10.2f} {r['per_sample_us']:>16.1f}"
            )

        # Just ensure it ran without errors
        assert len(results) == len(batch_sizes) * len(functions)

    def test_gradient_overhead(self):
        """Measure overhead of gradient computation."""
        r_cos, z_sin = _create_test_surfaces(10)

        # Forward only
        start = time.perf_counter()
        for _ in range(10):
            _ = geometry.aspect_ratio(r_cos, z_sin, 3)
        forward_time = (time.perf_counter() - start) / 10

        # Forward + backward
        r_cos_grad = r_cos.clone().requires_grad_(True)
        z_sin_grad = z_sin.clone().requires_grad_(True)
        start = time.perf_counter()
        for _ in range(10):
            r_cos_grad.grad = None
            z_sin_grad.grad = None
            ar = geometry.aspect_ratio(r_cos_grad, z_sin_grad, 3)
            ar.sum().backward()
        grad_time = (time.perf_counter() - start) / 10

        print("\n=== GRADIENT OVERHEAD ===")
        print(f"Forward only:       {forward_time * 1000:.2f} ms")
        print(f"Forward + backward: {grad_time * 1000:.2f} ms")
        print(f"Gradient overhead:  {((grad_time / forward_time) - 1) * 100:.1f}%")

        # Gradients should add some overhead
        assert grad_time > 0
        assert forward_time > 0


# =============================================================================
# STANDALONE ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 64)
    print(" GEOMETRY PERFORMANCE BENCHMARK")
    print("=" * 64)
    print()

    test = TestGeometryTimingManual()
    test.test_cpu_timing_report()
    test.test_memory_usage_batch_vs_sequential()
    test.test_gradient_overhead()

    print()
    print("=" * 64)
    print(" BENCHMARK COMPLETE")
    print("=" * 64)
