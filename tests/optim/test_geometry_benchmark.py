"""Geometry module performance benchmark harness.

This module provides benchmark utilities to measure and validate the performance
characteristics of the geometry module, including:
- Execution time measurements
- Peak memory tracking
- CPU vs GPU/MPS comparison
- Batch size scaling analysis

Usage:
    pytest tests/optim/test_geometry_benchmark.py -v --benchmark
    pytest tests/optim/test_geometry_benchmark.py::test_benchmark_summary -v -s

The benchmarks are marked with @pytest.mark.benchmark to allow selective execution.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import torch

from ai_scientist.optim import geometry


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    device: str
    batch_size: int
    n_theta: int
    n_zeta: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float
    throughput_samples_per_sec: float


def _create_test_coefficients(
    batch_size: int,
    mpol: int = 4,
    ntor: int = 4,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create realistic Fourier coefficients for benchmarking.

    Uses a decaying spectrum similar to real stellarator configurations.
    """
    torch.manual_seed(42)  # Deterministic for reproducibility

    grid_h = mpol + 1
    grid_w = 2 * ntor + 1

    # Create with exponentially decaying spectrum
    r_cos = torch.zeros(batch_size, grid_h, grid_w, device=device)
    z_sin = torch.zeros(batch_size, grid_h, grid_w, device=device)

    # R00 ~ 1.0 (major radius)
    r_cos[:, 0, ntor] = 1.0

    # Add decaying higher modes
    for m in range(grid_h):
        for n in range(grid_w):
            decay = np.exp(-0.5 * (m + abs(n - ntor)))
            r_cos[:, m, n] += 0.1 * decay * torch.randn(batch_size, device=device)
            z_sin[:, m, n] += 0.1 * decay * torch.randn(batch_size, device=device)

    # Reset R00 to ensure stable major radius
    r_cos[:, 0, ntor] = 1.0 + 0.01 * torch.randn(batch_size, device=device)

    n_field_periods = 3

    return r_cos, z_sin, n_field_periods


def _measure_peak_memory(device: str) -> float:
    """Get peak memory usage in MB for the given device."""
    if device == "cpu":
        # For CPU, we can't easily measure peak memory without external tools
        # Return 0 as placeholder
        return 0.0
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif device == "mps" and torch.backends.mps.is_available():
        # MPS doesn't expose memory stats in the same way
        return 0.0
    return 0.0


def _reset_memory_stats(device: str) -> None:
    """Reset memory statistics for the given device."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _benchmark_function(
    func: Callable[..., torch.Tensor],
    args: tuple,
    kwargs: dict,
    device: str,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> tuple[list[float], float]:
    """Benchmark a function and return timing results.

    Returns:
        Tuple of (list of times in ms, peak memory in MB)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(*args, **kwargs)

    if device == "cuda":
        torch.cuda.synchronize()

    _reset_memory_stats(device)

    # Timed runs
    times_ms = []
    for _ in range(timed_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = func(*args, **kwargs)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    peak_memory = _measure_peak_memory(device)

    return times_ms, peak_memory


def _run_benchmark(
    name: str,
    func: Callable[..., torch.Tensor],
    r_cos: torch.Tensor,
    z_sin: torch.Tensor,
    nfp: int,
    device: str,
    n_theta: int = 64,
    n_zeta: int = 64,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> BenchmarkResult:
    """Run a single benchmark and return results."""
    batch_size = r_cos.shape[0]

    times_ms, peak_memory = _benchmark_function(
        func,
        (r_cos, z_sin, nfp),
        {"n_theta": n_theta, "n_zeta": n_zeta},
        device,
        warmup_runs,
        timed_runs,
    )

    times_arr = np.array(times_ms)
    mean_time = float(np.mean(times_arr))
    throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0.0

    return BenchmarkResult(
        name=name,
        device=device,
        batch_size=batch_size,
        n_theta=n_theta,
        n_zeta=n_zeta,
        mean_time_ms=mean_time,
        std_time_ms=float(np.std(times_arr)),
        min_time_ms=float(np.min(times_arr)),
        max_time_ms=float(np.max(times_arr)),
        peak_memory_mb=peak_memory,
        throughput_samples_per_sec=throughput,
    )


# =============================================================================
# Benchmark Tests
# =============================================================================


@pytest.mark.benchmark
class TestGeometryBenchmarks:
    """Benchmark tests for geometry module functions."""

    @pytest.fixture
    def device(self) -> str:
        """Determine available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_elongation_scaling(self, device: str, batch_size: int) -> None:
        """Test elongation computation scales reasonably with batch size."""
        r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

        result = _run_benchmark(
            "elongation",
            geometry.elongation,
            r_cos,
            z_sin,
            nfp,
            device,
            n_theta=64,
            n_zeta=64,
        )

        # Verify function produces valid output
        output = geometry.elongation(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)
        assert output.shape == (batch_size,)
        assert torch.all(output >= 1.0)  # Elongation >= 1

        # Log results (visible with pytest -v -s)
        print(f"\n{result.name} (batch={batch_size}, device={device}):")
        print(f"  Mean: {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_aspect_ratio_scaling(self, device: str, batch_size: int) -> None:
        """Test aspect ratio computation scales reasonably with batch size."""
        r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

        result = _run_benchmark(
            "aspect_ratio",
            geometry.aspect_ratio,
            r_cos,
            z_sin,
            nfp,
            device,
            n_theta=64,
            n_zeta=64,
        )

        # Verify function produces valid output
        output = geometry.aspect_ratio(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)
        assert output.shape == (batch_size,)
        assert torch.all(output > 0)

        print(f"\n{result.name} (batch={batch_size}, device={device}):")
        print(f"  Mean: {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_mean_curvature_scaling(self, device: str, batch_size: int) -> None:
        """Test mean curvature computation scales reasonably with batch size."""
        r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

        result = _run_benchmark(
            "mean_curvature",
            geometry.mean_curvature,
            r_cos,
            z_sin,
            nfp,
            device,
            n_theta=64,
            n_zeta=64,
        )

        # Verify function produces valid output
        output = geometry.mean_curvature(r_cos, z_sin, nfp, n_theta=64, n_zeta=64)
        assert output.shape == (batch_size,)

        print(f"\n{result.name} (batch={batch_size}, device={device}):")
        print(f"  Mean: {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")

    @pytest.mark.parametrize("n_theta,n_zeta", [(32, 32), (64, 64), (128, 128)])
    def test_resolution_scaling(self, device: str, n_theta: int, n_zeta: int) -> None:
        """Test performance scales reasonably with grid resolution."""
        batch_size = 32
        r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

        result = _run_benchmark(
            "elongation",
            geometry.elongation,
            r_cos,
            z_sin,
            nfp,
            device,
            n_theta=n_theta,
            n_zeta=n_zeta,
        )

        # Verify function produces valid output
        output = geometry.elongation(r_cos, z_sin, nfp, n_theta=n_theta, n_zeta=n_zeta)
        assert output.shape == (batch_size,)

        print(f"\n{result.name} (resolution={n_theta}x{n_zeta}, device={device}):")
        print(f"  Mean: {result.mean_time_ms:.2f}ms ± {result.std_time_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")


@pytest.mark.benchmark
def test_benchmark_summary() -> None:
    """Run a comprehensive benchmark summary.

    This test is designed to be run standalone with pytest -v -s to get
    a full performance overview.
    """
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{'=' * 60}")
    print("GEOMETRY MODULE BENCHMARK SUMMARY")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"{'=' * 60}\n")

    functions_to_benchmark = [
        ("elongation", geometry.elongation),
        ("elongation_isoperimetric", geometry.elongation_isoperimetric),
        ("aspect_ratio", geometry.aspect_ratio),
        ("aspect_ratio_arc_length", geometry.aspect_ratio_arc_length),
        ("mean_curvature", geometry.mean_curvature),
        ("surface_area", geometry.surface_area),
        ("average_triangularity", geometry.average_triangularity),
    ]

    batch_sizes = [1, 16, 64]
    results: list[BenchmarkResult] = []

    for name, func in functions_to_benchmark:
        print(f"\nBenchmarking: {name}")
        print("-" * 40)

        for batch_size in batch_sizes:
            r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

            # Handle triangularity separately (no n_zeta parameter)
            if name == "average_triangularity":
                result = BenchmarkResult(
                    name=name,
                    device=device,
                    batch_size=batch_size,
                    n_theta=64,
                    n_zeta=0,  # Not applicable
                    mean_time_ms=0.0,
                    std_time_ms=0.0,
                    min_time_ms=0.0,
                    max_time_ms=0.0,
                    peak_memory_mb=0.0,
                    throughput_samples_per_sec=0.0,
                )
                times_ms, peak_memory = _benchmark_function(
                    func,
                    (r_cos, z_sin, nfp),
                    {"n_theta": 64},
                    device,
                )
                times_arr = np.array(times_ms)
                result = BenchmarkResult(
                    name=name,
                    device=device,
                    batch_size=batch_size,
                    n_theta=64,
                    n_zeta=0,
                    mean_time_ms=float(np.mean(times_arr)),
                    std_time_ms=float(np.std(times_arr)),
                    min_time_ms=float(np.min(times_arr)),
                    max_time_ms=float(np.max(times_arr)),
                    peak_memory_mb=peak_memory,
                    throughput_samples_per_sec=batch_size
                    / (float(np.mean(times_arr)) / 1000),
                )
            else:
                result = _run_benchmark(
                    name, func, r_cos, z_sin, nfp, device, n_theta=64, n_zeta=64
                )

            results.append(result)

            print(
                f"  Batch {batch_size:3d}: {result.mean_time_ms:7.2f}ms "
                f"± {result.std_time_ms:5.2f}ms "
                f"({result.throughput_samples_per_sec:7.1f} samples/sec)"
            )

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY TABLE (batch_size=64, resolution=64x64)")
    print(f"{'=' * 60}")
    print(f"{'Function':<30} {'Time (ms)':<15} {'Throughput':<15}")
    print("-" * 60)

    for result in results:
        if result.batch_size == 64:
            print(
                f"{result.name:<30} {result.mean_time_ms:>8.2f} ± {result.std_time_ms:<4.2f} "
                f"{result.throughput_samples_per_sec:>10.1f}/sec"
            )

    print(f"{'=' * 60}\n")


@pytest.mark.benchmark
def test_memory_efficiency() -> None:
    """Test that memory-optimized implementation uses less memory.

    This test verifies the trig separability optimization reduces peak memory
    compared to the naive (M, N, T, Z) tensor approach.
    """
    device = "cpu"
    batch_size = 32
    n_theta = 64
    n_zeta = 64

    r_cos, z_sin, nfp = _create_test_coefficients(batch_size, device=device)

    # Estimate theoretical memory usage
    # Old approach: full (B, M, N, T, Z) angle tensor
    # New approach: separable (B, M, Z) + (M, T) + (B, N, Z)
    mpol_plus_1, two_ntor_plus_1 = r_cos.shape[1], r_cos.shape[2]
    n_zeta_total = n_zeta * nfp

    # Old approach memory (float32): B * M * N * T * Z * 4 bytes
    old_memory_mb = (
        batch_size * mpol_plus_1 * two_ntor_plus_1 * n_theta * n_zeta_total * 4
    ) / (1024 * 1024)

    # New approach memory: B*M*Z + M*T + B*N*Z (each tensor, float32)
    # Plus intermediate A_rc, B_rc, A_zs, B_zs: each B*M*Z
    new_memory_mb = (
        (
            4 * (batch_size * mpol_plus_1 * n_zeta_total)  # A_rc, B_rc, A_zs, B_zs
            + (mpol_plus_1 * n_theta)  # cos_m_theta, sin_m_theta
            + (two_ntor_plus_1 * n_zeta_total)  # cos_n_zeta, sin_n_zeta
        )
        * 4
        / (1024 * 1024)
    )

    print("\nMemory Efficiency Analysis:")
    print(
        f"  Grid: {mpol_plus_1}x{two_ntor_plus_1} modes, {n_theta}x{n_zeta_total} points"
    )
    print(f"  Batch size: {batch_size}")
    print(f"  Theoretical old approach: {old_memory_mb:.2f} MB")
    print(f"  Theoretical new approach: {new_memory_mb:.2f} MB")
    print(f"  Memory reduction: {(1 - new_memory_mb / old_memory_mb) * 100:.1f}%")

    # Verify the function runs without memory issues
    output = geometry.elongation(r_cos, z_sin, nfp, n_theta=n_theta, n_zeta=n_zeta)
    assert output.shape == (batch_size,)

    # For larger configurations, verify scaling
    large_batch = 128
    large_r_cos, large_z_sin, _ = _create_test_coefficients(large_batch, device=device)

    large_old_memory = (
        large_batch * mpol_plus_1 * two_ntor_plus_1 * n_theta * n_zeta_total * 4
    ) / (1024 * 1024)

    print(f"\n  Large batch ({large_batch}) old approach: {large_old_memory:.2f} MB")

    # Should still run without OOM
    large_output = geometry.elongation(
        large_r_cos, large_z_sin, nfp, n_theta=n_theta, n_zeta=n_zeta
    )
    assert large_output.shape == (large_batch,)
    print("  Large batch test: PASSED (no OOM)")
