"""Benchmark tests for a patch test."""

import pytest


@pytest.mark.benchmark_suite
@pytest.mark.benchmark_analysis
@pytest.mark.parametrize("el_type", ["Tri3", "Tri6", "Quad4", "Quad8", "Quad9"])
def test_patch_small(benchmark, unit_square, el_type):
    """Benchmark test for a unit square patch with a small number of elements."""
    ps = unit_square(0.1, el_type)

    benchmark(ps.solve)


@pytest.mark.benchmark_suite
@pytest.mark.benchmark_analysis
@pytest.mark.parametrize("el_type", ["Tri3", "Tri6", "Quad4", "Quad8", "Quad9"])
def test_patch_large(benchmark, unit_square, el_type):
    """Benchmark test for a unit square patch with a large number of elements."""
    ps = unit_square(0.03, el_type)

    benchmark(ps.solve)
