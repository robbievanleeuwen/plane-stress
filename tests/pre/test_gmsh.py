"""Tests for the gmsh module."""

import gmsh


def test_initialize_gmsh():
    """Tests gmsh."""
    gmsh.initialize()
    gmsh.finalize()
