"""Tests for the gmsh module."""

import gmsh


def test_initialize_gmsh():
    """Tests loading the gmsh package."""
    gmsh.initialize()
    gmsh.finalize()
