"""Tests for the Geometry class."""

import pytest
import pytest_check as check
from shapely import MultiPolygon, Polygon

from planestress.pre.geometry import Geometry
from planestress.pre.material import Material


def test_geometry_simple():
    """Tests creating a simple geometry.

    Ensures that a geometry created with a polygon equals that created with a
    MultiPolygon.
    """
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mp = MultiPolygon([poly])
    geom_poly = Geometry(poly)
    geom_mp = Geometry(mp)

    check.almost_equal(geom_poly.calculate_area(), geom_mp.calculate_area())
    check.almost_equal(geom_poly.calculate_area(), 1.0)


def test_geometry_single_hole():
    """Tests creating a geometry with a single hole.

    Multiple methods are tested:
    - Polygon with a hole
    - Geometry subtraction (single poly)
    - Multipolygon with a hole
    - Geometry subtraction (multiple poly)
    """
    # polygon with a hole
    shell_simple = [(0, 0), (1, 0), (1, 1), (0, 1)]
    hole_simple = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    poly_simple = Polygon(shell_simple, [hole_simple])
    geom = Geometry(poly_simple)

    check.almost_equal(geom.calculate_area(), 1.0 - 0.5**2)

    # geometry subtraction (single poly)
    geom_outer = Geometry(Polygon(shell_simple))
    geom_inner = Geometry(Polygon(hole_simple))
    geom = geom_outer - geom_inner

    check.almost_equal(geom.calculate_area(), 1.0 - 0.5**2)

    # multipolygon with hole
    shell_additional = [(1, 0), (2, 0), (2, 1), (1, 1)]
    mp = MultiPolygon([poly_simple, Polygon(shell_additional)])
    geom = Geometry(mp)

    check.almost_equal(geom.calculate_area(), 2.0 - 0.5**2)

    # geometry subtraction (multi poly)
    geom_outer_right = Geometry(Polygon(shell_additional))
    geom = geom_outer - geom_inner + geom_outer_right

    check.almost_equal(geom.calculate_area(), 2.0 - 0.5**2)


def test_geometry_multiple_holes():
    """Tests creating a geometry with a multiple holes.

    Multiple methods are tested:
    - Polygon with a two holes
    - Geometry subtraction (single poly, 2 holes)
    - Multipolygon with three holes
    - Geometry subtraction (multiple poly, 3 holes)
    """
    # polygon with two holes
    shell = [(0, 0), (1, 0), (1, 1), (0, 1)]
    hole1 = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    hole2 = [(0.5, 0.5), (0.6, 0.5), (0.6, 0.8), (0.5, 0.8)]
    poly = Polygon(shell, [hole1, hole2])
    geom = Geometry(poly)

    check.almost_equal(geom.calculate_area(), 1.0 - 0.2 * 0.2 - 0.1 * 0.3)

    # geometry subtraction (single poly, two holes)
    geom_outer = Geometry(Polygon(shell))
    geom_hole1 = Geometry(Polygon(hole1))
    geom_hole2 = Geometry(Polygon(hole2))
    geom = geom_outer - geom_hole1 - geom_hole2

    check.almost_equal(geom.calculate_area(), 1.0 - 0.2 * 0.2 - 0.1 * 0.3)

    # multipolygon with three holes
    shell2 = [(-1, 0), (0, 0), (0, 1), (-1, 1)]
    hole3 = [(-0.25, 0.25), (-0.75, 0.25), (-0.75, 0.75), (-0.25, 0.75)]
    mp = MultiPolygon([poly, Polygon(shell2, [hole3])])
    geom = Geometry(mp)

    check.almost_equal(geom.calculate_area(), 2.0 - 0.2 * 0.2 - 0.1 * 0.3 - 0.5**2)

    # geometry subtraction (multi poly, three holes)
    geom_left = Geometry(Polygon(shell2))
    geom_hole3 = Geometry(Polygon(hole3))
    geom = (geom_outer - geom_hole1 - geom_hole2) + (geom_left - geom_hole3)

    check.almost_equal(geom.calculate_area(), 2.0 - 0.2 * 0.2 - 0.1 * 0.3 - 0.5**2)


def test_geometry_material_length():
    """Tests that an error is raised if the materials length is not correct."""
    mat1 = Material(elastic_modulus=2.0)
    mat2 = Material(elastic_modulus=4.0)

    shell_left = [(0, 0), (1, 0), (1, 1), (0, 1)]
    shell_right = [(1, 0), (2, 0), (2, 1), (1, 1)]
    poly_left = Polygon(shell_left)
    poly_right = Polygon(shell_right)

    # these geometries should create fine
    Geometry(MultiPolygon([poly_left, poly_right]), [mat1, mat2])
    Geometry(MultiPolygon([poly_left, poly_right]), mat1)
    Geometry(Polygon(shell_left), mat1)
    Geometry(MultiPolygon([poly_left, poly_right]), mat1)

    # this should fail
    with pytest.raises(ValueError, match="must equal number of polygons:"):
        Geometry(MultiPolygon([poly_left, poly_right]), [mat1])
