"""Tests for the Geometry class."""

import numpy as np
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
    check.almost_equal(geom_poly.calculate_centroid()[0], 0.5)
    check.almost_equal(geom_mp.calculate_centroid()[0], 0.5)
    check.almost_equal(geom_poly.calculate_centroid()[1], 0.5)
    check.almost_equal(geom_mp.calculate_centroid()[1], 0.5)


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


def test_zero_length_facet():
    """Tests that zero length facets correctly get removed."""
    poly = Polygon([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    geom = Geometry(poly)

    assert len(geom.facets) == 4


def test_remove_duplicate_facets():
    """Tests that duplicate facets are not added to list and curve loop is correct."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[1, 0], [2, 0], [2, 1], [1, 1]])
    mp = MultiPolygon([poly1, poly2])
    geom = Geometry(mp)

    # total number of facets should equal 8 - 1 (shared)
    assert len(geom.facets) == 7

    # check facet idx 2 is in both curve loops
    for c in geom.curve_loops:
        for f in c.facets:
            if f.idx == 2:
                break
        else:
            pytest.fail("Facet is not shared by curve loop.")


def test_overlapping_facets():
    """Tests for overlapping facets."""
    pass


def test_calculate_extents():
    """Tests the calculate extents method."""
    pts = np.array([[0, 1], [8, 1], [5, 7], [-3, -0.5]])
    poly = Polygon(pts.tolist())
    geom = Geometry(poly)
    extents = geom.calculate_extents()

    check.almost_equal(pts[:, 0].min(), extents[0])
    check.almost_equal(pts[:, 0].max(), extents[1])
    check.almost_equal(pts[:, 1].min(), extents[2])
    check.almost_equal(pts[:, 1].max(), extents[3])


def test_align_to():
    """Tests the align_to() method."""
    # two unit boxes next to each other
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    geom1 = Geometry(poly1)
    geom2 = geom1 + geom1.align_to(geom1, "right")

    check.almost_equal(geom2.calculate_area(), 2.0)
    check.almost_equal(geom2.calculate_extents()[0], 0.0)
    check.almost_equal(geom2.calculate_extents()[1], 2.0)
    check.almost_equal(geom2.calculate_extents()[2], 0.0)
    check.almost_equal(geom2.calculate_extents()[3], 1.0)
    check.almost_equal(geom2.calculate_centroid()[0], 1.0)
    check.almost_equal(geom2.calculate_centroid()[1], 0.5)

    # test align inner
    poly2 = Polygon([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    geom3 = Geometry(poly2).align_to(geom1, "right", True)
    geom4 = geom1 - geom3

    check.almost_equal(geom4.calculate_area(), 1.0 - 0.5**2)
    check.almost_equal(
        geom4.calculate_centroid()[0],
        (0.5 * 1 * 0.25 + 2 * (0.25 * 0.5) * 0.75) / (1.0 - 0.5**2),
    )
    check.almost_equal(geom4.calculate_centroid()[1], 0.5)

    # test align to point
    geom5 = geom1 + geom1.align_to((0, 1), "top")

    check.almost_equal(geom5.calculate_area(), 2.0)
    check.almost_equal(geom5.calculate_extents()[0], 0.0)
    check.almost_equal(geom5.calculate_extents()[1], 1.0)
    check.almost_equal(geom5.calculate_extents()[2], 0.0)
    check.almost_equal(geom5.calculate_extents()[3], 2.0)
    check.almost_equal(geom5.calculate_centroid()[0], 0.5)
    check.almost_equal(geom5.calculate_centroid()[1], 1.0)


def test_align_center():
    """Tests the align_center() method."""
    # align to center of other
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    geom1 = Geometry(poly1)
    geom2 = Geometry(poly2).align_center(geom1)
    geom = geom1 - geom2

    check.almost_equal(geom.calculate_area(), 1.0 - 0.5**2)
    check.almost_equal(geom.calculate_centroid()[0], 0.5)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)

    # align center of point
    geom2 = Geometry(poly2).align_center((0.5, 0.5))
    geom = geom1 - geom2

    check.almost_equal(geom.calculate_area(), 1.0 - 0.5**2)
    check.almost_equal(geom.calculate_centroid()[0], 0.5)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)

    # align center self
    geom = geom1.align_center()
    check.almost_equal(geom.calculate_area(), 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.0)
    check.almost_equal(geom.calculate_centroid()[1], 0.0)

    # test value error
    with pytest.raises(ValueError, match="align_to must be either a Geometry object"):
        geom = geom1.align_center(5)


def test_shift_geometry():
    """Tests the shift_geometry() method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    geom1 = Geometry(poly1)
    geom2 = geom1.shift_geometry(1.0, 1.0)

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], 1.0)
    check.almost_equal(geom2.calculate_extents()[1], 2.0)
    check.almost_equal(geom2.calculate_extents()[2], 1.0)
    check.almost_equal(geom2.calculate_extents()[3], 2.0)
    check.almost_equal(geom2.calculate_centroid()[0], 1.5)
    check.almost_equal(geom2.calculate_centroid()[1], 1.5)


def test_rotate_geometry():
    """Tests the rotate_geometry() method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    geom1 = Geometry(poly1)

    # rotate about origin
    geom2 = geom1.rotate_geometry(45, (0.0, 0.0))

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], -np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[1], np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[2], 0.0)
    check.almost_equal(geom2.calculate_extents()[3], np.sqrt(2.0))
    check.almost_equal(geom2.calculate_centroid()[0], 0.0)
    check.almost_equal(geom2.calculate_centroid()[1], np.sqrt(2.0) / 2.0)

    # rotate about origin with radians
    geom2 = geom1.rotate_geometry(np.pi / 4, (0.0, 0.0), True)

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], -np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[1], np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[2], 0.0)
    check.almost_equal(geom2.calculate_extents()[3], np.sqrt(2.0))
    check.almost_equal(geom2.calculate_centroid()[0], 0.0)
    check.almost_equal(geom2.calculate_centroid()[1], np.sqrt(2.0) / 2.0)

    # rotate about center
    geom2 = geom1.rotate_geometry(45)

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], 0.5 - np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[1], 0.5 + np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[2], 0.5 - np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_extents()[3], 0.5 + np.sqrt(2.0) / 2.0)
    check.almost_equal(geom2.calculate_centroid()[0], 0.5)
    check.almost_equal(geom2.calculate_centroid()[1], 0.5)


def test_mirror_geometry():
    """Tests the mirror_geometry() method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    geom1 = Geometry(poly1)

    # mirror x
    geom2 = geom1.mirror_geometry("x", (0.0, 0.0))

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], 0.0)
    check.almost_equal(geom2.calculate_extents()[1], 1.0)
    check.almost_equal(geom2.calculate_extents()[2], -1.0)
    check.almost_equal(geom2.calculate_extents()[3], 0.0)
    check.almost_equal(geom2.calculate_centroid()[0], 0.5)
    check.almost_equal(geom2.calculate_centroid()[1], -0.5)

    # mirror y
    geom2 = geom1.mirror_geometry("y", (0.0, 0.0))

    check.almost_equal(geom2.calculate_area(), 1.0)
    check.almost_equal(geom2.calculate_extents()[0], -1.0)
    check.almost_equal(geom2.calculate_extents()[1], 0.0)
    check.almost_equal(geom2.calculate_extents()[2], 0.0)
    check.almost_equal(geom2.calculate_extents()[3], 1.0)
    check.almost_equal(geom2.calculate_centroid()[0], -0.5)
    check.almost_equal(geom2.calculate_centroid()[1], 0.5)

    # value error
    with pytest.raises(ValueError, match="axis must be 'x' or 'y'"):
        geom1.mirror_geometry("a")


def test_union():
    """Tests the __or__ method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    geom1 = Geometry(poly1)
    geom2 = Geometry(poly2)
    geom = geom1 | geom2

    check.almost_equal(geom.calculate_area(), 2.0 - 0.5 * 0.5)
    check.almost_equal(geom.calculate_extents()[0], 0.0)
    check.almost_equal(geom.calculate_extents()[1], 1.5)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.5)
    check.almost_equal(geom.calculate_centroid()[0], 0.75)
    check.almost_equal(geom.calculate_centroid()[1], 0.75)


def test_sub():
    """Tests the __sub__ method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    geom1 = Geometry(poly1)
    geom2 = Geometry(poly2)
    geom = geom1 - geom2

    check.almost_equal(geom.calculate_area(), 1.0 - 0.5 * 0.5)
    check.almost_equal(geom.calculate_extents()[0], 0.0)
    check.almost_equal(geom.calculate_extents()[1], 1.0)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.5)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)

    # test value error
    geom2 = Geometry(poly1)

    with pytest.raises(ValueError, match="Cannot perform difference on these"):
        geom = geom1 - geom2


def test_add():
    """Tests the __add__ method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[1, 0], [2, 0], [2, 1], [1, 1]])
    geom1 = Geometry(poly1)
    geom2 = Geometry(poly2)
    geom = geom1 + geom2

    check.almost_equal(geom.calculate_area(), 2)
    check.almost_equal(geom.calculate_extents()[0], 0.0)
    check.almost_equal(geom.calculate_extents()[1], 2.0)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 1.0)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)

    # overlapping
    poly2 = Polygon([[0.5, 0], [1.5, 0], [1.5, 1], [0.5, 1]])
    geom2 = Geometry(poly2)
    geom = geom1 + geom2

    check.almost_equal(geom.calculate_area(), 2)
    check.almost_equal(geom.calculate_extents()[0], 0.0)
    check.almost_equal(geom.calculate_extents()[1], 1.5)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.75)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)


def test_and():
    """Tests the __and__ method."""
    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly2 = Polygon([[0.5, 0], [1.5, 0], [1.5, 1], [0.5, 1]])
    geom1 = Geometry(poly1)
    geom2 = Geometry(poly2)

    # overlapping example from __add__
    geom = geom1 & geom2
    check.almost_equal(geom.calculate_area(), 0.5)
    check.almost_equal(geom.calculate_extents()[0], 0.5)
    check.almost_equal(geom.calculate_extents()[1], 1.0)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.75)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)
