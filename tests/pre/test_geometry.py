"""Tests for the Geometry class."""

import pytest_check as check
from shapely import Polygon, MultiPolygon

from planestress.pre.material import Material
from planestress.pre.geometry import Geometry


def test_geometry_simple():
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    mp = MultiPolygon([poly])
    geom_poly = Geometry(poly)
    geom_mp = Geometry(mp)

    check.almost_equal(geom_poly.calculate_area(), geom_mp.calculate_area())
    check.almost_equal(geom_poly.calculate_area(), 1.0)

