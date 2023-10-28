"""Tests for the library module."""

import numpy as np
import pytest
import pytest_check as check

from planestress.pre.library import (
    circle,
    concrete_material,
    gravity,
    rectangle,
    steel_material,
)
from planestress.pre.material import DEFAULT_MATERIAL


def test_rectangle():
    """Tests the rectangle method."""
    geom = rectangle(1, 1)

    check.almost_equal(geom.calculate_area(), 1)
    check.almost_equal(geom.calculate_extents()[0], 0.0)
    check.almost_equal(geom.calculate_extents()[1], 1.0)
    check.almost_equal(geom.calculate_extents()[2], 0.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.5)
    check.almost_equal(geom.calculate_centroid()[1], 0.5)
    assert geom.materials[0] == DEFAULT_MATERIAL


def test_circle():
    """Tests the rectangle method."""
    geom = circle(1, 256)

    check.almost_equal(geom.calculate_area(), np.pi, rel=1e-3)  # discretisation
    check.almost_equal(geom.calculate_extents()[0], -1.0)
    check.almost_equal(geom.calculate_extents()[1], 1.0)
    check.almost_equal(geom.calculate_extents()[2], -1.0)
    check.almost_equal(geom.calculate_extents()[3], 1.0)
    check.almost_equal(geom.calculate_centroid()[0], 0.0)
    check.almost_equal(geom.calculate_centroid()[1], 0.0)
    assert geom.materials[0] == DEFAULT_MATERIAL


def test_gravity():
    """Tests the gravity method."""
    g_mpa1 = gravity()
    g_mpa2 = gravity(units="MPa")
    g_mpa3 = gravity(units="mpa")
    g_si1 = gravity(units="SI")
    g_si2 = gravity(units="Si")

    check.almost_equal(g_mpa1, 9.81e3)
    check.almost_equal(g_mpa2, 9.81e3)
    check.almost_equal(g_mpa3, 9.81e3)
    check.almost_equal(g_si1, 9.81)
    check.almost_equal(g_si2, 9.81)

    # test value error
    with pytest.raises(ValueError, match="is not a valid input for 'units'"):
        gravity("Pa")


def test_steel():
    """Tests the steel_material method."""
    steel_mpa1 = steel_material(16.0)
    steel_mpa2 = steel_material(16.0, "MPa")
    steel_mpa3 = steel_material(16.0, "mpa")
    steel_si1 = steel_material(0.016, "SI")
    steel_si2 = steel_material(0.016, "sI")

    assert steel_mpa1 == steel_mpa2
    assert steel_mpa2 == steel_mpa3
    assert steel_si1 == steel_si2
    check.almost_equal(steel_mpa1.elastic_modulus, 200e3)
    check.almost_equal(steel_mpa1.density, 7.85e-9)
    check.almost_equal(steel_mpa1.poissons_ratio, 0.3)
    check.almost_equal(steel_si1.elastic_modulus, 200e9)
    check.almost_equal(steel_si1.density, 7.85e3)
    check.almost_equal(steel_si1.poissons_ratio, 0.3)

    # test value error
    with pytest.raises(ValueError, match="is not a valid input for 'units'"):
        steel_material(16.0, "MP")


def test_concrete():
    """Tests the concrete_material method."""
    conc_mpa1 = concrete_material(30e3, 200.0)
    conc_mpa2 = concrete_material(30e3, 200.0, "MPa")
    conc_mpa3 = concrete_material(30e3, 200.0, "mpa")
    conc_si1 = concrete_material(30e9, 0.2, "SI")
    conc_si2 = concrete_material(30e9, 0.2, "sI")

    assert conc_mpa1 == conc_mpa2
    assert conc_mpa2 == conc_mpa3
    assert conc_si1 == conc_si2
    check.almost_equal(conc_mpa1.density, 2.4e-9)
    check.almost_equal(conc_mpa1.poissons_ratio, 0.2)
    check.almost_equal(conc_si1.density, 2.4e3)
    check.almost_equal(conc_si1.poissons_ratio, 0.2)

    # test value error
    with pytest.raises(ValueError, match="is not a valid input for 'units'"):
        concrete_material(30e3, 200.0, "MP")
