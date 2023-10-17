"""Library of commonly used geometries."""

from __future__ import annotations

import numpy as np
from shapely import Polygon

from planestress.pre.geometry import Geometry
from planestress.pre.material import DEFAULT_MATERIAL, Material


def rectangle(
    d: float,
    b: float,
    material: Material = DEFAULT_MATERIAL,
    tol: int = 12,
) -> Geometry:
    """Creates a rectangular geometry."""
    shell = [(0, 0), (b, 0), (b, d), (0, d)]
    poly = Polygon(shell=shell)

    return Geometry(polygons=poly, materials=[material], tol=tol)


def circle(
    d: float,
    n: int,
    material: Material = DEFAULT_MATERIAL,
    tol: int = 12,
) -> Geometry:
    """Creates a circular geometry."""
    points = []

    # loop through each point on the circle
    for idx in range(n):
        # determine polar angle
        theta = idx * 2 * np.pi * 1.0 / n

        # calculate location of the point
        x = 0.5 * d * np.cos(theta)
        y = 0.5 * d * np.sin(theta)

        # append the current point to the points list
        points.append((x, y))

    poly = Polygon(shell=points)

    return Geometry(polygons=poly, materials=[material], tol=tol)


def gravity(
    units: str = "MPa",
) -> float:
    """Returns gravity with consistent units."""
    # convert units to lower case
    units = units.lower()

    gravity_units = {
        "mpa": 9.81e3,  # mm/s^2
        "si": 9.81,  # m/s^2
    }

    return gravity_units[units]


def steel(
    thickness: float,
    units: str = "MPa",
    colour: str = "grey",
) -> Material:
    """Creates a steel material object."""
    # convert units to lower case
    units = units.lower()

    unit_props = {
        "mpa": {
            "name": "MPa",
            "elastic_modulus": 200e3,  # MPa = N/mm^2
            "density": 7.85e-6,  # kg/mm^3
        },
        "si": {
            "name": "SI",
            "elastic_modulus": 200e9,  # N/m^2 = Pa
            "density": 7.85e3,  # kg/m^3
        },
    }

    return Material(
        name=f"Steel [{unit_props[units]['name']}]",
        elastic_modulus=unit_props[units]["elastic_modulus"],
        poissons_ratio=0.3,
        thickness=thickness,
        density=unit_props[units]["density"],
        colour=colour,
    )
