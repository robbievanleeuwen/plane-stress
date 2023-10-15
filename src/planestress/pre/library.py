"""Library of commonly used geometries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely import Polygon

from planestress.pre.geometry import Geometry
from planestress.pre.material import DEFAULT_MATERIAL


if TYPE_CHECKING:
    from planestress.pre.material import Material


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
