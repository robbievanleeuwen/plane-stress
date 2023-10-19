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
    """Creates a rectangular geometry with the bottom-left corner at the origin.

    Args:
        d: Depth of the rectangle.
        b: Width of the rectangle.
        material: ```Material`` object to apply to the rectangle. Defaults to
            ``DEFAULT_MATERIAL``, i.e. a material with unit properties and a Poisson's
            ratio of zero.
        tol: The points in the geometry get rounded to ``tol`` digits. Defaults to
            ``12``.

    Returns:
        Rectangular ``Geometry`` object.

    Example:
        TODO.
    """
    shell = [(0, 0), (b, 0), (b, d), (0, d)]
    poly = Polygon(shell=shell)

    return Geometry(polygons=poly, materials=[material], tol=tol)


def circle(
    d: float,
    n: int,
    material: Material = DEFAULT_MATERIAL,
    tol: int = 12,
) -> Geometry:
    """Creates a circular geometry with the centre at the origin.

    Args:
        d: Diameter of the circle.
        n: Number of points to discretise the circle.
        material: ```Material`` object to apply to the circle. Defaults to
            ``DEFAULT_MATERIAL``, i.e. a material with unit properties and a Poisson's
            ratio of zero.
        tol: The points in the geometry get rounded to ``tol`` digits. Defaults to
            ``12``.

    Returns:
        Circular ``Geometry`` object.

    Example:
        TODO.
    """
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
    r"""Returns the gravitational acceleration with consistent units.

    Args:
        units: Units system to use. See below for options. Defaults to ``"MPa"``.

    Raises:
        ValueError: If the value of ``units`` is not in the list below.

    Returns:
        Acceleration due to gravity.

    .. admonition:: Units
        The value for ``units`` may be one of the following:

        - ``"MPa"`` - :math:`g = 9.81 \times 10^3 \textrm{mm/s}^2`
        - ``"SI"`` - :math:`g = 9.81 \textrm{m/s}^2`
    """
    # convert units to lower case
    units = units.lower()

    gravity_units = {
        "mpa": 9.81e3,  # mm/s^2
        "si": 9.81,  # m/s^2
    }

    try:
        return gravity_units[units]
    except KeyError as exc:
        raise ValueError(f"{units} is not a valid input for 'units'.") from exc


def steel_material(
    thickness: float,
    units: str = "MPa",
    color: str = "grey",
) -> Material:
    r"""Creates a steel material object with consistent units.

    Args:
        thickness: Thickness of the steel.
        units: Units system to use. See below for options. Defaults to ``"MPa"``.
        color: Material color for rendering. Defaults to ``"grey"``.

    Raises:
        ValueError: If the value of ``units`` is not in the list below.

    Returns:
        Steel material object.

    .. admonition:: Units
        The value for ``units`` may be one of the following:

        - ``"MPa"``: Newtons [N] and millimetres [mm]

          - Elastic modulus: :math:`200 \times 10^3 \textrm{MPa}`
          - Poisson's ratio: :math:`0.3`
          - Density: :math:`7.85 \times 10^{-6} \textrm{kg/mm}`^3`

        - ``"SI"``: Newtons [N] and metres [m]

          - Elastic modulus: :math:`200 \times 10^9 \textrm{Pa}`
          - Poisson's ratio: :math:`0.3`
          - Density: :math:`7.85 \times 10^3 \textrm{kg/m}`^3`

    Example:
        TODO.
    """
    # convert units to lower case
    units = units.lower()

    unit_props: dict[str, dict[str, str | float]] = {
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

    try:
        return Material(
            name=f"Steel [{unit_props[units]['name']}]",
            elastic_modulus=float(unit_props[units]["elastic_modulus"]),
            poissons_ratio=0.3,
            thickness=thickness,
            density=float(unit_props[units]["density"]),
            color=color,
        )
    except KeyError as exc:
        raise ValueError(f"{units} is not a valid input for 'units'.") from exc
