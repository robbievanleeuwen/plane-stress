"""Library of commonly used geometries."""

from __future__ import annotations

import numpy as np
from shapely import Polygon

from planestress.pre.geometry import Geometry
from planestress.pre.material import DEFAULT_MATERIAL, Material


def rectangle(
    d: float,
    b: float,
    n_x: int = 1,
    n_y: int = 1,
    material: Material = DEFAULT_MATERIAL,
    tol: int = 12,
) -> Geometry:
    """Creates a rectangular geometry with the bottom-left corner at the origin.

    Args:
        d: Depth of the rectangle.
        b: Width of the rectangle.
        n_x: Number of subdivisions in the ``x`` direction. Defaults to ``1``.
        n_y: Number of subdivisions in the ``y`` direction. Defaults to ``1``.
        material: ``Material`` object to apply to the rectangle. Defaults to
            ``DEFAULT_MATERIAL``, i.e. a material with unit properties and a Poisson's
            ratio of zero.
        tol: The points in the geometry get rounded to ``tol`` digits. Defaults to
            ``12``.

    Returns:
        Rectangular ``Geometry`` object.

    Example:
        TODO.
    """
    # initialise shell list
    shell = []
    b = float(b)
    d = float(d)

    # add points
    shell.extend([(i * b / n_x, 0.0) for i in range(n_x + 1)])
    shell.extend([(b, i * d / n_y) for i in range(n_y + 1)])
    shell.extend([(b - i * b / n_x, d) for i in range(n_x + 1)])
    shell.extend([(0.0, d - i * d / n_y) for i in range(n_y + 1)])
    poly = Polygon(shell=shell)

    return Geometry(polygons=poly, materials=[material], tol=tol)


def circle(
    r: float,
    n: int,
    material: Material = DEFAULT_MATERIAL,
    tol: int = 12,
) -> Geometry:
    """Creates a circular geometry with the center at the origin.

    Args:
        r: Radius of the circle.
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
        theta = idx * 2.0 * np.pi * 1.0 / n

        # calculate location of the point
        x = r * np.cos(theta)
        y = r * np.sin(theta)

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

        - ``"MPa"`` - :math:`g = 9.81 \times 10^3 \textrm{ mm/s}^2`
        - ``"SI"`` - :math:`g = 9.81 \textrm{ m/s}^2`
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

        - ``"MPa"``: Newtons :math:`[\textrm{N}]` and millimetres :math:`[\textrm{mm}]`.

          - Elastic modulus: :math:`200 \times 10^3 \textrm{ MPa}`
          - Poisson's ratio: :math:`0.3`
          - Density: :math:`7.85 \times 10^{-6} \textrm{ kg/mm}`^3`

        - ``"SI"``: Newtons :math:`[\textrm{N}]` and metres :math:`[\textrm{m}]`.

          - Elastic modulus: :math:`200 \times 10^9 \textrm{ Pa}`
          - Poisson's ratio: :math:`0.3`
          - Density: :math:`7.85 \times 10^3 \textrm{ kg/m}`^3`

    Example:
        TODO.
    """
    # convert units to lower case
    units = units.lower()

    unit_props: dict[str, dict[str, str | float]] = {
        "mpa": {
            "name": "MPa",
            "elastic_modulus": 200e3,  # MPa = N/mm^2
            "density": 7.85e-9,  # T/mm^3
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


def concrete_material(
    elastic_modulus: float,
    thickness: float,
    units: str = "MPa",
    color: str = "lightgrey",
) -> Material:
    r"""Creates a concrete material object with consistent units.

    Args:
        elastic_modulus: Elastic modulus of the concrete.
        thickness: Thickness of the concrete.
        units: Units system to use. See below for options. Defaults to ``"MPa"``.
        color: Material color for rendering. Defaults to ``"lightgrey"``.

    Raises:
        ValueError: If the value of ``units`` is not in the list below.

    Returns:
        Concrete material object.

    .. admonition:: Units

        The value for ``units`` may be one of the following:

        - ``"MPa"``: Newtons :math:`[\textrm{N}]` and millimetres :math:`[\textrm{mm}]`.

          - Poisson's ratio: :math:`0.2`
          - Density: :math:`2.4 \times 10^{-6} \textrm{ kg/mm}`^3`

        - ``"SI"``: Newtons :math:`[\textrm{N}]` and metres :math:`[\textrm{m}]`.

          - Poisson's ratio: :math:`0.2`
          - Density: :math:`2.4 \times 10^3 \textrm{ kg/m}`^3`

    Example:
        TODO.
    """
    # convert units to lower case
    units = units.lower()

    unit_props: dict[str, dict[str, str | float]] = {
        "mpa": {
            "name": "MPa",
            "density": 2.4e-9,  # T/mm^3
        },
        "si": {
            "name": "SI",
            "density": 2.4e3,  # kg/m^3
        },
    }

    try:
        return Material(
            name=f"Concrete [{unit_props[units]['name']}]",
            elastic_modulus=elastic_modulus,
            poissons_ratio=0.2,
            thickness=thickness,
            density=float(unit_props[units]["density"]),
            color=color,
        )
    except KeyError as exc:
        raise ValueError(f"{units} is not a valid input for 'units'.") from exc
