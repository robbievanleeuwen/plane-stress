"""Class describing a planestress load case."""

from __future__ import annotations

from dataclasses import dataclass

from planestress.pre.boundary_condition import BoundaryCondition


@dataclass
class LoadCase:
    """Class for a load case.

    Args:
        boundary_conditions: List of boundary conditions.
        global_accelerations: Global acceleration for the current load case. Defaults to
            ``0.0``.
    """

    boundary_conditions: list[BoundaryCondition]
    global_accelerations: float = 0.0


# TODO - add a persistent load case??
