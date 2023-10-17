"""Class describing a planestress load case."""

from __future__ import annotations

from dataclasses import dataclass

from planestress.pre.boundary_condition import BoundaryCondition


@dataclass
class LoadCase:
    """Class for a load case."""

    boundary_conditions: list[BoundaryCondition]
    global_accelerations: list[float] | None = None  # TODO - verify list lengths


# TODO - add a persistent load case??
