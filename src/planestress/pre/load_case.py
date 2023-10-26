"""Class describing a planestress load case."""

from __future__ import annotations

from dataclasses import dataclass

from planestress.pre.boundary_condition import BoundaryCondition


@dataclass
class LoadCase:
    """Class for a load case.

    Args:
        boundary_conditions: List of boundary conditions.
        acceleration_field: Acceleration field for the current load case (``a_x``,
            ``a_y``). Defaults to ``(0.0, 0.0)``.
    """

    boundary_conditions: list[BoundaryCondition]
    acceleration_field: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Post init method to sort boundary conditions."""
        # sort boundary conditions
        self.boundary_conditions.sort(key=lambda bc: bc.priority)


# TODO - add a persistent load case??
