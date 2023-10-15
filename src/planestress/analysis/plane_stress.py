"""Class for a planestress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from planestress.pre.geometry import Geometry
    from planestress.pre.load_case import LoadCase


class PlaneStress:
    """Class for a plane-stress analysis."""

    def __init__(
        self,
        geometry: Geometry,
        load_cases: list[LoadCase],
    ) -> None:
        """Inits the PlaneStress class."""
        self.geometry = geometry
        self.load_cases = load_cases
