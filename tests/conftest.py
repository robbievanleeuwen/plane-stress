"""pytest configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from planestress.analysis.plane_stress import PlaneStress
from planestress.pre.analysis_case import AnalysisCase
from planestress.pre.boundary_condition import LineLoad, LineSupport, NodeSupport
from planestress.pre.library import rectangle

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def unit_square() -> Callable:
    """Creates a unit square with a unit tensile load applied.

    The default material is used and the base of the square is fixed.

    Returns:
        Generator function, returning a ``PlaneStress`` object.
    """

    def _generate(
        lc: int,
        element_type: str,
    ) -> PlaneStress:
        """Generates the unit square.

        Args:
            lc: Characterisic mesh length.
            element_type: Element type, can be ``"Tri3"``, ``"Tri6"``, ``"Quad4"``,
                ``"Quad8"`` or ``"Quad9"``.

        Returns:
            Plane stress object.
        """
        geom = rectangle(1, 1)
        lhs = NodeSupport((0, 0), "x")
        bot = LineSupport((0, 0), (1, 0), "y")
        load = LineLoad((0, 1), (1, 1), "y", 1)
        analysis_case = AnalysisCase([lhs, bot, load])

        if element_type == "Tri3":
            geom.create_mesh(mesh_sizes=lc, mesh_order=1)
        elif element_type == "Tri6":
            geom.create_mesh(mesh_sizes=lc, mesh_order=2)
        elif element_type == "Quad4":
            geom.create_mesh(
                mesh_sizes=lc, quad_mesh=True, mesh_order=1, mesh_algorithm=8
            )
        elif element_type == "Quad8":
            geom.create_mesh(
                mesh_sizes=lc,
                quad_mesh=True,
                mesh_order=2,
                serendipity=True,
                mesh_algorithm=8,
            )
        elif element_type == "Quad9":
            geom.create_mesh(
                mesh_sizes=lc, quad_mesh=True, mesh_order=2, mesh_algorithm=8
            )

        return PlaneStress(geom, [analysis_case])

    return _generate
