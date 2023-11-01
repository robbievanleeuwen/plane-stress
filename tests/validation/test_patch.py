"""Simple patch validation tests."""

from typing import Callable

import pytest
import pytest_check as check

from planestress.analysis.plane_stress import PlaneStress
from planestress.pre.library import rectangle
from planestress.pre.load_case import LoadCase


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
        lhs = geom.add_node_support((0, 0), "x")
        bot = geom.add_line_support((0, 0), (1, 0), "y")
        load = geom.add_line_load((0, 1), (1, 1), "y", 1)
        load_case = LoadCase([lhs, bot, load])

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

        return PlaneStress(geom, [load_case])

    return _generate


@pytest.mark.parametrize("el_type", ["Tri3", "Tri6", "Quad4", "Quad8", "Quad9"])
@pytest.mark.parametrize("lc", [1, 0.1, 0.05])
def test_unit_square_tensile(unit_square, lc, el_type):
    """A patch test with a unit square under a unit tensile load."""
    ps = unit_square(lc, el_type)
    res = ps.solve()[0]

    # results are all equal to 1 unit!
    sig_x = 0
    sig_y = 1
    sig_xy = 0
    u_x = 0  # nu = 0
    u_y = 1

    # check displacements
    check.almost_equal(max(res.ux), u_x)
    check.almost_equal(max(res.uy), u_y)

    # check stresses
    sigs = res.get_nodal_stresses()
    check.almost_equal(max(sigs[:, 0]), sig_x)
    check.almost_equal(max(sigs[:, 1]), sig_y)
    check.almost_equal(max(sigs[:, 2]), sig_xy)

    # check reactions
    check.almost_equal(sum(res.f_r), -1)

    # check sum of forces
    check.almost_equal(sum(res.f), 0)
