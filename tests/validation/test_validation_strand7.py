"""Validation tests taken from the Strand7 validation manual.

Insert reference here...
"""

import numpy as np
import pytest
import pytest_check as check

from planestress.analysis.plane_stress import PlaneStress
from planestress.pre.library import circle, rectangle, steel_material
from planestress.pre.load_case import LoadCase


def test_vls1():
    """VLS1: Elliptic Membrane.

    An elliptical plate with an elliptical hole is analysed.
    Outer ellipse - (x / 3.25)**2 + (y / 2.75)**2 = 1
    Inner ellipse - (x / 2)**2 + (y)**2 = 1

    Uniform outward pressure is applied at the outer boundary. (10 MPa)
    As both the structure and the loading condition are symmetric, only a quarter of the
    structure is modelled. (1st quadrant).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=2, y=0) of 92.7 MPa.

    TODO - must first implement line load normal to curve.
    """
    pass


def test_vls8():
    """VLS8: Circular Membrane - Edge Pressure.

    A ring under uniform external pressure of 100 MPa is analysed.
    (inner = 10m, outer = 11m, t=1m).

    One eighth of the ring (45 deg) is modelled via nodal restraints in a UCS.
    In the validation of plane-stress one-quarter of the model will be analysed (no
    implementation of 45 deg rollers).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=10, y=0) of -1150 MPa.

    TODO - must first implement line load normal to curve.
    """
    pass


@pytest.mark.parametrize("el_type", ["Quad4", "Tri6", "Quad8", "Quad9"])
def test_vls9(el_type):
    """VLS9: Circular Membrane - Point Load.

    A ring under concentrated forces is analysed. (10000 kN at 4 x 45 deg).
    (inner = 10m, outer = 11m, t=1m).

    One eighth of the ring (45 deg) is modelled via nodal restraints in a UCS.
    In the validation of plane-stress one-quarter of the model will be analysed (no
    implementation of 45 deg rollers).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=10, y=0) of -53.2 MPa.
    """
    rel = 0.03  # aim for 3% error

    # define materials - use N & mm
    steel = steel_material(thickness=1000.0)

    # define geometry
    circle_outer = circle(r=11e3, n=128, material=steel)
    circle_inner = circle(r=10e3, n=128)
    bbox = rectangle(d=12e3, b=12e3)  # bounding box
    geom = (circle_outer - circle_inner) & bbox

    # create supports
    lhs_support = geom.add_line_support(
        point1=(0.0, 10e3), point2=(0.0, 11e3), direction="x"
    )
    rhs_support = geom.add_line_support(
        point1=(10e3, 0.0), point2=(11e3, 0.0), direction="y"
    )

    # create loads
    pt = (11e3 / np.sqrt(2), 11e3 / np.sqrt(2))
    force = 10e6 / np.sqrt(2)  # force component in x and y directions
    load_x = geom.add_node_load(point=pt, direction="x", value=-force)
    load_y = geom.add_node_load(point=pt, direction="y", value=-force)
    lc = LoadCase([lhs_support, rhs_support, load_x, load_y])

    # create mesh
    if el_type == "Quad4":
        geom.create_mesh(
            mesh_sizes=1000.0, quad_mesh=True, mesh_order=1, mesh_algorithm=11
        )
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=1000.0, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(
            mesh_sizes=1000.0,
            quad_mesh=True,
            mesh_order=2,
            serendipity=True,
            mesh_algorithm=11,
        )
    elif el_type == "Quad9":
        geom.create_mesh(
            mesh_sizes=1000.0, quad_mesh=True, mesh_order=2, mesh_algorithm=11
        )
    else:
        raise ValueError(f"{el_type} element not supported for this test.")

    # solve
    ps = PlaneStress(geom, [lc])
    results_list = ps.solve()
    res = results_list[0]

    # get stress at (x=10, y=0)
    node_idx = ps.mesh.get_node_idx_by_coordinates(10e3, 0.0)
    sig_yy = res.get_nodal_stresses()[node_idx][1]
    target_stress = -53.2

    check.almost_equal(target_stress, sig_yy, rel=rel)
