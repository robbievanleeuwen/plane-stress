"""Validation tests taken from the Strand7 validation manual.

Insert reference here...
"""

from typing import Callable

import numpy as np
import pytest
import pytest_check as check
from shapely import Polygon

import planestress.pre.boundary_condition as bc
from planestress.analysis import PlaneStress
from planestress.pre import AnalysisCase, Geometry, Material
from planestress.pre.library import circle, rectangle, steel_material


@pytest.fixture
def circular_membrane() -> Callable:
    """Creates a meshed circular membrane for VLS8 & VLS9.

    Returns:
        Generator function, returning a ``Geometry`` object.
    """

    def _generate(el_type: str) -> Geometry:
        """Generates the unit square.

        Args:
            lc: Characterisic mesh length.
            element_type: Element type, can be ``"Tri3"``, ``"Tri6"``, ``"Quad4"``,
                ``"Quad8"`` or ``"Quad9"``.

        Returns:
            Geometry object.
        """
        # define materials - use N & mm
        steel = steel_material(thickness=1000.0)

        # define geometry
        circle_outer = circle(r=11e3, n=128, material=steel)
        circle_inner = circle(r=10e3, n=128)
        bbox = rectangle(d=12e3, b=12e3)  # bounding box
        geom = (circle_outer - circle_inner) & bbox

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

        return geom

    return _generate


@pytest.mark.parametrize("el_type", ["Quad4", "Tri6", "Quad8", "Quad9"])
def test_vls1(el_type):
    """VLS1: Elliptic Membrane.

    An elliptical plate with an elliptical hole is analysed.
    Outer ellipse - (x / 3.25)**2 + (y / 2.75)**2 = 1
    Inner ellipse - (x / 2)**2 + (y)**2 = 1

    Uniform outward pressure is applied at the outer boundary. (10 MPa)
    As both the structure and the loading condition are symmetric, only a quarter of the
    structure is modelled. (1st quadrant).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=2, y=0) of 92.7 MPa.
    """
    rel = 0.03  # aim for 3% error

    # define materials - use N & mm
    steel = steel_material(thickness=1)

    # create geometry
    shell = []
    n_curve = 24

    # inner curve
    a = 2.0
    b = 1.0

    for idx in range(n_curve):
        theta = idx / (n_curve - 1) * np.pi / 2
        shell.append((a * np.cos(theta), b * np.sin(theta)))

    # outer curve
    a = 3.25
    b = 2.75

    for idx in range(n_curve):
        theta = np.pi / 2 - idx / (n_curve - 1) * np.pi / 2
        shell.append((a * np.cos(theta), b * np.sin(theta)))

    geom = Geometry(polygons=Polygon(shell=shell), materials=steel)

    # create loads
    loads = []
    sig = 10

    for idx in range(n_curve - 1):
        theta1 = idx / (n_curve - 1) * np.pi / 2
        theta2 = (idx + 1) / (n_curve - 1) * np.pi / 2
        pt1 = (a * np.cos(theta1), b * np.sin(theta1))
        pt2 = (a * np.cos(theta2), b * np.sin(theta2))
        loads.append(bc.LineLoad(pt1, pt2, "n", -sig))

    # create supports
    lhs_support = bc.LineSupport(point1=(0.0, 1), point2=(0.0, 2.75), direction="x")
    rhs_support = bc.LineSupport(point1=(2, 0.0), point2=(3.25, 0.0), direction="y")
    bcs = [lhs_support, rhs_support]
    bcs.extend(loads)
    case = AnalysisCase(bcs)

    # create mesh
    if el_type == "Quad4":
        geom.create_mesh(quad_mesh=True, mesh_order=1, mesh_algorithm=11)
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=0.15, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(
            quad_mesh=True, mesh_order=2, serendipity=True, mesh_algorithm=8
        )
    elif el_type == "Quad9":
        geom.create_mesh(quad_mesh=True, mesh_order=2, mesh_algorithm=8)
    else:
        raise ValueError(f"{el_type} element not supported for this test.")

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    # get stress at (x=2, y=0)
    node_idx = ps.mesh.get_node_idx_by_coordinates(2.0, 0.0)
    sig_yy = res.get_nodal_stresses()[node_idx][1]
    target_stress = 92.7

    check.almost_equal(target_stress, sig_yy, rel=rel)


@pytest.mark.parametrize("el_type", ["Quad4", "Tri6", "Quad8", "Quad9"])
def test_vls8(el_type, circular_membrane):
    """VLS8: Circular Membrane - Edge Pressure.

    A ring under uniform external pressure of 100 MPa is analysed.
    (inner = 10m, outer = 11m, t=1m).

    One eighth of the ring (45 deg) is modelled via nodal restraints in a UCS.
    In the validation of plane-stress one-quarter of the model will be analysed (no
    implementation of 45 deg rollers).

    Materials: Steel, E=200e3, v=0.3

    Target value - tangential stress at (x=10, y=0) of -1150 MPa.
    """
    rel = 0.03  # aim for 3% error

    # create meshed geometry
    geom = circular_membrane(el_type)

    # create supports
    lhs_support = bc.LineSupport(point1=(0.0, 10e3), point2=(0.0, 11e3), direction="x")
    rhs_support = bc.LineSupport(point1=(10e3, 0.0), point2=(11e3, 0.0), direction="y")

    # create loads
    loads = []
    sig = 100  # MPa
    p = sig * 1000  # N/mm
    for idx in range(32):
        theta1 = np.pi / 64 * idx
        theta2 = np.pi / 64 * (idx + 1)
        pt1 = (11e3 * np.cos(theta1), 11e3 * np.sin(theta1))
        pt2 = (11e3 * np.cos(theta2), 11e3 * np.sin(theta2))
        loads.append(bc.LineLoad(pt1, pt2, "n", p))

    bcs = [lhs_support, rhs_support]
    bcs.extend(loads)
    case = AnalysisCase(bcs)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    # get stress at (x=10, y=0)
    node_idx = ps.mesh.get_node_idx_by_coordinates(10e3, 0.0)
    sig_yy = res.get_nodal_stresses()[node_idx][1]
    target_stress = -1150.0

    check.almost_equal(target_stress, sig_yy, rel=rel)


@pytest.mark.parametrize("el_type", ["Quad4", "Tri6", "Quad8", "Quad9"])
def test_vls9(el_type, circular_membrane):
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

    # create meshed geometry
    geom = circular_membrane(el_type)

    # create supports
    lhs_support = bc.LineSupport(point1=(0.0, 10e3), point2=(0.0, 11e3), direction="x")
    rhs_support = bc.LineSupport(point1=(10e3, 0.0), point2=(11e3, 0.0), direction="y")

    # create loads
    pt = (11e3 / np.sqrt(2), 11e3 / np.sqrt(2))
    force = 10e6 / np.sqrt(2)  # force component in x and y directions
    load = bc.NodeLoad(point=pt, direction="xy", value=-force)
    case = AnalysisCase([lhs_support, rhs_support, load])

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    # get stress at (x=10, y=0)
    node_idx = ps.mesh.get_node_idx_by_coordinates(10e3, 0.0)
    sig_yy = res.get_nodal_stresses()[node_idx][1]
    target_stress = -53.2

    check.almost_equal(target_stress, sig_yy, rel=rel)


def test_vls11():
    """VLS11: Plate Patch Test.

    Enforced displacements are applied to acheive a uniform strain of 1e-3.

    Material properties: E = 1e6, nu = 0.25, t = 0.001.

    sig_xx = sig_yy = 1333
    sig_xy = 400
    """
    # create material
    material = Material(elastic_modulus=1e6, poissons_ratio=0.25, thickness=0.001)

    # create geometry
    geom = rectangle(d=0.12, b=0.24, material=material)

    # add nodes
    geom.embed_point(x=0.04, y=0.02)
    geom.embed_point(x=0.08, y=0.08)
    geom.embed_point(x=0.16, y=0.08)
    geom.embed_point(x=0.18, y=0.03)

    # create mesh
    geom.create_mesh(mesh_sizes=0.24)

    # apply loads
    loads = []
    pts = [
        (0, 0),
        (0, 0.12),
        (0.24, 0.12),
        (0.24, 0),
        (0.04, 0.02),
        (0.08, 0.08),
        (0.16, 0.08),
        (0.18, 0.03),
    ]

    for pt in pts:
        dx = 1e-3 * (pt[0] + 0.5 * pt[1])
        dy = 1e-3 * (pt[1] + 0.5 * pt[0])
        loads.append(bc.NodeSupport(point=pt, direction="x", value=dx))
        loads.append(bc.NodeSupport(point=pt, direction="y", value=dy))

    case = AnalysisCase(loads)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    # check stresses
    sig_xx = res.get_nodal_stresses()[0][0]
    sig_yy = res.get_nodal_stresses()[0][1]
    sig_xy = res.get_nodal_stresses()[0][2]

    check.almost_equal(4000 / 3, sig_xx)
    check.almost_equal(4000 / 3, sig_yy)
    check.almost_equal(400, sig_xy)
