"""Validation tests for simple beams in flexure."""

import pytest
import pytest_check as check

from planestress.analysis import PlaneStress
from planestress.pre import AnalysisCase, LineLoad, LineSupport, NodeLoad, NodeSupport
from planestress.pre.library import rectangle


@pytest.mark.parametrize("el_type", ["Tri6", "Quad8", "Quad9"])
def test_simple_beam_point_load(el_type):
    """Tests a simply supported beam with a point load."""
    # set parameters
    l = 1
    d = 0.05
    p = 1.0

    # calculate theoretical deflection and stress
    ixx = 1 * d**3 / 12
    u = p * l**3 / (48 * ixx)
    mx = p * l / 4
    sig = mx * 0.5 * d / ixx

    # create geometry
    geom = rectangle(d=d, b=l, n_x=2)

    # supports and load
    lhs_support = NodeSupport(point=(0, 0), direction="xy")
    rhs_support = NodeSupport(point=(l, 0), direction="y")
    load = NodeLoad(point=(0.5 * l, d), direction="y", value=-p)
    case = AnalysisCase([lhs_support, rhs_support, load])

    # create mesh
    ms = d / 2  # two elements thick

    if el_type == "Quad9":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2, serendipity=True)
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=ms, mesh_order=2)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    check.almost_equal(min(res.u), -u, rel=1e-2)
    check.almost_equal(max(res.get_nodal_stresses()[:, 0]), sig, rel=1.5e-2)


@pytest.mark.parametrize("el_type", ["Tri6", "Quad8", "Quad9"])
def test_simple_udl(el_type):
    """Tests a simply supported beam with a uniformly distributed load."""
    # set parameters
    l = 1
    d = 0.05
    w = 1.0

    # calculate theoretical deflection and stress
    ixx = 1 * d**3 / 12
    u = 5 * w * l**4 / (384 * ixx)
    mx = w * l**2 / 8
    sig = mx * 0.5 * d / ixx

    # create geometry
    geom = rectangle(d=d, b=l)

    # supports and load
    lhs_support = NodeSupport(point=(0, 0), direction="xy")
    rhs_support = NodeSupport(point=(l, 0), direction="y")
    load = LineLoad(point1=(0, d), point2=(l, d), direction="y", value=-w)
    case = AnalysisCase([lhs_support, rhs_support, load])

    # create mesh
    ms = d / 2  # two elements thick

    if el_type == "Quad9":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2, serendipity=True)
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=ms, mesh_order=2)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    check.almost_equal(min(res.u), -u, rel=1e-2)
    check.almost_equal(max(res.get_nodal_stresses()[:, 0]), sig, rel=1.5e-2)


@pytest.mark.parametrize("el_type", ["Tri6", "Quad8", "Quad9"])
def test_cantilever_point_load(el_type):
    """Tests a cantilever beam with a point load."""
    # set parameters
    l = 1
    d = 0.05
    p = 1.0

    # calculate theoretical deflection and stress
    ixx = 1 * d**3 / 12
    u = p * l**3 / (3 * ixx)
    mx = p * l
    sig = mx * 0.5 * d / ixx

    # create geometry
    geom = rectangle(d=d, b=l)

    # supports and load
    x_support = LineSupport(point1=(0, 0), point2=(0, d), direction="x")
    y_support = NodeSupport(point=(0, 0), direction="y")
    load = NodeLoad(point=(l, d), direction="y", value=-p)
    case = AnalysisCase([x_support, y_support, load])

    # create mesh
    ms = d / 2  # two elements thick

    if el_type == "Quad9":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2, serendipity=True)
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=ms, mesh_order=2)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    check.almost_equal(min(res.u), -u, rel=1e-2)
    check.almost_equal(max(res.get_nodal_stresses()[:, 0]), sig, rel=1.5e-2)


@pytest.mark.parametrize("el_type", ["Tri6", "Quad8", "Quad9"])
def test_cantilever_udl(el_type):
    """Tests a cantilever beam with a uniformly distributed load."""
    # set parameters
    l = 1
    d = 0.05
    w = 1.0

    # calculate theoretical deflection and stress
    ixx = 1 * d**3 / 12
    u = w * l**4 / (8 * ixx)
    mx = w * l**2 / 2
    sig = mx * 0.5 * d / ixx

    # create geometry
    geom = rectangle(d=d, b=l)

    # supports and load
    x_support = LineSupport(point1=(0, 0), point2=(0, d), direction="x")
    y_support = NodeSupport(point=(0, 0), direction="y")
    load = LineLoad(point1=(0, d), point2=(l, d), direction="y", value=-w)
    case = AnalysisCase([x_support, y_support, load])

    # create mesh
    ms = d / 2  # two elements thick

    if el_type == "Quad9":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2)
    elif el_type == "Quad8":
        geom.create_mesh(mesh_sizes=ms, quad_mesh=True, mesh_order=2, serendipity=True)
    elif el_type == "Tri6":
        geom.create_mesh(mesh_sizes=ms, mesh_order=2)

    # solve
    ps = PlaneStress(geom, [case])
    results_list = ps.solve(solver_type="pardiso")
    res = results_list[0]

    check.almost_equal(min(res.u), -u, rel=1e-2)
    check.almost_equal(max(res.get_nodal_stresses()[:, 0]), sig, rel=1.5e-2)
