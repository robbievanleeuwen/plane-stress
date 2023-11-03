"""Simple patch validation tests."""

import pytest
import pytest_check as check


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
    abs_error = 1e-6

    # check displacements
    check.almost_equal(max(res.ux), u_x, abs=abs_error)
    check.almost_equal(max(res.uy), u_y)

    # check stresses
    sigs = res.get_nodal_stresses()
    check.almost_equal(max(sigs[:, 0]), sig_x, abs=abs_error)
    check.almost_equal(max(sigs[:, 1]), sig_y)
    check.almost_equal(max(sigs[:, 2]), sig_xy, abs=abs_error)

    # check reactions
    check.almost_equal(sum(res.f_r), -1)

    # check sum of forces
    check.almost_equal(sum(res.f), 0, abs=abs_error)
