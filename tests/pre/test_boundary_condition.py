"""Tests for the BoundarCondition objects."""

import numpy as np
import pytest

from planestress.pre.boundary_condition import BoundaryCondition


def test_bc_abstract_class():
    """Tests for the abstract BoundaryCondition class."""
    boundary_condition = BoundaryCondition("x", 1.0, 0)

    with pytest.raises(NotImplementedError):
        boundary_condition.apply_bc(k=np.array([[1, 1], [1, 1]]), f=np.array([1, 2]))

    # test get dofs
    dofs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert boundary_condition.get_dofs_given_direction(dofs=dofs) == [0, 2, 4, 6, 8]

    boundary_condition.direction = "y"
    assert boundary_condition.get_dofs_given_direction(dofs=dofs) == [1, 3, 5, 7, 9]

    boundary_condition.direction = "xy"
    assert boundary_condition.get_dofs_given_direction(dofs=dofs) == dofs
