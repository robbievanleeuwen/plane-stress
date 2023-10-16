"""Classes describing a planestress boundary conditions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class BoundaryCondition:
    """Abstract base class for a boundary condition."""

    def __init__(
        self,
        marker_id: int,
    ) -> None:
        """Inits the BoundaryCondition class."""
        self.marker_id = marker_id

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition."""
        raise NotImplementedError


class NodeBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition at a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeBoundaryCondition class."""
        super().__init__(marker_id=marker_id)
        self.direction = direction  # TODO - verify
        self.value = value


class NodeSupport(NodeBoundaryCondition):
    """Class for adding a support to a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeSupport class."""
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition."""
        # get relevant dof
        dof = dofs[0] if self.direction == "x" else dofs[1]

        # apply bc
        k[dof, :] = 0
        k[dof, dof] = 1
        f[dof] = self.value

        return k, f


class NodeSpring(NodeBoundaryCondition):
    """Class for adding a spring to a node."""

    pass


class NodeLoad(NodeBoundaryCondition):
    """Class for adding a load to a node."""

    pass


class LineBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition along a line."""

    pass
