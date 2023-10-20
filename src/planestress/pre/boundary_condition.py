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
        """Inits the BoundaryCondition class.

        Args:
            marker_id: Mesh marker ID.
        """
        self.marker_id = marker_id

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.
            dofs: Degrees of freedom.

        Raises:
            NotImplementedError: If this method has not been implemented.
        """
        raise NotImplementedError


class NodeBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition at a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeBoundaryCondition class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id)
        self.direction = direction  # TODO - verify input
        self.value = value


class NodeSupport(NodeBoundaryCondition):
    """Class for adding a support to a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeSupport class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.
            dofs: Degrees of freedom, length is two for a node BC.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get relevant dof
        dof = dofs[0] if self.direction == "x" else dofs[1]

        # apply bc - TODO - confirm this theory!
        k[dof, :] = 0
        k[dof, dof] = 1
        f[dof] = self.value

        return k, f


class NodeSpring(NodeBoundaryCondition):
    """Class for adding a spring to a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeSpring class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.
            dofs: Degrees of freedom, length is two for a node BC.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get relevant dof
        dof = dofs[0] if self.direction == "x" else dofs[1]

        # apply bc - TODO - confirm this theory!
        k[dof, dof] += self.value

        return k, f


class NodeLoad(NodeBoundaryCondition):
    """Class for adding a load to a node."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeLoad class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.
            dofs: Degrees of freedom, length is two for a node BC.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get relevant dof
        dof = dofs[0] if self.direction == "x" else dofs[1]

        # apply bc
        f[dof] += self.value

        return k, f


class LineBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition along a line."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineBoundaryCondition class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id)
        self.direction = direction  # TODO - verify input
        self.value = value


class LineSupport(LineBoundaryCondition):
    """Class for adding supports along a line."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineSupport class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
        dofs: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.
            dofs: Degrees of freedom.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get relevant dofs
        dof_list = dofs[0::2] if self.direction == "x" else dofs[1::2]

        # apply bc - TODO - confirm this theory!
        for dof in dof_list:
            k[dof, :] = 0
            k[dof, dof] = 1
            f[dof] = self.value

        return k, f


class LineSpring(LineBoundaryCondition):
    """Class for adding springs along a line."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineSpring class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    # def apply_bc(
    #     self,
    #     k: npt.NDArray[np.float64],
    #     f: npt.NDArray[np.float64],
    #     dofs: list[int],
    # ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    #     """Applies the boundary condition.

    #     Args:
    #         k: Stiffness matrix.
    #         f: Load vector.
    #         dofs: Degrees of freedom.

    #     Returns:
    #         Modified stiffness matrix and load vector (``k``, ``f``).
    #     """
    # # get relevant dofs
    # dof_list = dofs[0::2] if self.direction == "x" else dofs[1::2]

    # # apply bc - TODO - confirm this theory!
    # for dof in dof_list:
    #     k[dof, dof] += self.value

    # return k, f
    # TODO - calculate equivalent spring stiffness + Tri6


class LineLoad(LineBoundaryCondition):
    """Class for adding a load to a line."""

    def __init__(
        self,
        marker_id: int,
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineLoad class.

        Args:
            marker_id: Mesh marker ID.
            direction: Direction of the boundary condition, ``"x"`` or ``"y"``.
            value: Value of the boundary condition.
        """
        super().__init__(marker_id=marker_id, direction=direction, value=value)

    # def apply_bc(
    #     self,
    #     k: npt.NDArray[np.float64],
    #     f: npt.NDArray[np.float64],
    #     dofs: list[int],
    # ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    #     """Applies the boundary condition."""
    # get relevant dof
    # dof = dofs[0] if self.direction == "x" else dofs[1]

    # # apply bc - TODO - confirm this theory!
    # k[dof, :] = 0
    # k[dof, dof] = 1
    # f[dof] = self.value

    # return k, f
    # TODO - calculate equivalent loads + Tri6
