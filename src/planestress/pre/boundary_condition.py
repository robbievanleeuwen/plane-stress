"""Classes describing a planestress boundary conditions.

Boundary condition application priorities:
0 - Loads
1 - Springs
2 - Supports
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from planestress.analysis.utils import dof_map


if TYPE_CHECKING:
    from planestress.pre.mesh import TaggedEntity, TaggedLine, TaggedNode


class BoundaryCondition:
    """Abstract base class for a boundary condition.

    Attributes:
        mesh_tag: Tagged entity object.
    """

    def __init__(
        self,
        direction: str,
        value: float,
        priority: int,
    ) -> None:
        """Inits the BoundaryCondition class.

        Args:
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
            priority: Integer denoting the order in which the boundary condition gets
                applied.
        """
        self.direction = direction  # TODO - verify input
        self.value = value
        self.priority = priority
        self.mesh_tag: TaggedEntity

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Raises:
            NotImplementedError: If this method has not been implemented.
        """
        raise NotImplementedError

    def get_dofs_given_direction(
        self,
        dofs: list[int],
    ) -> list[int]:
        """Gets the degrees of freedom based on the BC direction.

        Args:
            dofs: Degrees of freedom.

        Returns:
            Degrees of freeom in BC direction.
        """
        # get relevant dofs
        if self.direction == "x":
            dofs = dofs[0::2]
        elif self.direction == "y":
            dofs = dofs[1::2]
        else:
            dofs = dofs

        return dofs


class NodeBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition at a node.

    Attributes:
        mesh_tag: Tagged node object.
    """

    def __init__(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
        priority: int,
    ) -> None:
        """Inits the NodeBoundaryCondition class.

        Args:
            point: Point tuple (``x``, ``y``) describing the node location.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
            priority: Integer denoting the order in which the boundary condition gets
                applied.
        """
        super().__init__(direction=direction, value=value, priority=priority)
        self.point = point
        self.mesh_tag: TaggedNode

    def __repr__(self) -> str:
        """Override __repr__ method.

        Returns:
            String representation of the object.
        """
        try:
            tag = self.mesh_tag
        except AttributeError:
            tag = None

        return (
            f"BC Type: {self.__class__.__name__}, dir: {self.direction}, val: "
            f"{self.value}, mesh tag: {tag}"
        )

    def get_node_dofs(self) -> list[int]:
        """Get the degrees of freedom of the node.

        Returns:
            List (length 2) of degrees of freedom.
        """
        return dof_map(node_idxs=[self.mesh_tag.node_idx])


class NodeSupport(NodeBoundaryCondition):
    """Class for adding a support to a node.

    Attributes:
        mesh_tag: Tagged node object.
    """

    def __init__(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeSupport class.

        Args:
            point: Point tuple (``x``, ``y``) describing the node location.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(point=point, direction=direction, value=value, priority=2)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get nodal dofs
        dofs = self.get_node_dofs()

        # get relevant dofs
        dofs = self.get_dofs_given_direction(dofs=dofs)

        for dof in dofs:
            # apply bc - TODO - confirm this theory!
            k[dof, :] = 0
            k[dof, dof] = 1
            f[dof] = self.value

        return k, f


class NodeSpring(NodeBoundaryCondition):
    """Class for adding a spring to a node.

    Attributes:
        mesh_tag: Tagged node object.
    """

    def __init__(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeSpring class.

        Args:
            point: Point tuple (``x``, ``y``) describing the node location.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(point=point, direction=direction, value=value, priority=1)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get nodal dofs
        dofs = self.get_node_dofs()

        # get relevant dofs
        dofs = self.get_dofs_given_direction(dofs=dofs)

        for dof in dofs:
            # apply bc - TODO - confirm this theory!
            k[dof, dof] += self.value

        return k, f


class NodeLoad(NodeBoundaryCondition):
    """Class for adding a load to a node.

    Attributes:
        mesh_tag: Tagged node object.
    """

    def __init__(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the NodeLoad class.

        Args:
            point: Point tuple (``x``, ``y``) describing the node location.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(point=point, direction=direction, value=value, priority=0)

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get nodal dofs
        dofs = self.get_node_dofs()

        # get relevant dofs
        dofs = self.get_dofs_given_direction(dofs=dofs)

        for dof in dofs:
            # apply bc
            f[dof] += self.value

        return k, f


class LineBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition along a line.

    Attributes:
        mesh_tag: Tagged line object.
    """

    def __init__(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
        priority: int,
    ) -> None:
        """Inits the LineBoundaryCondition class.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line.
            point2: Point location (``x``, ``y``) of the end of the line.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
            priority: Integer denoting the order in which the boundary condition gets
                applied.
        """
        super().__init__(direction=direction, value=value, priority=priority)
        self.point1 = point1
        self.point2 = point2
        self.mesh_tag: TaggedLine

    def __repr__(self) -> str:
        """Override __repr__ method.

        Returns:
            String representation of the object.
        """
        try:
            tag = self.mesh_tag
        except AttributeError:
            tag = None

        return (
            f"BC Type: {self.__class__.__name__}, dir: {self.direction}, val: "
            f"{self.value}, mesh tag: {tag}"
        )

    def get_unique_nodes(self) -> list[int]:
        """Returns a list of unique node indexes along the line BC.

        Returns:
            List of unique node indexes along the line.
        """
        # get list of node indexes along line BC
        node_idxs = []

        # loop through all line elements along the line BC
        for line_el in self.mesh_tag.elements:
            # loop through all nodes that make up the line element
            for node_idx in line_el.node_idxs:
                # if we haven't encountered this node, add it to the list
                if node_idx not in node_idxs:
                    node_idxs.append(node_idx)

        return node_idxs


class LineSupport(LineBoundaryCondition):
    """Class for adding supports along a line.

    Attributes:
        mesh_tag: Tagged line object.
    """

    def __init__(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineSupport class.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line.
            point2: Point location (``x``, ``y``) of the end of the line.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(
            point1=point1, point2=point2, direction=direction, value=value, priority=2
        )

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get degrees of freedom for node indexes
        dofs = dof_map(node_idxs=self.get_unique_nodes())

        # get relevant dofs
        dofs = self.get_dofs_given_direction(dofs=dofs)

        # apply bc - TODO - confirm this theory!
        for dof in dofs:
            k[dof, :] = 0
            k[dof, dof] = 1
            f[dof] = self.value

        return k, f


class LineSpring(LineBoundaryCondition):
    """Class for adding springs along a line.

    Attributes:
        mesh_tag: Tagged line object.
    """

    def __init__(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineSpring class.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line.
            point2: Point location (``x``, ``y``) of the end of the line.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(
            point1=point1, point2=point2, direction=direction, value=value, priority=1
        )

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition.

        Args:
            k: Stiffness matrix.
            f: Load vector.

        Returns:
            Modified stiffness matrix and load vector (``k``, ``f``).
        """
        # get degrees of freedom for node indexes
        dofs = dof_map(node_idxs=self.get_unique_nodes())

        # get relevant dofs
        dofs = self.get_dofs_given_direction(dofs=dofs)

        # apply bc - TODO - confirm this theory!
        for dof in dofs:
            k[dof, dof] += self.value

        return k, f


class LineLoad(LineBoundaryCondition):
    """Class for adding a load to a line.

    Attributes:
        mesh_tag: Tagged line object.
    """

    def __init__(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
    ) -> None:
        """Inits the LineLoad class.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line.
            point2: Point location (``x``, ``y``) of the end of the line.
            direction: Direction of the boundary condition, ``"x"``, ``"y"`` or
                ``"xy"``.
            value: Value of the boundary condition.
        """
        super().__init__(
            point1=point1, point2=point2, direction=direction, value=value, priority=0
        )

    def apply_bc(
        self,
        k: npt.NDArray[np.float64],
        f: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Applies the boundary condition."""
        # loop through all line elements
        for element in self.mesh_tag.elements:
            # get element load vector
            f_el = element.element_load_vector(
                direction=self.direction, value=self.value
            )

            # get element degrees of freedom
            el_dofs = dof_map(node_idxs=element.node_idxs)

            # add element load vector to global load vector
            f[el_dofs] += f_el

        return k, f
