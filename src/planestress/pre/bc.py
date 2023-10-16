"""Classes describing a planestress boundary conditions."""

from __future__ import annotations


class BoundaryCondition:
    """Abstract base class for a boundary condition."""

    pass


class NodeBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition at a node."""

    def __init__(
        self,
        x: float,
        y: float,
        dir: str,
        value: float,
        exact: bool,
    ) -> None:
        """Inits the NodeBoundaryCondition class."""
        self.x = x
        self.y = y
        self.dir = dir
        self.value = value
        self.exact = exact


class NodeSupport(NodeBoundaryCondition):
    """Class for adding a support to a node."""

    def __init__(
        self,
        x: float,
        y: float,
        dir: str,
        value: float,
        exact: bool = True,
    ) -> None:
        """Inits the NodeSupport class."""
        super().__init__(x=x, y=y, dir=dir, value=value, exact=exact)


class NodeSpring(NodeBoundaryCondition):
    """Class for adding a spring to a node."""

    pass


class NodeLoad(NodeBoundaryCondition):
    """Class for adding a load to a node."""

    pass


class LineBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition along a line."""

    pass
