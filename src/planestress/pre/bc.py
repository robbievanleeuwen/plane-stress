"""Classes describing a planestress boundary conditions."""

from __future__ import annotations


class BoundaryCondition:
    """Abstract base class for a boundary condition."""

    pass


class NodeBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition at a node."""

    pass


class NodeSupport(NodeBoundaryCondition):
    """Class for adding a support to a node."""

    pass


class NodeSpring(NodeBoundaryCondition):
    """Class for adding a spring to a node."""

    pass


class NodeLoad(NodeBoundaryCondition):
    """Class for adding a load to a node."""

    pass


class LineBoundaryCondition(BoundaryCondition):
    """Abstract base class for a boundary condition along a line."""

    pass
