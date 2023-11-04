"""Class describing a planestress load case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import planestress.post.plotting as plotting
import planestress.pre.boundary_condition as bc


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.mesh import Mesh


@dataclass
class LoadCase:
    """Class for a load case.

    Args:
        boundary_conditions: List of boundary conditions.
        acceleration_field: Acceleration field for the current load case (``a_x``,
            ``a_y``). Defaults to ``(0.0, 0.0)``.
    """

    boundary_conditions: list[bc.BoundaryCondition]
    acceleration_field: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Post init method to sort boundary conditions."""
        # sort boundary conditions
        self.boundary_conditions.sort(
            key=lambda boundary_condition: boundary_condition.priority
        )

    def reset_mesh_tags(self) -> None:
        """Reset mesh tags."""
        for boundary_condition in self.boundary_conditions:
            boundary_condition.mesh_tag = None

    def assign_mesh_tags(
        self,
        mesh: Mesh,
    ) -> None:
        """Assigns mesh tags to all boundary conditions in the load case.

        Args:
            mesh: ``Mesh`` object.

        Raises:
            ValueError: If there is an invalid boundary condition in a load case.
        """
        for boundary_condition in self.boundary_conditions:
            # if a mesh tag hasn't been assigned yet
            if boundary_condition.mesh_tag is None:
                # if the boundary condition relates to a node
                if isinstance(boundary_condition, bc.NodeBoundaryCondition):
                    boundary_condition.mesh_tag = mesh.get_tagged_node_by_coordinates(
                        x=boundary_condition.point[0],
                        y=boundary_condition.point[1],
                    )
                # if the boundary condition relates to a line
                elif isinstance(boundary_condition, bc.LineBoundaryCondition):
                    boundary_condition.mesh_tag = mesh.get_tagged_line_by_coordinates(
                        point1=boundary_condition.point1,
                        point2=boundary_condition.point2,
                    )
                else:
                    raise ValueError(
                        f"{boundary_condition} is not a valid boundary condition."
                    )

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        max_dim: float,
        bc_text: bool,
        bc_fmt: str,
    ) -> None:
        """Plots the boundary conditions.

        Args:
            ax: Axis to plot on.
            max_dim: Maximum dimension of the geometry bounding box.
            bc_text: If set to ``True``, plots the values of the boundary conditions.
                Defaults to ``False``.
            bc_fmt: Boundary condition text formatting string. Defaults to ``".3e"``.
        """
        # create list of each boundary condition type
        node_loads = []
        node_supports = []

        # loop through boundary conditions to fill out lists
        for boundary_condition in self.boundary_conditions:
            if isinstance(boundary_condition, bc.NodeLoad):
                node_loads.append(boundary_condition)
            elif isinstance(boundary_condition, bc.NodeSupport):
                node_supports.append(boundary_condition)

        # plot node loads
        plotting.plot_node_loads(
            ax=ax,
            node_loads=node_loads,
            max_dim=max_dim,
            bc_text=bc_text,
            bc_fmt=bc_fmt,
        )

        # plot node supports
        plotting.plot_node_supports(
            ax=ax,
            node_supports=node_supports,
            max_dim=max_dim,
            bc_text=bc_text,
            bc_fmt=bc_fmt,
        )


# TODO - add a persistent load case??
