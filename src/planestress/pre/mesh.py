"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.tri import Triangulation
from shapely import Point, STRtree

from planestress.post.plotting import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.load_case import LoadCase
    from planestress.pre.material import Material


@dataclass
class Mesh:
    """Class for a plane-stress mesh.

    Args:
        nodes: List of nodes of the mesh, e.g. ``[[x1, y1], [x2, y2], ... ]``.
        elements: List of node indexes defining the elements in the mesh, e.g.
            ``[[idx1, idx2, idx3], [idx2, idx4, idx3], ... ]``.
        attributes: List of element attribute IDs.
        node_markers: List of node marker IDs.
        segments: List of node indexes defining the segments in the mesh, i.e. the edges
            that lie on the facets of the original geometry.
        segments_markers: List of segment marker IDs.
        linear: If ``True`` the mesh consists of 3-noded triangles, if ``False`` the
            mesh consists of 6-noded triangles.
        str_tree: A :class:`shapely.STRtree` of the nodes in the mesh.
    """

    nodes: npt.NDArray[np.float64]
    elements: npt.NDArray[np.int32]
    attributes: list[int]
    node_markers: list[int]
    segments: npt.NDArray[np.int32]
    segment_markers: list[int]
    linear: bool
    str_tree: STRtree = field(init=False)

    def __post_init__(self) -> None:
        """Mesh post_init method."""
        self.str_tree = STRtree(geoms=[Point(node[0], node[1]) for node in self.nodes])

    def num_nodes(self) -> int:
        """Returns the number of nodes in the mesh.

        Returns:
            Number of nodes in the mesh.
        """
        return len(self.nodes)

    def get_node(
        self,
        x: float,
        y: float,
    ) -> int:
        """Returns the node index at or nearest to the point (``x``, ``y``).

        Args:
            x: ``x`` location of the node to find.
            y: ``y`` location of the node to find.

        Returns:
            Index of the node closest to (``x``, ``y``).
        """
        idx = self.str_tree.nearest(geometry=Point(x, y))

        return cast(int, idx)

    def plot_mesh(
        self,
        load_case: LoadCase | None,
        material_list: list[Material],
        title: str,
        materials: bool,
        nodes: bool,
        node_indexes: bool,
        element_indexes: bool,
        alpha: float,
        mask: list[bool] | None,
        ux: npt.NDArray[np.float64] | None = None,
        uy: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the finite element mesh.

        Optionally also renders the boundary conditions of a load case if provided. Also
        plots a deformed mesh if ``ux`` and/or ``uy`` is provided. In this case, the
        undeformed mesh is also plotted with ``alpha=0.2``, ``materials`` is set to
        ``False`` and ``load_case`` is set to ``None``.

        Args:
            load_case: Plots the boundary conditions within a load case if provided.
                Defaults to ``None``.
            material_list: List of materials that correspond to the mesh attributes.
            title: Plot title.
            materials: If set to ``True`` shades the elements with the specified
                material colours.
            nodes: If set to ``True`` plots the nodes of the mesh.
            node_indexes: If set to ``True``, plots the indexes of each node.
            element_indexes: If set to ``True``, plots the indexes of each element.
            alpha: Transparency of the mesh outlines, :math:`0 \leq \alpha \leq 1`.
            mask: Mask array to mask out triangles, must be same length as number of
                elements in mesh.
            ux: Deformation component in the ``x`` direction. Defaults to ``None``.
            uy: Deformation component in the ``y`` direction. Defaults to ``None``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Returns:
            Matplotlib axes object.
        """
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax

            # add deformed shape
            if ux is not None:
                x = self.nodes[:, 0] + ux
                materials = False
                load_case = None
            else:
                x = self.nodes[:, 0]

            if uy is not None:
                y = self.nodes[:, 1] + uy
                materials = False
                load_case = None
            else:
                y = self.nodes[:, 1]

            # create mesh triangulation (add deformed shape)
            triang = Triangulation(x, y, self.elements[:, 0:3], mask=mask)

            # if displaying materials
            if materials:
                colour_array = []
                legend_labels = []
                c = []  # Indices of elements for mapping colors

                # create an array of finite element colors
                for idx, attr in enumerate(self.attributes):
                    colour_array.append(material_list[int(attr)].colour)
                    c.append(idx)

                # create a list of unique material legend entries
                for idx, material in enumerate(material_list):
                    # if the material has not be entered yet
                    if idx == 0 or material not in material_list[0:idx]:
                        # add the material color and name to the legend list
                        patch = Patch(color=material.colour, label=material.name)
                        legend_labels.append(patch)

                cmap = ListedColormap(colors=colour_array)  # custom colormap

                # plot the mesh colors
                ax.tripcolor(
                    triang,
                    c,
                    cmap=cmap,
                )

                # display the legend
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    handles=legend_labels,
                )

            # plot the mesh
            ax.triplot(
                triang,
                "ko-" if nodes else "k-",
                lw=0.5,
                alpha=alpha,
            )

            # if deformed shape, plot the original mesh
            if ux is not None or uy is not None:
                triang_orig = Triangulation(
                    self.nodes[:, 0], self.nodes[:, 1], self.elements[:, 0:3], mask=mask
                )

                ax.triplot(
                    triang_orig,
                    "ko-" if nodes else "k-",
                    lw=0.5,
                    alpha=0.2,
                )

            # node numbers
            if node_indexes:
                for idx, pt in enumerate(self.nodes):
                    ax.annotate(str(idx), xy=(pt[0], pt[1]), color="r")

            # element numbers
            if element_indexes:
                for idx, el in enumerate(self.elements):
                    pt1 = self.nodes[el[0]]
                    pt2 = self.nodes[el[1]]
                    pt3 = self.nodes[el[2]]
                    x = (pt1[0] + pt2[0] + pt3[0]) / 3
                    y = (pt1[1] + pt2[1] + pt3[1]) / 3
                    ax.annotate(str(idx), xy=(x, y), color="b")

            # plot the load case
            if load_case is not None:
                for boundary_condition in load_case.boundary_conditions:
                    # boundary_condition.plot()
                    print(boundary_condition.marker_id)  # TODO - plot this!

        return ax
