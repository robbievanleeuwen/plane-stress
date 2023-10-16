"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.tri import Triangulation
from shapely import MultiPoint, Point
from shapely.ops import nearest_points

from planestress.post.plotting import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.material import Material


@dataclass
class Mesh:
    """Class for a plane-stress mesh."""

    nodes: npt.NDArray[np.float64]
    elements: npt.NDArray[np.float64]
    attributes: list[int]
    node_markers: list[int]
    linear: bool

    def __post_init__(self) -> None:
        """Mesh post_init method."""
        self.multi_point = MultiPoint(points=self.nodes)
        self.node_list = self.nodes.tolist()

    def num_nodes(self) -> int:
        """Returns the number of nodes in the mesh."""
        return len(self.nodes)

    def get_node(
        self,
        x: float,
        y: float,
    ) -> int:
        """Returns the node index at the point (``x``, ``y``)."""
        try:
            return self.node_list.index([x, y])
        except ValueError as exc:
            raise ValueError(f"Cannot find node at x: {x}, y: {y}.") from exc

    def get_nearest_node(
        self,
        x: float,
        y: float,
    ) -> tuple[int, tuple[float, float]]:
        """Returns the nearest node index & coordinates to the point (``x``, ``y``)."""
        nd, _ = nearest_points(g1=self.multi_point, g2=Point(x, y))
        return self.node_list.index([nd.x, nd.y]), (nd.x, nd.y)

    def plot_mesh(
        self,
        material_list: list[Material],
        nodes: bool,
        nd_num: bool,
        el_num: bool,
        materials: bool,
        alpha: float,
        mask: list[bool] | None,
        title: str,
        ux: npt.NDArray[np.float64] | None = None,
        uy: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the finite element mesh."""
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax

            # add deformed shape
            if ux is not None:
                x = self.nodes[:, 0] + ux
            else:
                x = self.nodes[:, 0]

            if uy is not None:
                y = self.nodes[:, 1] + uy
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

            # node numbers
            if nd_num:
                for idx, pt in enumerate(self.nodes):
                    ax.annotate(str(idx), xy=(pt[0], pt[1]), color="r")

            # element numbers
            if el_num:
                for idx, el in enumerate(self.elements):
                    pt1 = self.nodes[el[0]]
                    pt2 = self.nodes[el[1]]
                    pt3 = self.nodes[el[2]]
                    x = (pt1[0] + pt2[0] + pt3[0]) / 3
                    y = (pt1[1] + pt2[1] + pt3[1]) / 3
                    ax.annotate(str(idx), xy=(x, y), color="b")

        return ax
