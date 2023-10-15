"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from matplotlib.tri import Triangulation
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
import numpy.typing as npt

from planestress.post.post import plotting_context

if TYPE_CHECKING:
    import matplotlib.axes
    from planestress.pre.material import Material

@dataclass
class Mesh:
    """Class for a plane-stress mesh."""

    nodes: npt.NDArray[np.float64]
    elements: npt.NDArray[np.int32]
    attributes: npt.NDArray[np.int32]

    def num_nodes(self) -> int:
        """Returns the number of nodes in the mesh."""
        return len(self.nodes)
    
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
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the finite element mesh."""
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax

            # create mesh triangulation
            triang = Triangulation(
                self.nodes[:, 0],
                self.nodes[:, 1],
                self.elements[:, 0:3],
                mask=mask,
            )

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


