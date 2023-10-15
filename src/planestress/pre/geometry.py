"""Class describing a planestress geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from planestress.post.post import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes
    from shapely import MultiPolygon, Polygon

    from planestress.pre.material import Material
    from planestress.pre.mesh import Mesh


class Geometry:
    """Class describing a geometric region."""

    def __init__(
        self,
        geoms: Polygon | MultiPolygon,
        materials: list[Material],
        tol: int = 12,
    ) -> None:
        """Inits the Geometry class."""
        # save input data
        self.tol = tol
        self.geoms = self.round_polygon_vertices(geoms=geoms)
        self.materials = materials

        # allocate points, facets, holes, control_points, mesh
        self.points: list[tuple[float, float]] = []
        self.facets: list[tuple[int, int]] = []
        self.holes: list[tuple[float, float]] = []
        self.control_points: list[tuple[float, float]] = []
        self.mesh: Mesh | None = None

        # compile the geometry into points, facets, holes and control_points
        self.compile_geometry()

    def round_polygon_vertices(
        self,
        geoms: Polygon | MultiPolygon,
    ) -> MultiPolygon:
        """Returns a ``MultiPolygon`` with its vertices rounded to ``tol``."""
        return None

    def compile_geometry(self) -> None:
        """Creates points, facets, holes and control_points from shapely geometry."""
        pass

    def plot_geometry(
        self,
        labels: tuple[str] = ("control_points",),
        title: str = "Geometry",
        cp: bool = True,
        legend: bool = True,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the geometry."""
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax

            # plot the points and facets
            label = "Points & Facets"
            for fct in self.facets:
                ax.plot(
                    [self.points[fct[0]][0], self.points[fct[1]][0]],
                    [self.points[fct[0]][1], self.points[fct[1]][1]],
                    "ko-",
                    markersize=2,
                    linewidth=1.5,
                    label=label,
                )
                label = None

            # plot the holes
            label = "Holes"
            for hl in self.holes:
                ax.plot(
                    hl[0], hl[1], "rx", markersize=5, markeredgewidth=1, label=label
                )
                label = None

            # plot the control points
            if cp:
                label = "Control Points"
                for cpts in self.control_points:
                    ax.plot(cpts[0], cpts[1], "bo", markersize=5, label=label)
                    label = None

            # display the labels
            for label in labels:
                # plot control_point labels
                if label == "control_points":
                    for idx, pt in enumerate(self.control_points):
                        ax.annotate(str(idx), xy=pt, color="b")

                # plot point labels
                if label == "points":
                    for idx, pt in enumerate(self.points):
                        ax.annotate(str(idx), xy=pt, color="r")

                # plot facet labels
                if label == "facets":
                    for idx, fct in enumerate(self.facets):
                        pt1 = self.points[fct[0]]
                        pt2 = self.points[fct[1]]
                        xy = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

                        ax.annotate(str(idx), xy=xy, color="b")

                # plot hole labels
                if label == "holes":
                    for idx, pt in enumerate(self.holes):
                        ax.annotate(str(idx), xy=pt, color="r")

            # display the legend
            if legend:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax
