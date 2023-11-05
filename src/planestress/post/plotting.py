"""planestress post-processor plotting functions."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.patches import Polygon


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    import planestress.pre.boundary_condition as bc


@contextlib.contextmanager
def plotting_context(
    ax: matplotlib.axes.Axes | None = None,
    pause: bool = True,
    title: str = "",
    filename: str = "",
    render: bool = True,
    axis_index: int | tuple[int, int] | None = None,
    **kwargs: Any,
) -> Generator[
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | Any | None], None, None
]:
    """Executes code required to set up a matplotlib figure.

    Args:
        ax: Axes object on which to plot. Defaults to ``None``.
        pause: If set to ``True``, the figure pauses the script until the window is
            closed. If set to ``False``, the script continues immediately after the
            window is rendered. Defaults to ``True``.
        title: Plot title. Defaults to ``""``.
        filename: Pass a non-empty string or path to save the image. If this option is
            used, the figure is closed after the file is saved. Defaults to ``""``.
        render: If set to ``False``, the image is not displayed. This may be useful if
            the figure or axes will be embedded or further edited before being
            displayed. Defaults to ``True``.
        axis_index: If more than 1 axis is created by subplot, then this is the axis to
            plot on. This may be a tuple if a 2D array of plots is returned. The default
            value of ``None`` will select the top left plot.
        kwargs: Passed to :func:`matplotlib.pyplot.subplots`

    Raises:
        ValueError: ``axis_index`` is invalid

    Yields:
        Matplotlib figure and axes
    """
    if filename:
        render = False

    if ax is None:
        if not render or pause:
            plt.ioff()
        else:
            plt.ion()

        ax_supplied = False
        fig, ax = plt.subplots(**kwargs)

        try:
            if axis_index is None:
                axis_index = (0,) * ax.ndim  # type: ignore

            ax = ax[axis_index]  # type: ignore
        except (AttributeError, TypeError):
            pass  # only 1 axis, not an array
        except IndexError as exc:
            raise ValueError(
                f"axis_index={axis_index} is not compatible with arguments to "
                f"subplots: {kwargs}."
            ) from exc
    else:
        fig = ax.get_figure()  # type: ignore
        assert fig
        ax_supplied = True

        if not render:
            plt.ioff()

    yield fig, ax

    if ax is not None:
        ax.set_title(title)
        plt.tight_layout()
        ax.set_aspect("equal", anchor="C")

    # if no axes was supplied, finish the plot and return the figure and axes
    if ax_supplied:
        # if an axis was supplied, don't continue displaying or configuring the plot
        return

    if filename:
        fig.savefig(filename, dpi=fig.dpi)
        plt.close(fig)  # close the figure to free the memory
        return  # if the figure was to be saved, then don't show it also

    if render:
        if pause:
            plt.show()  # type: ignore
        else:
            plt.draw()
            plt.pause(0.001)


def plot_boundary_conditions(
    ax: matplotlib.axes.Axes,
    node_loads: list[bc.NodeLoad],
    line_loads: list[bc.LineLoad],
    node_supports: list[bc.NodeSupport],
    line_supports: list[bc.LineSupport],
    node_springs: list[bc.NodeSpring],
    line_springs: list[bc.LineSpring],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
    arrow_length_scale: float,
    arrow_width_scale: float,
    support_scale: float,
    num_supports: int,
    multi_polygon: shapely.MultiPolygon,
) -> None:
    """Plots the boundary conditions.

    Args:
        ax: Axis to plot on.
        node_loads: List of ``NodeLoad`` objects.
        line_loads: List of ``LineLoad`` objects.
        node_supports: List of ``NodeSupport`` objects.
        line_supports: List of ``LineSupport`` objects.
        node_springs: List of ``NodeSpring`` objects.
        line_springs: List of ``LineSpring`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
        bc_fmt: Boundary condition text formatting string.
        arrow_length_scale: Arrow length scaling factor.
        arrow_width_scale: Arrow width scaling factor.
        support_scale: Support scaling factor.
        num_supports: Number of line supports to plot internally.
        multi_polygon: ``MultiPolygon`` describing the geometry.
    """
    # plot node loads
    plot_node_loads(
        ax=ax,
        node_loads=node_loads,
        max_dim=max_dim,
        bc_text=bc_text,
        bc_fmt=bc_fmt,
        arrow_length_scale=arrow_length_scale,
        arrow_width_scale=arrow_width_scale,
        multi_polygon=multi_polygon,
    )

    # plot line loads
    plot_line_loads(
        ax=ax,
        line_loads=line_loads,
        max_dim=max_dim,
        bc_text=bc_text,
        bc_fmt=bc_fmt,
        arrow_length_scale=arrow_length_scale,
        arrow_width_scale=arrow_width_scale,
        multi_polygon=multi_polygon,
    )

    # plot node supports
    plot_node_supports(
        ax=ax,
        node_supports=node_supports,
        max_dim=max_dim,
        bc_text=bc_text,
        bc_fmt=bc_fmt,
        arrow_length_scale=arrow_length_scale,
        arrow_width_scale=arrow_width_scale,
        support_scale=support_scale,
        multi_polygon=multi_polygon,
    )

    # plot line supports
    plot_line_supports(
        ax=ax,
        line_supports=line_supports,
        max_dim=max_dim,
        bc_text=bc_text,
        bc_fmt=bc_fmt,
        arrow_length_scale=arrow_length_scale,
        arrow_width_scale=arrow_width_scale,
        support_scale=support_scale,
        num_supports=num_supports,
        multi_polygon=multi_polygon,
    )

    # plot node springs
    plot_node_springs(
        ax=ax,
        node_springs=node_springs,
        max_dim=max_dim,
        support_scale=support_scale,
    )

    # plot line springs
    plot_line_springs(
        ax=ax,
        line_springs=line_springs,
        max_dim=max_dim,
        support_scale=support_scale,
        num_supports=num_supports,
    )


def plot_node_loads(
    ax: matplotlib.axes.Axes,
    node_loads: list[bc.NodeLoad],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
    arrow_length_scale: float,
    arrow_width_scale: float,
    multi_polygon: shapely.MultiPolygon,
) -> None:
    """Plots the nodal loads.

    Args:
        ax: Axis to plot on.
        node_loads: List of ``NodeLoad`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
        bc_fmt: Boundary condition text formatting string.
        arrow_length_scale: Arrow length scaling factor.
        arrow_width_scale: Arrow width scaling factor.
        multi_polygon: ``MultiPolygon`` describing the geometry.
    """
    # max arrow length and width
    max_arrow_length = arrow_length_scale * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = arrow_width_scale * max_dim

    # determine maximum load
    max_load = 0.0

    for node_load in node_loads:
        max_load = max(max_load, abs(node_load.value))

    # plot each load
    for node_load in node_loads:
        # get length of the arrow
        arrow_length = max(
            abs(node_load.value) / max_load * max_arrow_length, min_arrow_length
        )
        arrow_length = node_load.value / abs(node_load.value) * arrow_length  # signed

        # calculate position and translations
        x = node_load.point[0]
        y = node_load.point[1]
        dx = arrow_length if node_load.direction in ["x", "xy"] else 0.0
        dy = arrow_length if node_load.direction in ["y", "xy"] else 0.0

        # check to see if arrow tip is in geometry or on boundary
        pt = shapely.Point(x + dx, y + dy)

        if multi_polygon.contains(pt) or multi_polygon.boundary.contains(pt):
            # push arrow outside
            x -= dx
            y -= dy
            outside = True
        else:
            outside = False

        # plot load
        ax.arrow(
            x=x, y=y, dx=dx, dy=dy, width=width, length_includes_head=True, color="k"
        )

        # ensure text is in right place
        if outside:
            dx = 0
            dy = 0

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{node_load.value:>{bc_fmt}}", xy=(x + dx, y + dy), color="k"
            )


def plot_line_loads(
    ax: matplotlib.axes.Axes,
    line_loads: list[bc.LineLoad],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
    arrow_length_scale: float,
    arrow_width_scale: float,
    multi_polygon: shapely.MultiPolygon,
) -> None:
    """Plots the line loads.

    Args:
        ax: Axis to plot on.
        line_loads: List of ``LineLoad`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
        bc_fmt: Boundary condition text formatting string.
        arrow_length_scale: Arrow length scaling factor.
        arrow_width_scale: Arrow width scaling factor.
        multi_polygon: ``MultiPolygon`` describing the geometry.
    """
    # max arrow length and width
    max_arrow_length = arrow_length_scale * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = arrow_width_scale * max_dim

    # determine maximum load
    max_load = 0.0

    for line_load in line_loads:
        max_load = max(max_load, abs(line_load.value))

    # plot each load
    for line_load in line_loads:
        # get length of the arrow
        arrow_length = max(
            abs(line_load.value) / max_load * max_arrow_length, min_arrow_length
        )
        arrow_length = line_load.value / abs(line_load.value) * arrow_length  # signed

        # calculate position and translations
        x1 = line_load.point1[0]
        y1 = line_load.point1[1]
        x2 = line_load.point2[0]
        y2 = line_load.point2[1]
        dx = arrow_length if line_load.direction in ["x", "xy"] else 0.0
        dy = arrow_length if line_load.direction in ["y", "xy"] else 0.0

        # check to see if arrow tip is in geometry or on boundary
        pt = shapely.Point(0.5 * (x1 + x2) + dx, 0.5 * (y1 + y2) + dy)

        if multi_polygon.contains(pt) or multi_polygon.boundary.contains(pt):
            # push arrow outside
            x1 -= dx
            y1 -= dy
            x2 -= dx
            y2 -= dy
            outside = True
        else:
            outside = False

        # plot line load
        ax.arrow(
            x=x1, y=y1, dx=dx, dy=dy, width=width, length_includes_head=True, color="k"
        )
        ax.arrow(
            x=x2, y=y2, dx=dx, dy=dy, width=width, length_includes_head=True, color="k"
        )

        # ensure line and text are in right place if arrow is pushed outside
        if outside:
            dx = 0
            dy = 0

        # plot line
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], color="k")

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{line_load.value:>{bc_fmt}}",
                xy=(0.5 * (x1 + x2) + dx, 0.5 * (y1 + y2) + dy),
                color="k",
            )


def plot_node_supports(
    ax: matplotlib.axes.Axes,
    node_supports: list[bc.NodeSupport],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
    arrow_length_scale: float,
    arrow_width_scale: float,
    support_scale: float,
    multi_polygon: shapely.MultiPolygon,
) -> None:
    """Plots the nodal supports.

    Args:
        ax: Axis to plot on.
        node_supports: List of ``NodeSupport`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
        bc_fmt: Boundary condition text formatting string.
        arrow_length_scale: Arrow length scaling factor.
        arrow_width_scale: Arrow width scaling factor.
        support_scale: Support scaling factor.
        multi_polygon: ``MultiPolygon`` describing the geometry.
    """
    # split into fixed supports and imposed displacements and get max displacement
    node_displacements: list[bc.NodeSupport] = []
    node_fixed_supports: list[bc.NodeSupport] = []
    max_disp = 0.0

    for node_support in node_supports:
        if abs(node_support.value) > 0:
            node_displacements.append(node_support)
            max_disp = max(max_disp, abs(node_support.value))
        else:
            node_fixed_supports.append(node_support)

    # plot imposed displacements
    # max arrow length and width
    max_arrow_length = arrow_length_scale * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = arrow_width_scale * max_dim

    for node_disp in node_displacements:
        # get length of the arrow
        arrow_length = max(
            abs(node_disp.value) / max_disp * max_arrow_length, min_arrow_length
        )
        arrow_length = node_disp.value / abs(node_disp.value) * arrow_length  # signed

        # calculate position and translations
        x = node_disp.point[0]
        y = node_disp.point[1]
        dx = arrow_length if node_disp.direction in ["x", "xy"] else 0.0
        dy = arrow_length if node_disp.direction in ["y", "xy"] else 0.0

        # check to see if arrow tip is in geometry or on boundary
        pt = shapely.Point(x + dx, y + dy)

        if multi_polygon.contains(pt) or multi_polygon.boundary.contains(pt):
            # push arrow outside
            x -= dx
            y -= dy
            outside = True
        else:
            outside = False

        # plot load
        ax.arrow(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            width=width,
            length_includes_head=True,
            color="k",
            linestyle="--",
            fill=False,
        )

        # ensure text is in right place
        if outside:
            dx = 0
            dy = 0

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{node_disp.value:>{bc_fmt}}", xy=(x + dx, y + dy), color="k"
            )

    # plot supports
    # triangle coordinates
    dx = support_scale * max_dim  # scaling factor
    h = np.sqrt(3) / 2
    triangle = np.array([[-h, -h, -h, 0, -h], [-1, 1, 0.5, 0, -0.5]]) * dx

    for node_fix in node_fixed_supports:
        # calculate position
        x = node_fix.point[0]
        y = node_fix.point[1]

        # determine rotation
        if node_fix.direction == "x":
            angle = 0.0
        elif node_fix.direction == "y":
            angle = np.pi / 2
        else:
            angle = np.pi / 2

        # rotation matrix
        s = np.sin(angle)
        c = np.cos(angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # plot rollers or pin extras
        if node_fix.direction in ["x", "y"]:
            line = np.array([[-1.1, -1.1], [-1, 1]]) * dx
            rot_line = rot_mat @ line
            ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, "k-", linewidth=1)
        else:
            rect = np.array([[-1.4, -1.4, -h, -h], [-1, 1, 1, -1]]) * dx
            rot_rect = rot_mat @ rect
            rot_rect[0, :] += x
            rot_rect[1, :] += y
            ax.add_patch(Polygon(rot_rect.transpose(), facecolor=(0.7, 0.7, 0.7)))

        # rotate triangle
        rot_triangle = rot_mat @ triangle

        # plot triangle
        ax.plot(rot_triangle[0, :] + x, rot_triangle[1, :] + y, "k-", linewidth=1)


def plot_line_supports(
    ax: matplotlib.axes.Axes,
    line_supports: list[bc.LineSupport],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
    arrow_length_scale: float,
    arrow_width_scale: float,
    support_scale: float,
    num_supports: int,
    multi_polygon: shapely.MultiPolygon,
) -> None:
    """Plots the line supports.

    Args:
        ax: Axis to plot on.
        line_supports: List of ``LineSupport`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
        bc_fmt: Boundary condition text formatting string.
        arrow_length_scale: Arrow length scaling factor.
        arrow_width_scale: Arrow width scaling factor.
        support_scale: Support scaling factor.
        num_supports: Number of line supports to plot internally.
        multi_polygon: ``MultiPolygon`` describing the geometry.
    """
    # split into fixed supports and imposed displacements and get max displacement
    line_displacements: list[bc.LineSupport] = []
    line_fixed_supports: list[bc.LineSupport] = []
    max_disp = 0.0

    for line_support in line_supports:
        if abs(line_support.value) > 0:
            line_displacements.append(line_support)
            max_disp = max(max_disp, abs(line_support.value))
        else:
            line_fixed_supports.append(line_support)

    # plot imposed displacements
    # max arrow length and width
    max_arrow_length = arrow_length_scale * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = arrow_width_scale * max_dim

    for line_disp in line_displacements:
        # get length of the arrow
        arrow_length = max(
            abs(line_disp.value) / max_disp * max_arrow_length, min_arrow_length
        )
        arrow_length = line_disp.value / abs(line_disp.value) * arrow_length  # signed

        # calculate position and translations
        x1 = line_disp.point1[0]
        y1 = line_disp.point1[1]
        x2 = line_disp.point2[0]
        y2 = line_disp.point2[1]
        dx = arrow_length if line_disp.direction in ["x", "xy"] else 0.0
        dy = arrow_length if line_disp.direction in ["y", "xy"] else 0.0

        # check to see if arrow tip is in geometry or on boundary
        pt = shapely.Point(0.5 * (x1 + x2) + dx, 0.5 * (y1 + y2) + dy)

        if multi_polygon.contains(pt) or multi_polygon.boundary.contains(pt):
            # push arrow outside
            x1 -= dx
            y1 -= dy
            x2 -= dx
            y2 -= dy
            outside = True
        else:
            outside = False

        # plot load
        ax.arrow(
            x=x1,
            y=y1,
            dx=dx,
            dy=dy,
            width=width,
            length_includes_head=True,
            color="k",
            linestyle="--",
            fill=False,
        )
        ax.arrow(
            x=x2,
            y=y2,
            dx=dx,
            dy=dy,
            width=width,
            length_includes_head=True,
            color="k",
            linestyle="--",
            fill=False,
        )

        # ensure line and text are in right place if arrow is pushed outside
        if outside:
            dx = 0
            dy = 0

        # plot line
        ax.plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], color="k", linestyle="--")

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{line_disp.value:>{bc_fmt}}",
                xy=(0.5 * (x1 + x2) + dx, 0.5 * (y1 + y2) + dy),
                color="k",
            )

    # plot supports
    # triangle coordinates
    dx = support_scale * max_dim  # scaling factor
    h = np.sqrt(3) / 2
    triangle = np.array([[-h, -h, -h, 0, -h], [-1, 1, 0.5, 0, -0.5]]) * dx

    for line_fix in line_fixed_supports:
        # determine rotation
        if line_fix.direction == "x":
            angle = 0.0
        elif line_fix.direction == "y":
            angle = np.pi / 2
        else:
            angle = np.pi / 2

        # rotation matrix
        s = np.sin(angle)
        c = np.cos(angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # rotate triangle
        rot_triangle = rot_mat @ triangle

        # plot supports
        num_plots = num_supports + 2
        x_len = line_fix.point2[0] - line_fix.point1[0]
        y_len = line_fix.point2[1] - line_fix.point1[1]

        for idx in range(num_plots):
            # calculate position
            x = line_fix.point1[0] + idx / (num_plots - 1) * x_len
            y = line_fix.point1[1] + idx / (num_plots - 1) * y_len

            # plot triangle
            ax.plot(rot_triangle[0, :] + x, rot_triangle[1, :] + y, "k-", linewidth=1)

            # plot rollers or pin extras
            if line_fix.direction in ["x", "y"]:
                line = np.array([[-1.1, -1.1], [-1, 1]]) * dx
                rot_line = rot_mat @ line
                ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, "k-", linewidth=1)
            else:
                rect = np.array([[-1.4, -1.4, -h, -h], [-1, 1, 1, -1]]) * dx
                rot_rect = rot_mat @ rect
                rot_rect[0, :] += x
                rot_rect[1, :] += y
                ax.add_patch(Polygon(rot_rect.transpose(), facecolor=(0.7, 0.7, 0.7)))


def plot_node_springs(
    ax: matplotlib.axes.Axes,
    node_springs: list[bc.NodeSpring],
    max_dim: float,
    support_scale: float,
) -> None:
    """Plots the nodal supports.

    Args:
        ax: Axis to plot on.
        node_springs: List of ``NodeSpring`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        support_scale: Support scaling factor.
    """
    # triangle coordinates
    dx = support_scale * max_dim  # scaling factor
    h = np.sqrt(3) / 2
    hs = h / 8
    triangle = (
        np.array([[-2 * h, -2 * h, -2 * h, -h, -2 * h], [-1, 1, 0.5, 0, -0.5]]) * dx
    )
    spring = (
        np.array(
            [
                [
                    -8 * hs,
                    -7 * hs,
                    -6 * hs,
                    -5 * hs,
                    -4 * hs,
                    -3 * hs,
                    -2 * hs,
                    -1 * hs,
                    0,
                ],
                [0, 0, -0.25, 0.25, -0.25, 0.25, -0.25, 0, 0],
            ]
        )
        * dx
    )

    for node_spring in node_springs:
        # calculate position
        x = node_spring.point[0]
        y = node_spring.point[1]

        # determine rotation
        if node_spring.direction == "x":
            angles = [0.0]
        elif node_spring.direction == "y":
            angles = [np.pi / 2]
        else:
            angles = [0.0, np.pi / 2]

        for angle in angles:
            # rotation matrix
            s = np.sin(angle)
            c = np.cos(angle)
            rot_mat = np.array([[c, -s], [s, c]])

            # plot rollers
            line = np.array([[-1.1 - h, -1.1 - h], [-1, 1]]) * dx
            rot_line = rot_mat @ line
            ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, "k-", linewidth=1)

            # rotate triangle and spring
            rot_triangle = rot_mat @ triangle
            rot_spring = rot_mat @ spring

            # plot triangle and spring
            ax.plot(rot_triangle[0, :] + x, rot_triangle[1, :] + y, "k-", linewidth=1)
            ax.plot(rot_spring[0, :] + x, rot_spring[1, :] + y, "k-", linewidth=1)


def plot_line_springs(
    ax: matplotlib.axes.Axes,
    line_springs: list[bc.LineSpring],
    max_dim: float,
    support_scale: float,
    num_supports: int,
) -> None:
    """Plots the nodal supports.

    Args:
        ax: Axis to plot on.
        line_springs: List of ``LineSpring`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        support_scale: Support scaling factor.
        num_supports: Number of line supports to plot internally.
    """
    # triangle coordinates
    dx = support_scale * max_dim  # scaling factor
    h = np.sqrt(3) / 2
    hs = h / 8
    triangle = (
        np.array([[-2 * h, -2 * h, -2 * h, -h, -2 * h], [-1, 1, 0.5, 0, -0.5]]) * dx
    )
    spring = (
        np.array(
            [
                [
                    -8 * hs,
                    -7 * hs,
                    -6 * hs,
                    -5 * hs,
                    -4 * hs,
                    -3 * hs,
                    -2 * hs,
                    -1 * hs,
                    0,
                ],
                [0, 0, -0.25, 0.25, -0.25, 0.25, -0.25, 0, 0],
            ]
        )
        * dx
    )

    for line_spring in line_springs:
        # determine rotation
        if line_spring.direction == "x":
            angles = [0.0]
        elif line_spring.direction == "y":
            angles = [np.pi / 2]
        else:
            angles = [0.0, np.pi / 2]

        for angle in angles:
            # rotation matrix
            s = np.sin(angle)
            c = np.cos(angle)
            rot_mat = np.array([[c, -s], [s, c]])

            # rotate triangle and spring
            rot_triangle = rot_mat @ triangle
            rot_spring = rot_mat @ spring

            # plot supports
            num_plots = num_supports + 2
            x_len = line_spring.point2[0] - line_spring.point1[0]
            y_len = line_spring.point2[1] - line_spring.point1[1]

            for idx in range(num_plots):
                # calculate position
                x = line_spring.point1[0] + idx / (num_plots - 1) * x_len
                y = line_spring.point1[1] + idx / (num_plots - 1) * y_len

                # plot rollers
                line = np.array([[-1.1 - h, -1.1 - h], [-1, 1]]) * dx
                rot_line = rot_mat @ line
                ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, "k-", linewidth=1)

                # plot triangle and spring
                ax.plot(
                    rot_triangle[0, :] + x, rot_triangle[1, :] + y, "k-", linewidth=1
                )
                ax.plot(rot_spring[0, :] + x, rot_spring[1, :] + y, "k-", linewidth=1)
