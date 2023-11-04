"""planestress post-processor plotting functions."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from planestress.pre.boundary_condition import NodeLoad, NodeSupport


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


def plot_node_loads(
    ax: matplotlib.axes.Axes,
    node_loads: list[NodeLoad],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
) -> None:
    """Plots the nodal loads.

    Args:
        ax: Axis to plot on.
        node_loads: List of ``NodeLoad`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
            Defaults to ``False``.
        bc_fmt: Boundary condition text formatting string. Defaults to ``".3e"``.
    """
    # max arrow length and width
    max_arrow_length = 0.1 * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = 0.003 * max_dim

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

        # plot load
        ax.arrow(
            x=x, y=y, dx=dx, dy=dy, width=width, length_includes_head=True, color="r"
        )

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{node_load.value:>{bc_fmt}}", xy=(x + dx, y + dy), color="r"
            )


def plot_node_supports(
    ax: matplotlib.axes.Axes,
    node_supports: list[NodeSupport],
    max_dim: float,
    bc_text: bool,
    bc_fmt: str,
) -> None:
    """Plots the nodal supports.

    Args:
        ax: Axis to plot on.
        node_supports: List of ``NodeSupport`` objects.
        max_dim: Maximum dimension of the geometry bounding box.
        bc_text: If set to ``True``, plots the values of the boundary conditions.
            Defaults to ``False``.
        bc_fmt: Boundary condition text formatting string. Defaults to ``".3e"``.
    """
    # split into fixed supports and imposed displacements and get max displacement
    node_displacements: list[NodeSupport] = []
    node_fixed_supports: list[NodeSupport] = []
    max_disp = 0.0

    for node_support in node_supports:
        if abs(node_support.value) > 0:
            node_displacements.append(node_support)
            max_disp = max(max_disp, abs(node_support.value))
        else:
            node_fixed_supports.append(node_support)

    # plot imposed displacements
    # max arrow length and width
    max_arrow_length = 0.1 * max_dim
    min_arrow_length = 0.2 * max_arrow_length
    width = 0.003 * max_dim

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

        # plot load
        ax.arrow(
            x=x, y=y, dx=dx, dy=dy, width=width, length_includes_head=True, color="b"
        )

        # plot load text
        if bc_text:
            ax.annotate(
                text=f"{node_disp.value:>{bc_fmt}}", xy=(x + dx, y + dy), color="b"
            )

    # plot supports
    # triangle coordinates
    dx = 0.03 * max_dim  # scaling factor
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

        # plot hinge
        ax.plot(x, y, "ko", markerfacecolor="w", linewidth=1, markersize=4)
