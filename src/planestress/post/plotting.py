"""planestress post-processor plotting functions."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


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
