"""Classes for storing ``planestress`` results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.colors import CenteredNorm
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from planestress.analysis.utils import dof_map
from planestress.post.plotting import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.analysis.finite_element import ElementResults, FiniteElement
    from planestress.analysis.plane_stress import PlaneStress


@dataclass
class Results:
    """Class for plane-stress results.

    Args:
        plane_stress: ``PlaneStress`` analysis object used to generate these results.
        u: Displacement vector.

    Attributes:
        ux: ``x`` component of the displacement vector.
        uy: ``y`` component of the displacement vector.
        uxy: Resultant component of the displacement vector.
        f: Calculated nodal forces.
        element_results: List of ``ElementResults`` objects.
    """

    plane_stress: PlaneStress
    u: npt.NDArray[np.float64]
    ux: npt.NDArray[np.float64] = field(init=False)
    uy: npt.NDArray[np.float64] = field(init=False)
    uxy: npt.NDArray[np.float64] = field(init=False)
    f: npt.NDArray[np.float64] = field(init=False)
    element_results: list[ElementResults] = field(init=False)

    def __post_init__(self) -> None:
        """Post init method for the Results class."""
        self.partition_displacements()

    def partition_displacements(self) -> None:
        """Partitions the ``u`` vector into ``x`` and ``y`` displacement vectors."""
        self.ux = self.u[0::2]
        self.uy = self.u[1::2]
        self.uxy = (self.ux**2 + self.uy**2) ** 0.5

    def calculate_node_forces(
        self,
        k: npt.NDArray[np.float64],
    ) -> None:
        """Calculates and stores the resultant nodal forces.

        Args:
            k: Original stiffness matrix (before modification).
        """
        self.f = k @ self.u

    def calculate_element_results(self, elements: list[FiniteElement]) -> None:
        """Calculates and stores the element results in ``self.element_results``.

        Args:
            elements: List of ``FiniteElement`` objects used during the analysis.
        """
        # initialise list of ElementResults
        self.element_results = []

        for el in elements:
            # get element degrees of freedom
            el_dofs = dof_map(node_idxs=el.node_idxs)

            # get ElementResults object and store
            el_res = el.get_element_results(u=self.u[el_dofs])
            self.element_results.append(el_res)

    def get_nodal_stresses(
        self,
        agg_func: Callable[[list[float]], float] = np.average,
    ) -> npt.NDArray[np.float64]:
        """Calculates the nodal stresses.

        Args:
            agg_func: A function that aggregates the stresses if the point is shared by
                several elements. The function must receive a list of stresses and
                return a single stress. Defaults to ``np.average``.

        Returns:
            A :class:`numpy.ndarray` of the nodal stresses of size ``[n x 3]``, where
            ``n`` is the number of nodes in the mesh. The columns consist of the three
            stress components, (``sig_xx``, ``sig_yy``, ``sig_xy``).
        """
        # get number of nodes
        num_nodes = self.plane_stress.mesh.num_nodes()

        # allocate list of nodal results
        sigs = np.zeros((num_nodes, 3))
        sigs_res: list[list[list[float]]] = [[] for _ in range(num_nodes)]

        # loop through each element
        for el in self.element_results:
            # get nodal stresses for element
            # add each nodal result to list
            for node_idx in el.node_idxs:
                sigs_res[node_idx].append(el.sigs[0])

        # apply aggregation function
        for idx, node_res in enumerate(sigs_res):
            # unpack stresses
            sig_xx = [sig[0] for sig in node_res]
            sig_yy = [sig[1] for sig in node_res]
            sig_xy = [sig[2] for sig in node_res]

            # apply aggregation function
            sigs[idx][0] = agg_func(sig_xx)
            sigs[idx][1] = agg_func(sig_yy)
            sigs[idx][2] = agg_func(sig_xy)

        return sigs

    def plot_displacement_contour(
        self,
        direction: str,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the displacement contours.

        Args:
            direction: Displacement field to plot. May be either ``"x"``, ``"y"`` or
                ``"xy"`` (resultant displacement).
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to
                ``"Displacement Contours [{direction}]"``.
            contours (bool): If set to ``True``, plots contour lines. Defaults to
                ``False``.
            colormap (str): Matplotlib color map, see
                https://matplotlib.org/stable/tutorials/colors/colormaps.html for more
                detail. Defaults to ``"coolwarm"``.
            normalize (bool): If set to ``True``, ``CenteredNorm`` is used to scale the
                colormap, if set to False, the default linear scaling is used.
                ``CenteredNorm`` effectively places the centre of the colormap at zero
                displacement. Defaults to ``True``.
            colorbar_format (str):  Number formatting string for displacements, see
                https://docs.python.org/3/library/string.html. Defaults to
                ``"{x:.4e}"``, i.e. exponential format with 4 decimal places.
            colorbar_label (str): Colorbar label. Defaults to ``"Displacement"``.
            alpha (float):  Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.2``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Raises:
            ValueError: If the value for ``direction`` is not valid.

        Returns:
            Matplotlib axes object.

        Example:
            TODO.
        """
        # get keyword arguments
        title: str = kwargs.pop("title", f"Displacement Contours [{direction}]")
        contours: bool = kwargs.pop("contours", False)
        colormap: str = kwargs.pop("colormap", "coolwarm")
        normalize: bool = kwargs.pop("normalize", True)
        colorbar_format: str = kwargs.pop("colorbar_format", "{x:.4e}")
        colorbar_label: str = kwargs.pop("colorbar_label", "Displacement")
        alpha: float = kwargs.pop("alpha", 0.2)

        # get displacement values
        if direction == "x":
            u = self.ux
        elif direction == "y":
            u = self.uy
        elif direction == "xy":
            u = self.uxy
        else:
            raise ValueError(f"'direction' must be 'x', 'y' or 'xy', not {direction}.")

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            cmap = matplotlib.colormaps.get_cmap(cmap=colormap)

            # create triangulation
            triang = Triangulation(
                self.plane_stress.mesh.nodes[:, 0],
                self.plane_stress.mesh.nodes[:, 1],
                self.plane_stress.mesh.elements[:, 0:3],
            )

            # determine min. and max. displacements
            u_min = min(u) - 1e-12
            u_max = max(u) + 1e-12

            v = np.linspace(start=u_min, stop=u_max, num=15, endpoint=True)

            if np.isclose(v[0], v[-1], atol=1e-12):
                v = 15
                ticks = None
            else:
                ticks = v

            if normalize:
                norm = CenteredNorm()
            else:
                norm = None

            # plot the filled contours
            trictr = ax.tricontourf(triang, u, v, cmap=cmap, norm=norm)

            # plot the contour lines
            if contours:
                ax.tricontour(triang, u, colors="k", levels=v)

            # display the colorbar
            divider = make_axes_locatable(axes=ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)

            fig.colorbar(
                mappable=trictr,
                label=colorbar_label,
                format=colorbar_format,
                ticks=ticks,
                cax=cax,
            )

            # plot the finite element mesh
            self.plane_stress.mesh.plot_mesh(
                load_case=None,
                material_list=self.plane_stress.geometry.materials,
                title=title,
                materials=False,
                nodes=False,
                node_indexes=False,
                element_indexes=False,
                alpha=alpha,
                mask=None,
                **dict(kwargs, ax=ax),
            )

        return ax

    def plot_deformed_shape(
        self,
        displacement_scale: float,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the deformed shape.

        Args:
            displacement_scale: Displacement scale.
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to
                ``"Deformed Shape [ds = {displacement_scale}]"``.
            alpha (float):  Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.8``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Returns:
            Matplotlib axes object.

        Example:
            TODO.
        """
        # get keyword arguments
        title: str = kwargs.pop("title", f"Deformed Shape [ds = {displacement_scale}]")
        alpha: float = kwargs.pop("alpha", 0.8)

        return self.plane_stress.mesh.plot_mesh(
            load_case=None,
            material_list=self.plane_stress.geometry.materials,
            title=title,
            materials=False,
            nodes=False,
            node_indexes=False,
            element_indexes=False,
            alpha=alpha,
            mask=None,
            ux=self.ux * displacement_scale,
            uy=self.uy * displacement_scale,
            **kwargs,
        )

    def plot_stress(
        self,
        stress: str,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the stress contours.

        Args:
            stress: Stress value to plot. See below for list of values.
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to ``"Stress Contour Plot - {stress}"``.
            contours (bool): If set to ``True``, plots contour lines. Defaults to
                ``False``.
            colormap (str): Matplotlib color map, see
                https://matplotlib.org/stable/tutorials/colors/colormaps.html for more
                detail. Defaults to ``"coolwarm"``.
            stress_limits (tuple[float, float] | None): Custom colorbar stress limits
                (``sig_min``, ``sig_max``). Values outside these limits will appear as
                white. Defaults to ``None``.
            normalize (bool): If set to ``True``, ``CenteredNorm`` is used to scale the
                colormap, if set to False, the default linear scaling is used.
                ``CenteredNorm`` effectively places the centre of the colormap at zero
                displacement. Defaults to ``True``.
            colorbar_format (str):  Number formatting string for stresses, see
                https://docs.python.org/3/library/string.html. Defaults to
                ``"{x:.4e}"``, i.e. exponential format with 4 decimal places.
            colorbar_label (str): Colorbar label. Defaults to ``"Stress"``.
            alpha (float):  Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.5``.
            agg_func (Callable[[list[float]], float]): A function that aggregates the
                stresses if the point is shared by several elements. The function must
                receive a list of stresses and return a single stress. Defaults to
                ``np.average``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Raises:
            ValueError: If the value for ``stress`` is not valid.

        Returns:
            Matplotlib axes object.

        Example:
            TODO.

        .. admonition:: Stress plotting options

            Below is a list of the acceptable values for stress:

            -  ``"xx"`` - stress in the `x` direction.
            -  ``"yy"`` - stress in the `y` direction.
            -  ``"xy"`` - shear stress in the `xy` direction.
            - TODO others.
        """
        # dictionary of acceptable stresses
        stress_dict: dict[str, dict[str, str | int]] = {
            "xx": {
                "attribute": "sigs",
                "idx": 0,
                "title": r"Stress Contour Plot - $\sigma_{xx}$",
            },
            "yy": {
                "attribute": "sigs",
                "idx": 1,
                "title": r"Stress Contour Plot - $\sigma_{yy}$",
            },
            "xy": {
                "attribute": "sigs",
                "idx": 2,
                "title": r"Stress Contour Plot - $\sigma_{xy}$",
            },
        }

        # get keyword arguments
        title: str = kwargs.pop("title", str(stress_dict[stress]["title"]))
        contours: bool = kwargs.pop("contours", False)
        colormap: str = kwargs.pop("colormap", "coolwarm")
        stress_limits: tuple[float, float] | None = kwargs.pop("stress_limits", None)
        normalize: bool = kwargs.pop("normalize", True)
        colorbar_format: str = kwargs.pop("colorbar_format", "{x:.4e}")
        colorbar_label: str = kwargs.pop("colorbar_label", "Stress")
        alpha: float = kwargs.pop("alpha", 0.5)
        agg_func: Callable[[list[float]], float] = kwargs.pop("agg_func", np.average)

        # TODO - implement material_list -> see sectionproperties

        # populate stresses and plotted material groups
        try:
            stress_idx = int(stress_dict[stress]["idx"])
            sigs = self.get_nodal_stresses(agg_func=agg_func)[:, stress_idx]
        except KeyError as exc:
            raise ValueError(
                f"{stress} is not a valid value for 'stress'. Refer to the "
                f"documentation for acceptable values."
            ) from exc

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            cmap = matplotlib.colormaps.get_cmap(cmap=colormap)

            # create triangulation
            triang = Triangulation(
                self.plane_stress.mesh.nodes[:, 0],
                self.plane_stress.mesh.nodes[:, 1],
                self.plane_stress.mesh.elements[:, 0:3],
            )

            # determine minimum and maximum stress values for the contour list
            if stress_limits is None:
                sig_min = min(sigs) - 1e-12
                sig_max = max(sigs) + 1e-12
            else:
                sig_min = stress_limits[0]
                sig_max = stress_limits[1]

            v = np.linspace(start=sig_min, stop=sig_max, num=15, endpoint=True)

            if np.isclose(v[0], v[-1], atol=1e-12):
                v = 15
                ticks = None
            else:
                ticks = v

            if normalize:
                norm = CenteredNorm()
            else:
                norm = None

            # plot the filled contours
            trictr = ax.tricontourf(triang, sigs, v, cmap=cmap, norm=norm)

            # plot the contour lines
            if contours:
                ax.tricontour(triang, sigs, colors="k", levels=v)

            # display the colorbar
            divider = make_axes_locatable(axes=ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)

            fig.colorbar(
                mappable=trictr,
                label=colorbar_label,
                format=colorbar_format,
                ticks=ticks,
                cax=cax,
            )

            # plot the finite element mesh
            self.plane_stress.mesh.plot_mesh(
                load_case=None,
                material_list=self.plane_stress.geometry.materials,
                title=title,
                materials=False,
                nodes=False,
                node_indexes=False,
                element_indexes=False,
                alpha=alpha,
                mask=None,
                **dict(kwargs, ax=ax),
            )

        return ax
