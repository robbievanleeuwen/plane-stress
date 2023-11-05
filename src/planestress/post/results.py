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
    from scipy.sparse import csc_array

    from planestress.analysis.finite_elements.finite_element import FiniteElement
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
        f_r: Calculated nodal reactions.
        element_results: List of ``ElementResults`` objects.
    """

    plane_stress: PlaneStress
    u: npt.NDArray[np.float64]
    ux: npt.NDArray[np.float64] = field(init=False)
    uy: npt.NDArray[np.float64] = field(init=False)
    uxy: npt.NDArray[np.float64] = field(init=False)
    f: npt.NDArray[np.float64] = field(init=False)
    f_r: npt.NDArray[np.float64] = field(init=False)
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
        k: csc_array,
    ) -> None:
        """Calculates and stores the resultant nodal forces.

        Args:
            k: Original stiffness matrix (before modification).
        """
        self.f = k @ self.u

    def calculate_reactions(
        self,
        f: npt.NDArray[np.float64],
    ) -> None:
        """Calculates and stores the nodal reactions.

        Args:
            f: Applied force vector.
        """
        self.f_r = self.f - f

    def calculate_stresses(
        self,
        elements: list[FiniteElement],
    ) -> None:
        """Calculates and stores the element stresses in ``self.element_results``.

        Args:
            elements: List of ``FiniteElement`` objects used during the analysis.
        """
        # initialise list of ElementResults
        self.element_results = []

        for el in elements:
            # get element degrees of freedom
            el_dofs = dof_map(node_idxs=tuple(el.node_idxs))

            # get ElementResults object and store
            el_res = el.calculate_element_stresses(u=self.u[el_dofs])
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
            for idx, node_idx in enumerate(el.node_idxs):
                sigs_res[node_idx].append(el.sigs[idx])

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

    def get_principal_stresses(
        self,
        agg_func: Callable[[list[float]], float] = np.average,
    ) -> npt.NDArray[np.float64]:
        r"""Calculates the prinicipal stresses at the nodes.

        Args:
            agg_func: A function that aggregates the stresses if the point is shared by
                several elements. The function must receive a list of stresses and
                return a single stress. Defaults to ``np.average``.

        Returns:
            A :class:`numpy.ndarray` of the principal nodal stresses of size
            ``[n x 3]``, where ``n`` is the number of nodes in the mesh. The columns
            consist of the two principal stress components and the principal stress
            angle: (:math:`\sigma_{11}`, :math:`\sigma_{22}`, :math:`\theta_p`).
        """
        # allocate principal stress
        sigs_p = np.zeros((self.plane_stress.mesh.num_nodes(), 3))

        # get raw nodal stresses
        sigs = self.get_nodal_stresses(agg_func=agg_func)

        # calculate principal stresses and angle
        for idx, sig in enumerate(sigs):
            # unpack stresses
            sig_xx, sig_yy, sig_xy = sig

            # calculate principal stresses
            sig_m = 0.5 * (sig_xx + sig_yy)
            sig_v = np.sqrt(0.25 * (sig_xx - sig_yy) ** 2 + sig_xy**2)
            sig_11 = sig_m + sig_v
            sig_22 = sig_m - sig_v
            theta_p = 0.5 * np.arctan2(2 * sig_xy, sig_xx - sig_yy)

            # store principal stress
            sigs_p[idx, 0] = sig_11
            sigs_p[idx, 1] = sig_22
            sigs_p[idx, 2] = theta_p

        return sigs_p

    def get_von_mises_stresses(
        self,
        agg_func: Callable[[list[float]], float] = np.average,
    ) -> npt.NDArray[np.float64]:
        r"""Calculates the von Mises stress at the nodes.

        Args:
            agg_func: A function that aggregates the stresses if the point is shared by
                several elements. The function must receive a list of stresses and
                return a single stress. Defaults to ``np.average``.

        Returns:
            A :class:`numpy.ndarray` of the von Mises nodal stresses.
        """
        # allocate von  stress
        sigs_vm = np.zeros(self.plane_stress.mesh.num_nodes())

        # get nodal stresses
        sigs = self.get_nodal_stresses(agg_func=agg_func)

        # calculate von Mises stresses
        for idx, sig in enumerate(sigs):
            # unpack stresses
            sig_xx, sig_yy, sig_xy = sig

            # calculate von mises stress
            sigs_vm[idx] = np.sqrt(
                sig_xx**2 - sig_xx * sig_yy + sig_yy**2 + 3 * sig_xy**2
            )

        return sigs_vm

    def get_tresca_stresses(
        self,
        agg_func: Callable[[list[float]], float] = np.average,
    ) -> npt.NDArray[np.float64]:
        r"""Calculates the Tresca stress at the nodes.

        Args:
            agg_func: A function that aggregates the stresses if the point is shared by
                several elements. The function must receive a list of stresses and
                return a single stress. Defaults to ``np.average``.

        Returns:
            A :class:`numpy.ndarray` of the Tresca nodal stresses.
        """
        # allocate von  stress
        sigs_t = np.zeros(self.plane_stress.mesh.num_nodes())

        # get principal stresses
        sigs_p = self.get_principal_stresses(agg_func=agg_func)

        # calculate Tresca stresses
        for idx, sig_p in enumerate(sigs_p):
            # unpack stresses
            sig_11, sig_22, _ = sig_p

            # calculate von mises stress
            sigs_t[idx] = abs(sig_11 - sig_22)

        return sigs_t

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
                ``CenteredNorm`` effectively places the center of the colormap at zero
                displacement. Defaults to ``True``.
            num_levels (int): Number of contour levels. Defaults to ``11``.
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
        num_levels: int = kwargs.pop("num_levels", 11)
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
                self.plane_stress.mesh.triangulation,
            )

            # determine min. and max. displacements
            u_min = min(u) - 1e-12
            u_max = max(u) + 1e-12

            v = np.linspace(start=u_min, stop=u_max, num=num_levels, endpoint=True)

            if np.isclose(v[0], v[-1], atol=1e-12):
                v = num_levels
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
            title=title,
            materials=False,
            node_indexes=False,
            element_indexes=False,
            alpha=alpha,
            ux=self.ux * displacement_scale,
            uy=self.uy * displacement_scale,
            **kwargs,
        )

    def plot_stress(
        self,
        stress: str,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Generates a stress contour plot.

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
                ``CenteredNorm`` effectively places the center of the colormap at zero
                displacement. Defaults to ``True``.
            num_levels (int): Number of contour levels. Defaults to ``11``.
            colorbar_format (str):  Number formatting string for stresses, see
                https://docs.python.org/3/library/string.html. Defaults to
                ``"{x:.4e}"``, i.e. exponential format with 4 decimal places.
            colorbar_label (str): Colorbar label. Defaults to ``"Stress"``.
            alpha (float):  Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.2``.
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

            Below is a list of the acceptable values for ``stress``:

            - ``"xx"`` - stress in the `x` direction.
            - ``"yy"`` - stress in the `y` direction.
            - ``"xy"`` - shear stress in the `xy` direction.
            - ``11`` - maximum principal stress.
            - ``22`` - minimum principal stress.
            - ``vm`` - von Mises stress.
            - ``tr`` - Tresca stress.
            - TODO others.
        """
        # dictionary of stresses
        stress_dict: dict[str, dict[str, str | int]] = {
            "xx": {
                "idx": 0,
                "title": r"Stress Contour Plot - $\sigma_{xx}$",
            },
            "yy": {
                "idx": 1,
                "title": r"Stress Contour Plot - $\sigma_{yy}$",
            },
            "xy": {
                "idx": 2,
                "title": r"Stress Contour Plot - $\sigma_{xy}$",
            },
            "11": {
                "idx": 0,
                "title": r"Stress Contour Plot - $\sigma_{11}$",
            },
            "22": {
                "idx": 1,
                "title": r"Stress Contour Plot - $\sigma_{22}$",
            },
            "vm": {
                "idx": -1,
                "title": r"Stress Contour Plot - $\sigma_{vM}$",
            },
            "tr": {
                "idx": -1,
                "title": r"Stress Contour Plot - $\sigma_{Tr}$",
            },
        }

        stress_func_dict: dict[str, Callable[[Any], npt.NDArray[np.float64]]] = {
            "xx": self.get_nodal_stresses,
            "yy": self.get_nodal_stresses,
            "xy": self.get_nodal_stresses,
            "11": self.get_principal_stresses,
            "22": self.get_principal_stresses,
            "vm": self.get_von_mises_stresses,
            "tr": self.get_tresca_stresses,
        }

        # get keyword arguments
        contours: bool = kwargs.pop("contours", False)
        colormap: str = kwargs.pop("colormap", "coolwarm")
        stress_limits: tuple[float, float] | None = kwargs.pop("stress_limits", None)
        normalize: bool = kwargs.pop("normalize", True)
        num_levels: int = kwargs.pop("num_levels", 11)
        colorbar_format: str = kwargs.pop("colorbar_format", "{x:.4e}")
        colorbar_label: str = kwargs.pop("colorbar_label", "Stress")
        alpha: float = kwargs.pop("alpha", 0.2)
        agg_func: Callable[[list[float]], float] = kwargs.pop("agg_func", np.average)

        # TODO - implement material_list -> see sectionproperties

        # populate stresses and plotted material groups
        try:
            # get stress index and function
            stress_idx = int(stress_dict[stress]["idx"])
            stress_func = stress_func_dict[stress]

            # if stress func resturns a 1D array
            if stress_idx == -1:
                sigs = stress_func(agg_func)
            # if stress func resturns a 2D array
            else:
                sigs = stress_func(agg_func)[:, stress_idx]
        except KeyError as exc:
            raise ValueError(
                f"{stress} is not a valid value for 'stress'. Refer to the "
                f"documentation for acceptable values."
            ) from exc

        # get title (ensure stress is valid first!)
        title: str = kwargs.pop("title", str(stress_dict[stress]["title"]))

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            cmap = matplotlib.colormaps.get_cmap(cmap=colormap)

            # create triangulation
            triang = Triangulation(
                self.plane_stress.mesh.nodes[:, 0],
                self.plane_stress.mesh.nodes[:, 1],
                self.plane_stress.mesh.triangulation,
            )

            # determine minimum and maximum stress values for the contour list
            if stress_limits is None:
                sig_min = min(sigs) - 1e-12
                sig_max = max(sigs) + 1e-12
            else:
                sig_min = stress_limits[0]
                sig_max = stress_limits[1]

            v = np.linspace(start=sig_min, stop=sig_max, num=num_levels, endpoint=True)

            if np.isclose(v[0], v[-1], atol=1e-12):
                v = num_levels
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

    def plot_principal_stress_vectors(
        self,
        stress: str,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Generates a vector plot of the principal stresses.

        Args:
            stress: Stress value to plot. See below for list of values.
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to ``"Stress Vector Plot - {stress}"``.
            colormap (str): Matplotlib color map, see
                https://matplotlib.org/stable/tutorials/colors/colormaps.html for more
                detail. Defaults to ``"coolwarm"``.
            stress_limits (tuple[float, float] | None): Custom stress limits
                (``sig_min``, ``sig_max``). Values outside these limits will not be
                plotted. Defaults to ``None``.
            num_levels (int): Number of contour levels. Defaults to ``11``.
            colorbar_format (str):  Number formatting string for stresses, see
                https://docs.python.org/3/library/string.html. Defaults to
                ``"{x:.4e}"``, i.e. exponential format with 4 decimal places.
            colorbar_label (str): Colorbar label. Defaults to ``"Stress"``.
            alpha (float):  Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.2``.
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

            Below is a list of the acceptable values for ``stress``:

            - ``11`` - maximum principal stress.
            - ``22`` - minimum principal stress.
            - TODO 11 and 22 together.
        """
        # dictionary of stresses
        title_dict: dict[str, str] = {
            "11": r"Stress Vector Plot - $\sigma_{11}$",
            "22": r"Stress Vector Plot - $\sigma_{22}$",
        }

        # get keyword arguments
        try:
            title: str = kwargs.pop("title", str(title_dict[stress]))
        except KeyError as exc:
            raise ValueError(
                f"{stress} is not a valid value for 'stress'. Refer to the "
                f"documentation for acceptable values."
            ) from exc

        colormap: str = kwargs.pop("colormap", "coolwarm")
        stress_limits: tuple[float, float] | None = kwargs.pop("stress_limits", None)
        num_levels: int = kwargs.pop("num_levels", 11)
        colorbar_format: str = kwargs.pop("colorbar_format", "{x:.4e}")
        colorbar_label: str = kwargs.pop("colorbar_label", "Stress")
        alpha: float = kwargs.pop("alpha", 0.2)
        agg_func: Callable[[list[float]], float] = kwargs.pop("agg_func", np.average)

        # get principal stresses
        sigs_p = self.get_principal_stresses(agg_func=agg_func)
        sigs_x = np.zeros(len(sigs_p))
        sigs_y = np.zeros(len(sigs_p))
        signs = np.zeros(len(sigs_p))

        # get stress index and angle increment
        stress_idx = 0 if stress == "11" else 1
        delta_theta = 0 if stress == "11" else np.pi / 2

        # break into x and y component
        for idx, sig_p in enumerate(sigs_p):
            # get stress and angle
            sig = sig_p[stress_idx]
            theta = sig_p[2] + delta_theta

            # if we are outside the stress limits assign zero stress
            if stress_limits and (sig < stress_limits[0] or sig > stress_limits[1]):
                sigs_x[idx] = 0.0
                sigs_y[idx] = 0.0
                signs[idx] = 0.0
            else:
                sigs_x[idx] = sig * np.cos(theta)
                sigs_y[idx] = sig * np.sin(theta)
                signs[idx] = 1 if sig > 0 else -1

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            cmap = matplotlib.colormaps.get_cmap(cmap=colormap)

            # scale the color with respect to the magnitude of the vector
            c_unsigned = np.hypot(sigs_x, sigs_y)
            c = [c_i * sign for c_i, sign in zip(c_unsigned, signs)]
            c_min = min(c)
            c_max = max(c)

            quiv = ax.quiver(
                self.plane_stress.mesh.nodes[:, 0],
                self.plane_stress.mesh.nodes[:, 1],
                sigs_x,
                sigs_y,
                c,
                cmap=cmap,
                pivot="mid",
                angles="xy",
                headwidth=0,
                headlength=0,
                headaxislength=0,
            )

            # display the colorbar
            v = np.linspace(
                start=c_min - 1e-12, stop=c_max + 1e-12, num=num_levels, endpoint=True
            )
            divider = make_axes_locatable(axes=ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)

            fig.colorbar(
                mappable=quiv,
                label=colorbar_label,
                format=colorbar_format,
                ticks=v,
                cax=cax,
            )

            # plot the finite element mesh
            self.plane_stress.mesh.plot_mesh(
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


class ElementResults:
    """Class for storing the results of a finite element.

    Attributes:
        sigs_vm: von Mises stresses for each node in the element.
    """

    def __init__(
        self,
        el_idx: int,
        node_idxs: list[int],
        sigs: npt.NDArray[np.float64],
    ) -> None:
        """Inits the ElementResults class.

        Args:
            el_idx: Element index.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2, idx3]``.
            sigs: Raw nodal stresses, e.g.
                ``[[sigxx_1, sigyy_1, sigxy_1], ..., [sigxx_3, sigyy_3, sigxy_3]]``.
        """
        self.el_idx = el_idx
        self.node_idxs = node_idxs
        self.sigs = sigs
