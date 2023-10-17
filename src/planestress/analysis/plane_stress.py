"""Class for a planestress analysis."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable

import matplotlib
import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import planestress.analysis.solver as solver
from planestress.analysis.finite_element import FiniteElement, Tri3, Tri6
from planestress.analysis.utils import dof_map
from planestress.post.plotting import plotting_context
from planestress.post.results import Results


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.geometry import Geometry
    from planestress.pre.load_case import LoadCase
    from planestress.pre.mesh import Mesh


class PlaneStress:
    """Class for a plane-stress analysis."""

    def __init__(
        self,
        geometry: Geometry,
        load_cases: list[LoadCase],
        int_points: int = 3,
    ) -> None:
        """Inits the PlaneStress class."""
        self.geometry = geometry
        self.load_cases = load_cases
        self.int_points = int_points

        # initialise other class variables
        self.elements: list[FiniteElement] = []

        # check mesh has been created
        if self.geometry.mesh is None:
            raise RuntimeError(
                "No mesh detected, run Geometry.create_mesh() before creating a "
                "PlaneStress object."
            )

        self.mesh: Mesh = self.geometry.mesh

        # get finite element type
        el_type: type = Tri3 if self.mesh.linear else Tri6

        # loop through each element in the mesh
        for idx, node_idxs in enumerate(self.mesh.elements):
            # create a list containing the vertex and mid-node coordinates
            coords = self.mesh.nodes[node_idxs, :].transpose()

            # get attribute index of current element
            att_el = self.mesh.attributes[idx]

            # fetch the material
            material = self.geometry.materials[att_el]

            # add element to the list of elements
            self.elements.append(
                el_type(
                    el_idx=idx, coords=coords, node_idxs=node_idxs, material=material
                )
            )

    def solve(self) -> list[Results]:
        """Solves each load case."""
        # get number of degrees of freedom
        num_dofs = self.mesh.num_nodes() * 2

        # allocate stiffness matrix and load vector
        k = np.zeros((num_dofs, num_dofs))
        f = np.zeros(num_dofs)

        # allocate results
        results: list[Results] = []

        # assemble stiffness matrix
        for el in self.elements:
            # get element stiffness matrix
            k_el = el.element_stiffness_matrix(n_points=self.int_points)

            # get element degrees of freedom
            el_dofs = dof_map(node_idxs=el.node_idxs)
            el_dofs_mat = np.ix_(el_dofs, el_dofs)

            # add element stiffness matrix to global stiffness matrix
            k[el_dofs_mat] += k_el

        # for each load case
        for lc in self.load_cases:
            # initialise modified stiffness matrix
            k_mod = copy.deepcopy(k)

            # assemble load vector
            for el in self.elements:
                # get element load vector
                f_el = el.element_load_vector(n_points=self.int_points)

                # get element degrees of freedom
                el_dofs = dof_map(node_idxs=el.node_idxs)

                # add element load vector to global load vector
                f[el_dofs] += f_el

            # apply boundary conditions
            for bc in lc.boundary_conditions:
                # get node index of current boundary condition - TODO: for segments?
                node_idx = self.mesh.node_markers.index(bc.marker_id)

                # get degrees of freedom for node index
                dofs = dof_map(node_idxs=[node_idx])

                # apply boundary condition
                k_mod, f = bc.apply_bc(k=k_mod, f=f, dofs=dofs)

            # solve system
            u = solver.solve_direct(k=k_mod, f=f)

            # post-processing
            res = Results(num_nodes=self.mesh.num_nodes(), u=u)
            res.calculate_node_forces(k=k)
            res.calculate_element_results(elements=self.elements)

            # add to results list
            results.append(res)

        return results

    def plot_displacement_contour(
        self,
        results: Results,
        direction: str,
        title: str | None = None,
        cmap: str = "coolwarm",
        normalize: bool = True,
        fmt: str = "{x:.4e}",
        colorbar_label: str = "Displacement",
        alpha: float = 0.2,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the displacement contours."""
        # get displacement values
        if direction == "x":
            u = results.ux
        elif direction == "y":
            u = results.uy
        elif direction == "xy":
            u = results.uxy
        else:
            raise ValueError(f"direction must be 'x', 'y' or 'xy', not {direction}.")

        # apply title
        if not title:
            title = f"Displacement Contours [{direction}]"

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            colormap = matplotlib.colormaps.get_cmap(cmap=cmap)

            # create triangulation
            triang = Triangulation(
                self.mesh.nodes[:, 0],
                self.mesh.nodes[:, 1],
                self.mesh.elements[:, 0:3],
            )

            if normalize:
                norm = CenteredNorm()
            else:
                norm = None

            trictr = ax.tricontourf(triang, u, cmap=colormap, norm=norm)

            # display the colorbar
            divider = make_axes_locatable(axes=ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)

            fig.colorbar(
                mappable=trictr,
                label=colorbar_label,
                format=fmt,
                cax=cax,
            )

            # plot the finite element mesh
            self.mesh.plot_mesh(
                material_list=self.geometry.materials,
                nodes=False,
                nd_num=False,
                el_num=False,
                materials=False,
                mask=None,
                alpha=alpha,
                title=title,
                **dict(kwargs, ax=ax),
            )

        return ax

    def plot_deformed_shape(
        self,
        results: Results,
        displacement_scale: float,
        title: str | None = None,
        alpha: float = 0.2,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the deformed shape."""
        # apply title
        if not title:
            title = f"Deformed Shape [ds = {displacement_scale}]"

        return self.mesh.plot_mesh(
            material_list=self.geometry.materials,
            nodes=False,
            nd_num=False,
            el_num=False,
            materials=False,
            mask=None,
            alpha=alpha,
            title=title,
            ux=results.ux * displacement_scale,
            uy=results.uy * displacement_scale,
            **kwargs,
        )

    def plot_stress(
        self,
        results: Results,
        stress: str,
        title: str | None = None,
        cmap: str = "coolwarm",
        stress_limits: tuple[float, float] | None = None,
        normalize: bool = True,
        fmt: str = "{x:.4e}",
        colorbar_label: str = "Stress",
        alpha: float = 0.5,
        # material_list: list[Material] | None = None, # TODO
        agg_func: Callable[[list[float]], float] = np.average,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the stress contours."""
        # get required variables for stress plot
        stress_dict = {
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

        # populate stresses and plotted material groups
        sigs = results.get_nodal_stresses(agg_func=agg_func)[
            :, int(stress_dict[stress]["idx"])
        ]

        # apply title
        if not title:
            title = str(stress_dict[stress]["title"])

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (fig, ax):
            assert ax

            # set up the colormap
            colormap = matplotlib.colormaps.get_cmap(cmap=cmap)

            # create triangulation
            triang = Triangulation(
                self.mesh.nodes[:, 0],
                self.mesh.nodes[:, 1],
                self.mesh.elements[:, 0:3],
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

            trictr = ax.tricontourf(triang, sigs, v, cmap=colormap, norm=norm)

            # display the colorbar
            divider = make_axes_locatable(axes=ax)
            cax = divider.append_axes(position="right", size="5%", pad=0.1)

            fig.colorbar(
                mappable=trictr,
                label=colorbar_label,
                format=fmt,
                ticks=ticks,
                cax=cax,
            )

            # plot the finite element mesh
            self.mesh.plot_mesh(
                material_list=self.geometry.materials,
                nodes=False,
                nd_num=False,
                el_num=False,
                materials=False,
                mask=None,
                alpha=alpha,
                title=title,
                **dict(kwargs, ax=ax),
            )

        return ax
