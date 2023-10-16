"""Class for a planestress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
from matplotlib.colors import CenteredNorm
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import planestress.analysis.solver as solver
from planestress.analysis.finite_element import FiniteElement, Tri3, Tri6
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

    @staticmethod
    def dof_map(node_idxs: list[int]) -> list[int]:
        """Maps a list of node indices to a list of degrees of freedom."""
        dofs = []

        for node_idx in node_idxs:
            dofs.extend([node_idx * 2, node_idx * 2 + 1])

        return dofs

    def solve(self) -> list[Results]:
        """Solves each load case."""
        # get number of degrees of freedom
        num_dofs = self.mesh.num_nodes() * 2

        # allocate stiffness matrix and load vector
        k = np.zeros((num_dofs, num_dofs))
        k_mod = np.zeros((num_dofs, num_dofs))
        f = np.zeros(num_dofs)

        # allocate results
        results: list[Results] = []

        # assemble stiffness matrix
        for el in self.elements:
            # get element stiffness matrix
            k_el = el.element_stiffness_matrix(n_points=self.int_points)

            # get element degrees of freedom
            el_dofs = self.dof_map(node_idxs=el.node_idxs)
            el_dofs_mat = np.ix_(el_dofs, el_dofs)

            # add element stiffness matrix to global stiffness matrix
            k[el_dofs_mat] += k_el

        # for each load case
        for lc in self.load_cases:
            # assemble load vector
            for el in self.elements:
                # get element load vector
                f_el = el.element_load_vector(n_points=self.int_points)

                # get element degrees of freedom
                el_dofs = self.dof_map(node_idxs=el.node_idxs)

                # add element load vector to global load vector
                f[el_dofs] += f_el

            # apply boundary conditions
            for idx, bc in enumerate(lc.boundary_conditions):
                # get node index of current boundary condition - TODO: for segments?
                node_idx = self.mesh.node_markers.index(bc.marker_id)

                # get degrees of freedom for node index
                dofs = self.dof_map(node_idxs=[node_idx])

                # if first boundary condition get unmodified k
                k_mod = k if idx == 0 else k_mod
                k_mod, f = bc.apply_bc(k=k_mod, f=f, dofs=dofs)

            # solve system
            u = solver.solve_direct(k=k, f=f)

            # add to results
            results.append(Results(u=u))

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
            u = results.u[0::2]
        elif direction == "y":
            u = results.u[1::2]
        elif direction == "xy":
            u = (results.u[0::2] ** 2 + results.u[1::2] ** 2) ** 0.5
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
            ux=results.u[0::2] * displacement_scale,
            uy=results.u[1::2] * displacement_scale,
            **kwargs,
        )
