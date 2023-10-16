"""Class for a planestress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import planestress.analysis.solver as solver
from planestress.analysis.finite_element import FiniteElement, Tri3, Tri6


if TYPE_CHECKING:
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

    def solve(self) -> None:
        """Solves each load case."""
        # for each load case - TODO: only assemble k & f once
        for lc in self.load_cases:
            # allocate stiffness matrix and load vector
            num_dofs = self.mesh.num_nodes() * 2
            k = np.zeros((num_dofs, num_dofs))
            f = np.zeros(num_dofs)

            # assemble stiffness matrix and load vector
            for el in self.elements:
                # get element stiffness matrix and load vector
                k_el = el.element_stiffness_matrix(n_points=self.int_points)
                f_el = el.element_load_vector(n_points=self.int_points)

                # get element degrees of freedom
                el_dofs = self.dof_map(node_idxs=el.node_idxs)
                el_dofs_mat = np.ix_(el_dofs, el_dofs)

                # add element results to global
                k[el_dofs_mat] += k_el
                f[el_dofs] += f_el

            # apply boundary conditions
            for bc in lc.boundary_conditions:
                # get node index of current boundary condition - TODO: for segments?
                node_idx = self.mesh.node_markers.index(bc.marker_id)
                dofs = self.dof_map(node_idxs=[node_idx])
                k, f = bc.apply_bc(k=k, f=f, dofs=dofs)

            # solve
            u = solver.solve_direct(k=k, f=f)

            from rich.pretty import pprint

            pprint(u)
