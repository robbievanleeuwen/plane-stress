"""Class for a planestress analysis."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

import planestress.analysis.solver as solver
import planestress.pre.boundary_condition as bc
from planestress.analysis.finite_element import FiniteElement, Tri3, Tri6
from planestress.analysis.utils import dof_map
from planestress.post.results import Results


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
        """Inits the PlaneStress class.

        Args:
            geometry: ``Geometry`` object containing a meshed geometry.
            load_cases: List of load cases to analyse.
            int_points: Number of integration points to use. Defaults to ``3``.

        Raises:
            RuntimeError: If there is no mesh in the ``Geometry`` object.
        """
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
        """Solves each load case.

        Returns:
            A list of ``Results`` objects for post-processing corresponding to each load
            case.
        """
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

            # apply boundary conditions # TODO - Tri6 elements LineBC!
            for boundary_condition in lc.boundary_conditions:
                # get node indexes of current boundary condition
                # if we are a node boundary condition
                if isinstance(boundary_condition, bc.NodeBoundaryCondition):
                    # get index of the node the boundary condition is applied to
                    node_idxs = [
                        self.mesh.node_markers.index(boundary_condition.marker_id)
                    ]
                # otherwise we must be a line boundary condition
                else:
                    # get indexes of the segment the boundary condition is applied to
                    seg_idxs = [
                        idx
                        for idx, seg_marker in enumerate(self.mesh.segment_markers)
                        if seg_marker == boundary_condition.marker_id
                    ]

                    # get nodes indexes of segments
                    node_idxs = []

                    # loop through segment indexes
                    for seg_idx in seg_idxs:
                        seg = self.mesh.segments[seg_idx]

                        # loop through each node index in segment
                        for node_idx in seg:
                            if node_idx not in node_idxs:
                                node_idxs.append(node_idx)

                # get degrees of freedom for node indexes
                dofs = dof_map(node_idxs=node_idxs)

                # apply boundary condition
                k_mod, f = boundary_condition.apply_bc(k=k_mod, f=f, dofs=dofs)

            # solve system
            u = solver.solve_direct(k=k_mod, f=f)

            # post-processing
            res = Results(plane_stress=self, u=u)
            res.calculate_node_forces(k=k)
            res.calculate_element_results(elements=self.elements)

            # add to results list
            results.append(res)

        return results
