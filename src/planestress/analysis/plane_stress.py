"""Class for a planestress analysis."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

import planestress.analysis.solver as solver
import planestress.pre.boundary_condition as bc
from planestress.analysis.utils import dof_map
from planestress.post.results import Results


if TYPE_CHECKING:
    from planestress.pre.geometry import Geometry
    from planestress.pre.load_case import LoadCase
    from planestress.pre.mesh import Mesh


class PlaneStress:
    """Class for a plane-stress analysis.

    Attributes:
        geometry: ``Geometry`` object containing a meshed geometry.
        load_cases: List of load cases to analyse.
        mesh: ``Mesh`` object.
    """

    def __init__(
        self,
        geometry: Geometry,
        load_cases: list[LoadCase],
    ) -> None:
        """Inits the PlaneStress class.

        Args:
            geometry: ``Geometry`` object containing a meshed geometry.
            load_cases: List of load cases to analyse.

        Raises:
            RuntimeError: If there is no mesh in the ``Geometry`` object.
            ValueError: If there is an invalid boundary condition in a load case.
        """
        self.geometry = geometry
        self.load_cases = load_cases

        # check mesh has been created
        if len(self.geometry.mesh.nodes) < 1:
            raise RuntimeError(
                "No mesh detected, run Geometry.create_mesh() before creating a "
                "PlaneStress object."
            )

        self.mesh: Mesh = self.geometry.mesh

        # assign tagged items to boundary conditions
        for load_case in self.load_cases:
            for b in load_case.boundary_conditions:
                # if a mesh tag hasn't been assigned yet
                if not hasattr(b, "mesh_tag"):
                    # if the boundary condition relates to a node
                    if isinstance(b, bc.NodeBoundaryCondition):
                        b.mesh_tag = self.mesh.get_tagged_node_by_coordinates(
                            x=b.point[0],
                            y=b.point[1],
                        )
                    # if the boundary condition relates to a line
                    elif isinstance(b, bc.LineBoundaryCondition):
                        b.mesh_tag = self.mesh.get_tagged_line_by_coordinates(
                            point1=b.point1,
                            point2=b.point2,
                        )
                    else:
                        raise ValueError(f"{b} is not a valid boundary condition.")

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
        for el in self.mesh.elements:
            # get element stiffness matrix
            k_el = el.element_stiffness_matrix()

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
            for el in self.mesh.elements:
                # get element load vector
                f_el = el.element_load_vector()

                # get element degrees of freedom
                el_dofs = dof_map(node_idxs=el.node_idxs)

                # add element load vector to global load vector
                f[el_dofs] += f_el

            # apply boundary conditions
            for boundary_condition in lc.boundary_conditions:
                # apply boundary condition
                k_mod, f = boundary_condition.apply_bc(k=k_mod, f=f)

            # solve system
            u = solver.solve_direct(k=k_mod, f=f)

            # post-processing
            res = Results(plane_stress=self, u=u)
            res.calculate_node_forces(k=k)
            res.calculate_element_results(elements=self.mesh.elements)

            # add to results list
            results.append(res)

        return results
