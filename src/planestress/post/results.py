"""Classes for storing ``planestress`` results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from planestress.analysis.utils import dof_map


if TYPE_CHECKING:
    from planestress.analysis.finite_element import ElementResults, FiniteElement


@dataclass
class Results:
    """Class for plane-stress results."""

    num_nodes: int
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

        k original before mod
        """
        self.f = k @ self.u

    def calculate_element_results(self, elements: list[FiniteElement]) -> None:
        """Calculates and stores the element results."""
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
        """Gets a list of the nodal stresses."""
        # allocate list of nodal results
        sigs = np.zeros((self.num_nodes, 3))
        sigs_res = [[] for _ in range(self.num_nodes)]

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
