"""Class describing a planestress material."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(eq=True)
class Material:
    """Class for a plane-stress material."""

    name: str = "default"
    elastic_modulus: float = 1.0
    poissons_ratio: float = 0.0
    thickness: float = 1.0
    density: float = 1.0
    colour: str = "w"

    @property
    def mu(self) -> float:
        """Returns Lame parameter mu."""
        return self.elastic_modulus / (2 * (1 + self.poissons_ratio))

    @property
    def lda(self) -> float:
        """Returns the Lame parameter lambda."""
        return (
            self.poissons_ratio
            * self.elastic_modulus
            / ((1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
        )

    def get_d_matrix(
        self,
    ) -> npt.NDArray:
        """Returns the D (constitutive) matrix for plane-stress.

        sig = D . eps

        Cache this!
        """
        mu = self.mu
        lda = self.lda

        d_mat = np.zeros((3, 3))  # allocate D matrix
        d_mat[0:2, 0:2] = lda + 2 * mu
        d_mat[2, 2] = mu
        d_mat[0, 1] = lda
        d_mat[1, 0] = lda

        return d_mat


DEFAULT_MATERIAL = Material()
