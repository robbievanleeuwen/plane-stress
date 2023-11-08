"""Class describing a planestress material."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt


@dataclass(eq=True, frozen=True)
class Material:
    """Class for a plane-stress material.

    The color can be a multitude of different formats, refer to
    https://matplotlib.org/stable/api/colors_api.html and
    https://matplotlib.org/stable/gallery/color/named_colors.html for more information.

    Args:
        name: Material name. Defaults to ``"default"``.
        elastic_modulus: Material modulus of elasticity. Defaults to ``1.0``.
        poissons_ratio: Material Poisson's ratio. Defaults to ``0.0``.
        thickness: Material thickness. Defaults to ``1.0``.
        density: Material density (mass per unit volume). Defaults to ``1.0``.
        color: Defaults to ``"w"``.
    """

    name: str = "default"
    elastic_modulus: float = 1.0
    poissons_ratio: float = 0.0
    thickness: float = 1.0
    density: float = 1.0
    color: str = "w"

    @cache
    def get_d_matrix(self) -> npt.NDArray[np.float64]:
        r"""Returns the constitutive matrix for plane-stress.

        The constitutive matrix (D) is defined as
        :math:`\boldsymbol{\sigma} = \textbf{D} \boldsymbol{\varepsilon}`.

        TODO - consider caching the result.

        Returns:
            Constitutive matrix.
        """
        d_mat = np.array(
            [
                [1, self.poissons_ratio, 0],
                [self.poissons_ratio, 1, 0],
                [0, 0, (1 - self.poissons_ratio) / 2],
            ]
        )

        return d_mat * self.elastic_modulus / (1 - self.poissons_ratio**2)


DEFAULT_MATERIAL = Material()
