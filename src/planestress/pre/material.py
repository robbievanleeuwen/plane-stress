"""Class describing a planestress material."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=True)
class Material:
    """Class for a plane-stress material."""

    name: str = "Default"
    elastic_modulus: float = 1.0
    poissons_ratio: float = 0.0
    thickness: float = 1.0
    density: float = 1.0
    colour: str = "w"


DEFAULT_MATERIAL = Material()
