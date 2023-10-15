"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Mesh:
    """Class for a plane-stress mesh."""

    nodes: list[tuple[float, float]]
    elements: list[list[int]]
    attributes: list[list[int]]
