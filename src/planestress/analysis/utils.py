"""planestress utility functions."""

from __future__ import annotations


def dof_map(node_idxs: list[int]) -> list[int]:
    """Maps a list of node indexes to a list of degrees of freedom.

    Args:
        node_idxs: Node indexes to map.

    Returns:
        Global degrees of freedom for each node index in ``node_idxs``.
    """
    dofs = []

    for node_idx in node_idxs:
        dofs.extend([node_idx * 2, node_idx * 2 + 1])

    return dofs
