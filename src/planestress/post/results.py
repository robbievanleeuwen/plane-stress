"""Class for storing ``planestress`` results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Results:
    """Class for plane-stress results."""

    u: npt.NDArray[np.float64]
