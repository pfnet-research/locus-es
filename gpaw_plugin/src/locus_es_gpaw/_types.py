"""Array type aliases used throughout the package."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
IntArray = npt.NDArray[np.integer[Any]]
BoolArray = npt.NDArray[np.bool_]
