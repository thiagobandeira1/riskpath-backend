"""numpy → JSON serialization helpers.

Pydantic v2 does not natively serialize numpy arrays. SHAP returns ndarray
output, so we centralize the conversion here (one helper instead of three
subtly-different inline implementations across endpoints).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def ndarray_to_list(arr: np.ndarray) -> Any:
    """Convert an ndarray to nested Python lists with NaN -> None.

    Pydantic / standard JSON cannot serialize NaN; replacing with None is the
    consumer-friendly choice (JSON `null` is the natural form of "missing").
    Inf is preserved as a float (Pydantic float validators accept it); guard
    upstream if your endpoint needs to reject Inf.
    """
    return _replace_nan(arr.tolist())


def _replace_nan(value: Any) -> Any:
    if isinstance(value, list):
        return [_replace_nan(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
