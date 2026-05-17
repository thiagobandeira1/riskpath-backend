"""Shared singletons + FastAPI dependency providers.

The ReadmissionPredictor is expensive to build (~3 s — loads the V7 parquet
to fit the LabelEncoders). One instance is shared across the whole process
via lru_cache + pre-warmed at startup by `app.main:lifespan`.

Usage in routers:

    from typing import Annotated
    from fastapi import Depends
    from app.dependencies import get_predictor
    from src.inference import ReadmissionPredictor

    @router.post(...)
    def handler(predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)]):
        proba = predictor.predict_proba(df)
        ...
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.inference import ReadmissionPredictor

# Attaches to uvicorn's logger so our startup line uses the same INFO format
# as uvicorn's own "Started server process" / "Application startup complete" lines.
_logger = logging.getLogger("uvicorn")


@lru_cache(maxsize=1)
def get_predictor() -> ReadmissionPredictor:
    """Return the process-wide ReadmissionPredictor.

    First call builds + warms it. Subsequent calls return the cached instance.
    """
    _logger.info("Building ReadmissionPredictor (loading model + fitting encoders)...")
    p = ReadmissionPredictor()
    _logger.info("Predictor ready.")
    return p
