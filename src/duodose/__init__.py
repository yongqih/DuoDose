"""DuoDose: homotypic-aware doublet detection for single-cell RNA-seq."""

from .api import DuoDose, detect_doublets
from .config import DLConfig, DuoDoseConfig, FeatureConfig, SemiRealConfig
from .models.registry import DEFAULT_DUODOSE_BACKEND, DUODOSE_BACKENDS
from .result import DuoDoseResult

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DUODOSE_BACKEND",
    "DLConfig",
    "DUODOSE_BACKENDS",
    "DuoDose",
    "DuoDoseConfig",
    "DuoDoseResult",
    "FeatureConfig",
    "SemiRealConfig",
    "detect_doublets",
    "__version__",
]
