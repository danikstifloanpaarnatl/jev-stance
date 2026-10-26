"""StanceJEV: CoVer-style stance detection with typed JEV choice decisions."""

from .types import StanceLabel, Prediction, BatchItem
from .pipeline import StanceJEVPipeline

__version__ = "0.1.0"
__all__ = [
    "StanceLabel",
    "Prediction",
    "BatchItem",
    "StanceJEVPipeline",
    "__version__",
]
