"""treegptq: post-training quantization for fitted tree ensembles."""

from .api import Report, load, quantize, save, sweep
from .quant import METHODS

__version__ = "0.1.0"
__all__ = ["quantize", "sweep", "save", "load", "Report", "METHODS", "__version__"]
