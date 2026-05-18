"""GoFish: a Fisher forecast tool for galaxy clustering surveys.

Public API:
    >>> from gofish import run_forecast, ForecastConfig
    >>> config = ForecastConfig.from_ini("config/test.ini")
    >>> result = run_forecast(config)
    >>> print(result.combined_errors)

Or from the command line:
    $ gofish config/test.ini
"""

from .config import ForecastConfig
from .data import TracerData
from .cosmology import Cosmology
from .fisher import compute_fisher_matrix, renormalize_covariance
from .pipeline import run_forecast, ForecastResult

__all__ = [
    "ForecastConfig",
    "TracerData",
    "Cosmology",
    "compute_fisher_matrix",
    "renormalize_covariance",
    "run_forecast",
    "ForecastResult",
]

__version__ = "0.2.0"
