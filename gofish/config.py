"""Configuration handling for GoFish forecasts.

The original code used a ``configobj.ConfigObj`` dict directly, which meant every
consumer had to remember to call ``float(pardict["kmin"])`` or
``pardict.as_bool("BAO_only")``. We hide that here behind a typed dataclass.

Existing ``*.ini`` files are still accepted via :meth:`ForecastConfig.from_ini`.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ForecastConfig:
    """All parameters needed to run a Fisher forecast.

    Cosmology parameters mirror the names CAMB expects, so the same keys work.
    Exactly one of ``H0``, ``h``, or ``thetastar`` must be set.
    Exactly one of ``A_s`` or ``ln10_As`` (i.e. ``ln10^{10}A_s``) must be set.
    """

    # Power spectrum amplitude (give one of these)
    A_s: Optional[float] = None
    ln10_As: Optional[float] = None  # corresponds to "ln10^{10}A_s" in INI

    # Other cosmology
    n_s: float = 0.9649
    H0: Optional[float] = None
    h: Optional[float] = None
    thetastar: Optional[float] = None
    omega_b: float = 0.02237
    omega_cdm: float = 0.12
    tau_reio: float = 0.0544
    Sum_mnu: float = 0.06
    Omega_k: float = 0.0
    nu_hierarchy: str = "degenerate"
    w0_fld: Optional[float] = None

    # Forecasting parameters
    kmin: float = 0.001
    kmax: float = 0.5
    skyarea: float = 14000.0  # square degrees
    scale_bias: bool = True
    BAO_only: bool = True
    fast_integration: bool = True  # was ``GoFast`` in the old code

    # I/O
    inputfile: Path = field(default_factory=lambda: Path("input.txt"))
    outputfile: Path = field(default_factory=lambda: Path("output"))

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ini(cls, path: str | Path) -> "ForecastConfig":
        """Load from an INI file in the legacy GoFish format.

        Unlike the original ``ConfigObj`` parser, ``configparser`` requires a
        section header, so we add a synthetic ``[gofish]`` section before
        parsing. This keeps the existing config files in the repo working.
        """
        text = Path(path).read_text()
        parser = ConfigParser(inline_comment_prefixes=("#", ";"))
        parser.read_string("[gofish]\n" + text)
        raw = dict(parser.items("gofish"))

        # Translate legacy keys
        if "ln10^{10}a_s" in raw:
            raw["ln10_as"] = raw.pop("ln10^{10}a_s")
        if "gofast" in raw:
            raw["fast_integration"] = raw.pop("gofast")

        # configparser lowercases keys; map back to dataclass field names
        fieldmap = {f.lower(): f for f in cls.__dataclass_fields__}
        kwargs: dict = {}
        for key, value in raw.items():
            if key not in fieldmap:
                continue  # silently ignore unknown keys
            value = value.strip().strip("'").strip('"')
            field_name = fieldmap[key]
            field_type = cls.__dataclass_fields__[field_name].type
            kwargs[field_name] = _coerce(value, field_type)

        config = cls(**kwargs)
        config._validate()
        return config

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.A_s is None and self.ln10_As is None:
            raise ValueError("Either A_s or ln10_As must be given.")
        if self.A_s is None:
            self.A_s = float(np.exp(self.ln10_As) / 1.0e10)
        if self.H0 is None and self.h is None and self.thetastar is None:
            raise ValueError("Either H0, h, or thetastar must be given.")
        if self.H0 is None and self.h is not None:
            self.H0 = 100.0 * self.h


def _coerce(value: str, target_type) -> object:
    """Best-effort string-to-type conversion for INI values."""
    # `target_type` may be a string (from `from __future__ import annotations`)
    # or an actual type. Handle the common cases.
    t = target_type if isinstance(target_type, str) else getattr(target_type, "__name__", "")

    if "bool" in t:
        return value.lower() in {"1", "true", "yes", "on"}
    if "int" in t and "Optional" not in t:
        return int(value)
    if "float" in t:
        return float(value)
    if "Path" in t:
        return Path(value)
    if "Optional[float]" in t or "float" in t:
        try:
            return float(value)
        except ValueError:
            return value
    return value


# Imported here to avoid circular issues; `_validate` uses numpy when ln10_As path is hit.
import numpy as np  # noqa: E402
