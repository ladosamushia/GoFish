"""Input table parsing for tracer number densities and biases.

The input file is a whitespace-delimited table whose first line is a header
beginning with ``#``. Required columns are ``zmin``, ``zmax``, and matching
``nz_*`` / ``bias_*`` pairs for each tracer. An optional ``volume`` column
overrides the volume computed from the input cosmology.

Example::

    # zmin  zmax  nz_BGS  bias_BGS  nz_LRG  bias_LRG  volume
    0.00    0.10  1165.0  1.34      0.0     1.7       0.04
    ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

NULL_NBAR = 1.0e-30  # numerical floor for empty (tracer, z) cells


@dataclass
class TracerData:
    """Per-redshift, per-tracer number densities and linear biases.

    Attributes
    ----------
    zmin, zmax : np.ndarray
        Redshift bin edges, shape ``(nz,)``.
    nz : np.ndarray
        Galaxies per square-degree per dz, shape ``(n_tracers, nz)``.
        Becomes ``nbar`` in :math:`(h/{\\rm Mpc})^3` after :meth:`convert_nbar`.
    bias : np.ndarray
        Linear bias per tracer per redshift, shape ``(n_tracers, nz)``.
    volume : np.ndarray
        Optional comoving volume override in Gpc^3/h^3, shape ``(nz,)``.
        Negative entries mean "use the cosmology-derived volume."
    nbar : np.ndarray | None
        Set by :meth:`convert_nbar`. Same shape as ``nz``, units (h/Mpc)^3.
    tracer_names : list[str]
        Names of the tracers (e.g. ``["BGS", "LRG", "ELG", "QSO"]``).
    """

    zmin: np.ndarray
    zmax: np.ndarray
    nz: np.ndarray
    bias: np.ndarray
    volume: np.ndarray
    tracer_names: list[str]
    nbar: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str | Path) -> "TracerData":
        """Read a whitespace-delimited table; column names from the ``#`` header."""
        path = Path(path)
        with path.open() as f:
            header = f.readline().lstrip("#").strip().split()
        df = pd.read_csv(path, sep=r"\s+", names=header, comment="#", header=None)

        zmin = df["zmin"].to_numpy(dtype=float)
        zmax = df["zmax"].to_numpy(dtype=float)

        nz_cols = [c for c in df.columns if c.startswith("nz_")]
        bias_cols = [c for c in df.columns if c.startswith("bias_")]
        tracer_names = [c[len("nz_"):] for c in nz_cols]
        if {c[len("bias_"):] for c in bias_cols} != set(tracer_names):
            raise ValueError(
                f"Mismatched tracer columns. nz_* gives {tracer_names}, "
                f"bias_* gives {[c[5:] for c in bias_cols]}"
            )

        nz = np.array([df[f"nz_{t}"].to_numpy(dtype=float) for t in tracer_names])
        bias = np.array([df[f"bias_{t}"].to_numpy(dtype=float) for t in tracer_names])

        if "volume" in df.columns:
            volume = df["volume"].to_numpy(dtype=float)
        else:
            volume = -np.ones(len(df))

        data = cls(zmin, zmax, nz, bias, volume, tracer_names)
        data._mask_empty_bins()
        return data

    # ------------------------------------------------------------------ #
    def _mask_empty_bins(self) -> None:
        """Where a tracer has zero galaxies in a bin, set bias=0 and nz to a
        tiny floor. This zeroes out information from that (tracer, z) cell
        without dividing by zero downstream.
        """
        empty = self.nz == 0
        self.bias[empty] = 0.0
        self.nz[empty] = NULL_NBAR

    def convert_nbar(self, volume_mpch3: np.ndarray, skyarea_sqdeg: float) -> None:
        """Convert ``nz`` (galaxies / sq.deg. / dz) to number density in (h/Mpc)^3.

        Parameters
        ----------
        volume_mpch3 : np.ndarray
            Comoving volume per redshift bin in :math:`({\\rm Mpc}/h)^3`.
        skyarea_sqdeg : float
            Survey footprint in square degrees.
        """
        dz = self.zmax - self.zmin
        self.nbar = skyarea_sqdeg * self.nz * (dz / volume_mpch3)

    def scale_bias_by_growth(self, growth: np.ndarray) -> None:
        """Rescale bias so that b * D(z) is constant (i.e. ``b -> b / D(z)``)."""
        self.bias = self.bias / growth

    @property
    def n_tracers(self) -> int:
        return self.bias.shape[0]
