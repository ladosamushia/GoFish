"""File output utilities for forecast results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_redshift_fisher(
    output_stem: str | Path,
    cov: np.ndarray,
    means: np.ndarray,
    redshift: float,
) -> tuple[Path, Path]:
    """Write the 3x3 ``[f sigma8, D_A, H]`` covariance and means for one bin.

    Writes two text files named ``{output_stem}_cov_{z:.2f}.txt`` and
    ``{output_stem}_data_{z:.2f}.txt``. Returns the two paths.
    """
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"_{redshift:.2f}.txt"
    cov_path = output_stem.with_name(output_stem.name + f"_cov{suffix}")
    data_path = output_stem.with_name(output_stem.name + f"_data{suffix}")
    np.savetxt(cov_path, cov[-3:, -3:])
    np.savetxt(data_path, means)
    return cov_path, data_path
