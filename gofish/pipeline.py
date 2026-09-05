"""High-level forecast pipeline.

This module replaces the procedural ``if __name__ == '__main__':`` block in the
original ``GoFish.py``. It exposes :func:`run_forecast` which takes a config
and returns a structured :class:`ForecastResult`. The CLI is a thin wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import ForecastConfig
from .cosmology import Cosmology
from .data import TracerData
from .fisher import (
    compute_fisher_matrix,
    compute_reconstruction_factor,
    precompute_alpha_derivatives,
    renormalize_covariance,
)
from .io import write_redshift_fisher


# --------------------------------------------------------------------------- #
# Result containers                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class RedshiftBinResult:
    """Forecast results for one redshift bin."""

    z: float
    volume_gpc3: float
    means: np.ndarray            # [f*sigma8, D_A (Mpc), H (km/s/Mpc)]
    cov_renormalised: np.ndarray  # full (n_tracers+3) x (n_tracers+3) matrix
    errors_pct: np.ndarray       # 1-sigma errors on means, in percent
    iso_alpha_err_pct: float     # error on isotropic alpha, in percent
    is_empty: bool = False


@dataclass
class ForecastResult:
    """Per-bin and combined forecast outputs."""

    bin_results: list[RedshiftBinResult] = field(default_factory=list)
    combined: Optional[RedshiftBinResult] = None
    full_fisher: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
def run_forecast(
    config: ForecastConfig,
    *,
    data: TracerData | None = None,
    cosmo: Cosmology | None = None,
    write_outputs: bool = True,
    verbose: bool = True,
) -> ForecastResult:
    """Run the full Fisher forecast and (optionally) write outputs to disk.

    The caller can pre-supply ``data`` and ``cosmo`` for testing, otherwise
    they are built from ``config``.

    Returns
    -------
    ForecastResult
        Per-bin and combined results.
    """
    if data is None:
        data = TracerData.from_file(config.inputfile)
    if cosmo is None:
        cosmo = Cosmology.from_config(config, data.zmin, data.zmax)

    # Override with input-file volumes if provided
    if np.any(data.volume > 0):
        cosmo.volume = data.volume * 1.0e9  # Gpc^3 -> Mpc^3

    data.convert_nbar(cosmo.volume, config.skyarea)

    if config.scale_bias:
        data.scale_bias_by_growth(cosmo.growth)

    if verbose:
        print("#  Data nbar")
        print(data.nbar)
        print("#  Data bias")
        print(data.bias)

    recon = compute_reconstruction_factor(cosmo, data)
    if verbose:
        print("#  Data recon factor")
        print(recon)

    der_alpha = precompute_alpha_derivatives(cosmo, BAO_only=config.BAO_only)
    der_alpha_bao = precompute_alpha_derivatives(cosmo, BAO_only=True)

    n_tracers = data.n_tracers
    n_z = len(cosmo.z)
    full_fisher = np.zeros((n_z * n_tracers + 3, n_z * n_tracers + 3))

    if verbose:
        print(
            "#  z  V(Gpc/h)^3  fsigma8  fsigma8_err(%)  "
            "Da(Mpc/h)  Da_err(%)  H(km/s/Mpc)  H_err(%)   alpha_err(%)"
        )

    result = ForecastResult()

    for iz in range(n_z):
        bin_result = _process_redshift_bin(
            iz, cosmo, data, recon, der_alpha, der_alpha_bao,
            config, full_fisher,
        )
        result.bin_results.append(bin_result)

        if verbose:
            _print_bin_result(bin_result)

        if write_outputs and not bin_result.is_empty:
            write_redshift_fisher(
                str(config.outputfile),
                bin_result.cov_renormalised,
                bin_result.means,
                bin_result.z,
            )

    # Combined result over the whole redshift range
    result.full_fisher = full_fisher
    result.combined = _combine_bins(config, cosmo, data, full_fisher)
    if verbose:
        print("#  Combined errors")
        print("#=================")
        _print_bin_result(result.combined)

    return result


# --------------------------------------------------------------------------- #
# Per-bin processing                                                          #
# --------------------------------------------------------------------------- #
def _process_redshift_bin(
    iz: int,
    cosmo: Cosmology,
    data: TracerData,
    recon: np.ndarray,
    der_alpha: list,
    der_alpha_bao: list,
    config: ForecastConfig,
    full_fisher: np.ndarray,
) -> RedshiftBinResult:
    """Compute and record the per-bin Fisher matrix; accumulate into the full one."""
    n_tracers = data.n_tracers
    z = float(cosmo.z[iz])
    volume_gpc3 = float(cosmo.volume[iz] / 1e9)

    # Empty bin: no tracers with galaxies. Skip computation, write nothing.
    if not np.any(data.nz[:, iz] > 1.0e-30):
        return RedshiftBinResult(
            z=z, volume_gpc3=volume_gpc3,
            means=np.full(3, np.nan),
            cov_renormalised=np.full((n_tracers + 3, n_tracers + 3), np.nan),
            errors_pct=np.full(3, np.nan),
            iso_alpha_err_pct=np.inf,
            is_empty=True,
        )

    # Main Fisher matrix integration up to kmax
    fisher = compute_fisher_matrix(
        cosmo, data, iz,
        recon=recon[iz],
        der_p_alpha=der_alpha,
        kmin=cosmo.kmin, kmax=cosmo.kmax,
        BAO_only=config.BAO_only,
        fast_integration=config.fast_integration,
    )

    # Optionally extend BAO-only information above kmax for the alphas.
    kmax_bao = cosmo.kmax if config.kmax_bao is None else config.kmax_bao
    if kmax_bao > cosmo.kmax:
        fisher_extra = compute_fisher_matrix(
            cosmo, data, iz,
            recon=recon[iz],
            der_p_alpha=der_alpha_bao,
            kmin=cosmo.kmax, kmax=kmax_bao,
            BAO_only=True,
            fast_integration=config.fast_integration,
        )
        fisher[-2:, -2:] += fisher_extra[-2:, -2:]

    # Accumulate into the full multi-z Fisher matrix
    z_slice = slice(iz * n_tracers, (iz + 1) * n_tracers)
    full_fisher[z_slice, z_slice] += fisher[:n_tracers, :n_tracers]
    full_fisher[z_slice, -3:] += fisher[:n_tracers, -3:]
    full_fisher[-3:, z_slice] += fisher[-3:, :n_tracers]
    full_fisher[-3:, -3:] += fisher[-3:, -3:]

    # Per-bin covariance
    cov = np.linalg.inv(fisher)

    # Isotropic alpha error: alpha = (2/3) alpha_perp + (1/3) alpha_par.
    J_iso = np.array([2.0 / 3.0, 1.0 / 3.0])
    iso_alpha_err_pct = 100.0 * float(np.sqrt(J_iso @ cov[-2:, -2:] @ J_iso.T))

    means = np.array([
        cosmo.f[iz] * cosmo.sigma8[iz],
        cosmo.da[iz],
        cosmo.h[iz],
    ])
    cov_renorm = renormalize_covariance(cov, means)
    errors_pct = 100.0 * np.sqrt(np.diag(cov_renorm)[-3:]) / means

    return RedshiftBinResult(
        z=z,
        volume_gpc3=volume_gpc3,
        means=means,
        cov_renormalised=cov_renorm,
        errors_pct=errors_pct,
        iso_alpha_err_pct=iso_alpha_err_pct,
    )


def _combine_bins(
    config: ForecastConfig,
    cosmo: Cosmology,
    data: TracerData,
    full_fisher: np.ndarray,
) -> RedshiftBinResult:
    """Invert the full multi-z Fisher matrix and report combined errors.

    The combined cosmology is computed at the overall redshift range
    (zmin[0] to zmax[-1]) — this matches the original code's behaviour.
    """
    combined_cosmo = Cosmology.from_config(
        config,
        np.atleast_1d(data.zmin[0]),
        np.atleast_1d(data.zmax[-1]),
    )
    cov = np.linalg.inv(full_fisher)

    J_iso = np.array([2.0 / 3.0, 1.0 / 3.0])
    iso_alpha_err_pct = 100.0 * float(np.sqrt(J_iso @ cov[-2:, -2:] @ J_iso.T))

    means = np.array([
        combined_cosmo.f[0] * combined_cosmo.sigma8[0],
        combined_cosmo.da[0],
        combined_cosmo.h[0],
    ])
    cov_renorm = renormalize_covariance(cov, means)
    errors_pct = 100.0 * np.sqrt(np.diag(cov_renorm)[-3:]) / means

    return RedshiftBinResult(
        z=float(combined_cosmo.z[0]),
        volume_gpc3=float(combined_cosmo.volume[0] / 1e9),
        means=means,
        cov_renormalised=cov_renorm,
        errors_pct=errors_pct,
        iso_alpha_err_pct=iso_alpha_err_pct,
    )


# --------------------------------------------------------------------------- #
# Output formatting                                                           #
# --------------------------------------------------------------------------- #
def _print_bin_result(r: RedshiftBinResult) -> None:
    if r.is_empty:
        print(
            f" {r.z:.2f}     {r.volume_gpc3:.4f}        -            -          "
            f"-           -          -           -          -"
        )
        return
    print(
        f" {r.z:.2f}    {r.volume_gpc3:.4f}     {r.means[0]:.3f}       "
        f"{r.errors_pct[0]:.2f}         {r.means[1]:.1f}       "
        f"{r.errors_pct[1]:.2f}        {r.means[2]:.1f}       "
        f"{r.errors_pct[2]:.2f}       {r.iso_alpha_err_pct:.3f}"
    )
