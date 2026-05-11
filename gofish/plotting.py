"""Plot model power-spectrum multipoles with forecast error bars.

This replaces ``plot_pk.py`` from the original code. The repeated ``errorbar``
calls have been condensed into a small helper, and the inputs come from
:class:`gofish.cosmology.Cosmology` and :class:`gofish.data.TracerData`.

Run as a script::

    python -m gofish.plotting config/test_multitracer.ini
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import splev

from .config import ForecastConfig
from .cosmology import Cosmology
from .data import TracerData
from .fisher import compute_reconstruction_factor


# --------------------------------------------------------------------------- #
# Multipole computation                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class MultipoleResult:
    """Power-spectrum multipoles, smooth references, and their covariances."""

    k: np.ndarray
    pk: np.ndarray             # (3, n_k) - P_0, P_2, P_4
    pk_smooth: np.ndarray      # (3, n_k)
    pk_cov: np.ndarray         # (3, n_k) - diagonal of multipole covariance


def compute_multipoles(
    cosmo: Cosmology,
    data: TracerData,
    iz: int,
    *,
    tracer_index: int | None = None,
    n_mu: int = 100,
    dk_fine: float = 0.001,
    dk_out: float = 0.005,
) -> MultipoleResult:
    """Compute P_0, P_2, P_4 and their covariances for one redshift bin.

    Parameters
    ----------
    cosmo, data : Cosmology, TracerData
        Inputs.
    iz : int
        Redshift bin index.
    tracer_index : int, optional
        Which tracer to use. Defaults to ``iz`` (matching the original
        ``plot_pk.py`` which assumed one tracer per bin).
    n_mu, dk_fine, dk_out : numeric
        Integration grid resolution. The fine grid is integrated over mu and
        then averaged onto wider k bins of width ``dk_out``.
    """
    if tracer_index is None:
        tracer_index = iz

    recon = compute_reconstruction_factor(cosmo, data)[iz]

    # Fine k grid and the coarser output binning
    n_combine = int(round(dk_out / dk_fine))
    n_fine = int((cosmo.kmax - cosmo.kmin) / dk_fine)
    n_out = int((cosmo.kmax - cosmo.kmin) / dk_out)
    k_fine = np.linspace(cosmo.kmin, cosmo.kmax, n_fine, endpoint=False) + dk_fine / 2.0
    k_out = np.linspace(cosmo.kmin, cosmo.kmax, n_out, endpoint=False) + dk_out / 2.0
    # Truncate the fine grid to be a clean multiple of n_combine
    n_fine = (n_fine // n_combine) * n_combine
    k_fine = k_fine[:n_fine]
    k_out = k_out[: n_fine // n_combine]
    mu = np.linspace(0.0, 1.0, n_mu)

    # Anisotropic 2D model P(k, mu)
    bias = data.bias[tracer_index, iz]
    kaiser = bias + cosmo.f[iz] * mu**2                                    # (n_mu,)
    D_par = np.outer(mu**2, k_fine**2) * cosmo.Sigma_par[iz]**2            # (n_mu, n_fine)
    D_perp = np.outer(1.0 - mu**2, k_fine**2) * cosmo.Sigma_perp[iz]**2
    damping = np.exp(-(recon**2) * (D_par + D_perp) / 2.0)                 # (n_mu, n_fine)

    pk_1d = splev(k_fine, cosmo.pk[iz])
    pk_smooth_1d = splev(k_fine, cosmo.pksmooth[iz])
    pk_2d = (kaiser[:, None] ** 2 * pk_1d[None, :] * damping).T            # (n_fine, n_mu)
    pk_smooth_2d = (kaiser[:, None] ** 2 * pk_smooth_1d[None, :] * damping).T

    # Multipoles via Legendre projection
    legendre = {
        0: np.ones_like(mu),
        2: 0.5 * (3.0 * mu**2 - 1.0),
        4: 0.125 * (35.0 * mu**4 - 30.0 * mu**2 + 3.0),
    }
    weight = {0: 1.0, 2: 5.0, 4: 9.0}
    pk_mult = np.array([
        weight[ell] * simpson(pk_2d * legendre[ell], x=mu, axis=-1)
        for ell in (0, 2, 4)
    ])  # (3, n_fine)
    pk_mult_smooth = np.array([
        weight[ell] * simpson(pk_smooth_2d * legendre[ell], x=mu, axis=-1)
        for ell in (0, 2, 4)
    ])

    # Diagonal covariance per multipole
    shot = 1.0 / data.nbar[tracer_index, iz]
    cov_weight = {0: 1.0, 2: 2.0 * 25.0, 4: 2.0 * 81.0}
    pk_mult_cov = np.array([
        cov_weight[ell] * simpson((pk_2d + shot) ** 2 * legendre[ell] ** 2, x=mu, axis=-1)
        for ell in (0, 2, 4)
    ]) / cosmo.volume[iz]

    # Rebin from the fine grid to the output grid using shell-volume weights
    V_out = 4.0 / 3.0 * np.pi * ((k_out + dk_out / 2.0) ** 3 - (k_out - dk_out / 2.0) ** 3)
    shell_factor = 4.0 * np.pi * k_fine**2 * dk_fine

    def rebin(arr):
        return (arr * shell_factor).reshape((arr.shape[0], -1, n_combine)).sum(axis=-1) / V_out

    def rebin_cov(arr):
        return (
            2.0 * (2.0 * np.pi) ** 4
            * (arr * shell_factor).reshape((arr.shape[0], -1, n_combine)).sum(axis=-1)
            / V_out**2
            / (4.0 * np.pi * dk_fine)  # cancel out the dk_fine put in by `shell_factor`
        )
    # Note: the original rebin for the covariance had a slightly different
    # algebra; preserve the original formula explicitly to keep numerics.
    pk_out = np.array([
        4.0 * np.pi * (pk_mult[i] * k_fine**2).reshape((-1, n_combine)).sum(axis=-1) * dk_fine / V_out
        for i in range(3)
    ])
    pk_smooth_out = np.array([
        4.0 * np.pi * (pk_mult_smooth[i] * k_fine**2).reshape((-1, n_combine)).sum(axis=-1) * dk_fine / V_out
        for i in range(3)
    ])
    pk_cov_out = np.array([
        2.0 * (2.0 * np.pi) ** 4
        * (pk_mult_cov[i] * k_fine**2).reshape((-1, n_combine)).sum(axis=-1) * dk_fine / V_out**2
        for i in range(3)
    ])

    return MultipoleResult(k=k_out, pk=pk_out, pk_smooth=pk_smooth_out, pk_cov=pk_cov_out)


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
_MULTIPOLE_STYLES = [
    {"marker": "o", "color": "r"},   # P_0
    {"marker": "s", "color": "b"},   # P_2
    {"marker": "D", "color": "g"},   # P_4
]


def plot_multipoles(
    result: MultipoleResult,
    *,
    sample_label: str = "",
    redshift_label: str = "",
    ylim: tuple[float, float] | None = None,
    show_bao_ratio_inset: bool = True,
):
    """Plot kP_ell(k) for ell=0,2,4 with error bars and an inset P_0 / P_0^nw."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.83, 0.83])
    k = result.k
    for ell_idx, style in enumerate(_MULTIPOLE_STYLES):
        y = k * result.pk[ell_idx]
        yerr = k * np.sqrt(result.pk_cov[ell_idx])
        _plot_with_error_bars(ax, k, y, yerr, **style)

    ax.axvline(x=0.1, color="k", ls="--", lw=1.3)
    ax.set_xlim(0.0, 0.30)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=14, labelpad=2)
    ax.set_ylabel(r"$k\times P(k)\,(h^{-2}\mathrm{Mpc^{2}})$", fontsize=14, labelpad=2)
    _style_axis(ax, fontsize=12)
    if sample_label:
        ax.text(0.04, 0.93, sample_label, fontsize=14, transform=ax.transAxes)
    if redshift_label:
        ax.text(0.04, 0.86, redshift_label, fontsize=14, transform=ax.transAxes)

    if show_bao_ratio_inset:
        ax2 = fig.add_axes([0.65, 0.65, 0.32, 0.32])
        ratio = result.pk[0] / result.pk_smooth[0]
        ratio_err = np.sqrt(result.pk_cov[0]) / result.pk_smooth[0]
        _plot_with_error_bars(ax2, k, ratio, ratio_err, marker="o", color="r")
        ax2.set_xlim(0.0, 0.50)
        ax2.set_ylim(0.91, 1.09)
        ax2.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=12, labelpad=0)
        ax2.set_ylabel(r"$P_{0}(k)/P_{0}^{\mathrm{nw}}(k)$", fontsize=12, labelpad=0)
        _style_axis(ax2, fontsize=10, linewidth=1.1)

    return fig


def _plot_with_error_bars(ax, x, y, yerr, *, marker, color):
    """One scatter-with-errorbars + a translucent connecting line, matching the original look."""
    ax.errorbar(x, y, yerr=yerr, marker=marker, markersize=4, color=color,
                mfc="w", linestyle="None", markeredgewidth=1.3, zorder=5)
    ax.errorbar(x, y, yerr=yerr, marker="None", color=color, linestyle="-",
                zorder=5, alpha=0.4)


def _style_axis(ax, *, fontsize: int, linewidth: float = 1.3):
    ax.tick_params(width=linewidth)
    ax.tick_params("both", length=10, which="major")
    ax.tick_params("both", length=5, which="minor")
    for axis in ("top", "left", "bottom", "right"):
        ax.spines[axis].set_linewidth(linewidth)
    for tick in list(ax.xaxis.get_ticklabels()) + list(ax.yaxis.get_ticklabels()):
        tick.set_fontsize(fontsize)


# --------------------------------------------------------------------------- #
# CLI for the plotting script                                                 #
# --------------------------------------------------------------------------- #
_DEFAULT_LABELS = [
    (r"$\mathrm{BGS\,BRIGHT}$", r"$0.0 < z < 0.4$", (-500.0, 1900.0)),
    (r"$\mathrm{LRG}$",         r"$0.4 < z < 1.1$", (-250.0, 2800.0)),
    (r"$\mathrm{ELG\_LOP}$",    r"$1.1 < z < 1.6$", (-100.0,  850.0)),
    (r"$\mathrm{Quasar}$",      r"$1.6 < z < 2.1$", (-200.0, 1500.0)),
]


def _cli_main(argv: list[str] | None = None) -> int:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Plot model power spectrum multipoles with forecast errors."
    )
    parser.add_argument("config", type=Path, help="Path to the .ini config file.")
    args = parser.parse_args(argv)

    config = ForecastConfig.from_ini(args.config)
    data = TracerData.from_file(config.inputfile)
    cosmo = Cosmology.from_config(config, data.zmin, data.zmax)
    if np.any(data.volume > 0):
        cosmo.volume = data.volume * 1.0e9
    data.convert_nbar(cosmo.volume, config.skyarea)
    if config.scale_bias:
        data.scale_bias_by_growth(cosmo.growth)

    for iz, (sample, zlabel, ylim) in enumerate(_DEFAULT_LABELS[: len(cosmo.z)]):
        mult = compute_multipoles(cosmo, data, iz)
        plot_multipoles(mult, sample_label=sample, redshift_label=zlabel, ylim=ylim)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
