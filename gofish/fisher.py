"""Fisher matrix computation for galaxy clustering.

The parameter vector at each redshift bin is
``[b_0 * sigma8, ..., b_{Ntracers-1} * sigma8, f*sigma8, alpha_perp, alpha_par]``.
The power spectrum vector contains the npop * (npop + 1) / 2 auto- and
cross-spectra in ``itertools.combinations_with_replacement`` order:
``P_00, P_01, ..., P_0{n-1}, P_11, P_12, ..., P_{n-1}{n-1}``.

The mapping between these orderings is built once via :func:`_pair_index_map`
to avoid the obscure arithmetic indexing used in the original code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np
from findiff import FinDiff
from scipy.integrate import quad_vec, simpson
from scipy.interpolate import RegularGridInterpolator, splev, splrep

if TYPE_CHECKING:
    from .cosmology import Cosmology
    from .data import TracerData


# --------------------------------------------------------------------------- #
# BAO reconstruction lookup table                                             #
# (from Seo & Eisenstein 2007 / DESI BAO reconstruction studies)              #
# --------------------------------------------------------------------------- #
_RECON_NP_TABLE = np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0])
_RECON_RFACTOR_TABLE = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5])
_RECON_MU_FIDUCIAL = 0.6  # mu and k at which nP is evaluated for the lookup
_RECON_K_FIDUCIAL = 0.14
_RECON_NORMALISATION = 0.1734  # divisor that maps nP_comb onto the table's nP

# --------------------------------------------------------------------------- #
# Pair-index utilities                                                        #
# --------------------------------------------------------------------------- #
def _pair_index_map(n_tracers: int) -> dict[tuple[int, int], int]:
    """Index of the spectrum ``P_{ij}`` in the flat (auto+cross) list."""
    return {
        pair: idx
        for idx, pair in enumerate(combinations_with_replacement(range(n_tracers), 2))
    }


# --------------------------------------------------------------------------- #
# Public: BAO reconstruction factor per redshift                              #
# --------------------------------------------------------------------------- #
def compute_reconstruction_factor(cosmo: "Cosmology", data: "TracerData") -> np.ndarray:
    """Interpolate the BAO reconstruction factor :math:`r(z)` for each bin.

    Returns shape ``(nz,)``. Values are clipped to the lookup-table endpoints.
    """
    r_spline = splrep(_RECON_NP_TABLE, _RECON_RFACTOR_TABLE)

    kaiser = data.bias + cosmo.f * _RECON_MU_FIDUCIAL**2  # (n_tracers, nz)
    nbar_eff = np.sum(data.nbar * kaiser**2, axis=0)        # (nz,)

    recon = np.empty(len(cosmo.z))
    for iz in range(len(cosmo.z)):
        nP = nbar_eff[iz] * splev(_RECON_K_FIDUCIAL, cosmo.pk[iz]) / _RECON_NORMALISATION
        if nP <= _RECON_NP_TABLE[0]:
            recon[iz] = _RECON_RFACTOR_TABLE[0]
        elif nP >= _RECON_NP_TABLE[-1]:
            recon[iz] = _RECON_RFACTOR_TABLE[-1]
        else:
            recon[iz] = float(splev(nP, r_spline))
    return recon


# --------------------------------------------------------------------------- #
# Public: derivative of P(k') w.r.t. alpha_perp and alpha_par                 #
# --------------------------------------------------------------------------- #
def precompute_alpha_derivatives(
    cosmo: "Cosmology", BAO_only: bool = False, dk: float = 1.0e-4
) -> list[RegularGridInterpolator]:
    """Return interpolators ``[dP/dalpha_perp(k,mu), dP/dalpha_par(k,mu)]``.

    These are evaluated at the *first* redshift bin (i.e. ``cosmo.pk[0]``); for
    other redshift bins they are scaled by ``(sigma8(z)/sigma8(z_0))^2`` when
    ``BAO_only`` is False. When ``BAO_only`` is True we use ``P/P_smooth`` so
    the scaling does not apply.
    """
    order = 4
    nmu = 100
    mu = np.linspace(0.0, 1.0, nmu)

    # Sample P(k+i*dk) at 2*order+1 points and 4th-order finite-difference d/dk.
    stencil = np.empty((2 * order + 1, len(cosmo.k)))
    for i in range(-order, order + 1):
        k_shift = cosmo.k + i * dk
        if BAO_only:
            stencil[i + order] = splev(k_shift, cosmo.pk[0]) / splev(k_shift, cosmo.pksmooth[0])
        else:
            stencil[i + order] = splev(k_shift, cosmo.pk[0])
    dP_dk = FinDiff(0, dk, acc=4)(stencil)[order]

    # dP/dalpha_perp = (mu^2 - 1) k * dP/dk;  dP/dalpha_par = -mu^2 k * dP/dk
    der_perp = np.outer(dP_dk * cosmo.k, mu**2 - 1.0)
    der_par = -np.outer(dP_dk * cosmo.k, mu**2)

    return [
        RegularGridInterpolator([cosmo.k, mu], der_perp),
        RegularGridInterpolator([cosmo.k, mu], der_par),
    ]


# --------------------------------------------------------------------------- #
# Core: Fisher matrix at a single redshift bin                                #
# --------------------------------------------------------------------------- #
@dataclass
class _IntegrandContext:
    """Everything :func:`_fisher_kernel` needs at a fixed redshift."""

    iz: int
    n_tracers: int
    n_pk: int
    pair_idx: dict
    nbar: np.ndarray
    bias: np.ndarray
    f: float
    sigma8: float
    sigma8_ratio: float          # sigma8(z) / sigma8(z=ref)
    Sigma_perp_sq: float
    Sigma_par_sq: float
    recon_sq: float
    pk_spline: tuple
    pksmooth_spline: tuple
    der_p_alpha: list            # [interp_perp, interp_par]
    BAO_only: bool


def compute_fisher_matrix(
    cosmo: "Cosmology",
    data: "TracerData",
    iz: int,
    recon: float,
    der_p_alpha: list,
    kmin: float,
    kmax: float,
    BAO_only: bool,
    fast_integration: bool = True,
) -> np.ndarray:
    """Fisher information matrix at a single redshift bin.

    Parameters
    ----------
    cosmo, data : Cosmology, TracerData
        Inputs.
    iz : int
        Redshift bin index.
    recon : float
        BAO reconstruction factor for this bin.
    der_p_alpha : list
        Output of :func:`precompute_alpha_derivatives`.
    kmin, kmax : float
        Integration range in h/Mpc.
    BAO_only : bool
        If True, only BAO wiggles contribute to alpha derivatives (standard
        BAO analysis). If False, the full Kaiser-modulated P(k) does
        (full-shape analysis).
    fast_integration : bool
        If True, use Simpson's rule on a fixed (400, 100) grid in (k, mu).
        Otherwise, use adaptive quadrature (``scipy.integrate.quad_vec``).

    Returns
    -------
    np.ndarray
        Fisher matrix of shape ``(n_tracers + 3, n_tracers + 3)`` ordered as
        ``[b_i sigma8 ..., f sigma8, alpha_perp, alpha_par]``.
    """
    n_tracers = data.n_tracers
    pair_idx = _pair_index_map(n_tracers)

    ctx = _IntegrandContext(
        iz=iz,
        n_tracers=n_tracers,
        n_pk=len(pair_idx),
        pair_idx=pair_idx,
        nbar=data.nbar[:, iz].copy(),
        bias=data.bias[:, iz].copy(),
        f=float(cosmo.f[iz]),
        sigma8=float(cosmo.sigma8[iz]),
        sigma8_ratio=float(cosmo.sigma8[iz] / cosmo.sigma8[0]),
        Sigma_perp_sq=float(cosmo.Sigma_perp[iz] ** 2),
        Sigma_par_sq=float(cosmo.Sigma_par[iz] ** 2),
        recon_sq=float(recon**2),
        pk_spline=cosmo.pk[iz],
        pksmooth_spline=cosmo.pksmooth[iz],
        der_p_alpha=der_p_alpha,
        BAO_only=BAO_only,
    )

    if fast_integration:
        fisher = _integrate_simpson(ctx, kmin, kmax)
    else:
        fisher = _integrate_quad_vec(ctx, kmin, kmax)

    # Prefactor from the standard Fisher-from-power-spectrum derivation.
    fisher *= cosmo.volume[iz] / (2.0 * np.pi**2)
    return fisher


# --------------------------------------------------------------------------- #
# Integration drivers                                                         #
# --------------------------------------------------------------------------- #
def _integrate_simpson(ctx: _IntegrandContext, kmin: float, kmax: float) -> np.ndarray:
    """Fixed-grid Simpson's rule in (k, mu)."""
    k_grid = np.linspace(kmin, kmax, 400)
    mu_grid = np.linspace(0.0, 1.0, 100)

    # Evaluate kernel on the grid. Shape: (len(k), len(mu), n_par, n_par).
    n_par = ctx.n_tracers + 3
    grid = np.empty((len(k_grid), len(mu_grid), n_par, n_par))
    for i, k in enumerate(k_grid):
        for j, mu in enumerate(mu_grid):
            grid[i, j] = _fisher_kernel(mu, k, ctx)

    # Integrate over mu, then k.
    integrated_mu = simpson(grid, x=mu_grid, axis=1)
    return simpson(integrated_mu, x=k_grid, axis=0)


def _integrate_quad_vec(ctx: _IntegrandContext, kmin: float, kmax: float) -> np.ndarray:
    """Adaptive vector-valued quadrature in (k, mu)."""

    def inner(k: float) -> np.ndarray:
        return quad_vec(
            lambda mu: _fisher_kernel(mu, k, ctx),
            0.0, 1.0, epsabs=1.0e-6, epsrel=1.0e-6, limit=10000,
        )[0]

    return quad_vec(inner, kmin, kmax, epsabs=1.0e-5, epsrel=1.0e-5, limit=1000)[0]


# --------------------------------------------------------------------------- #
# Pointwise integrand                                                         #
# --------------------------------------------------------------------------- #
def _fisher_kernel(mu: float, k: float, ctx: _IntegrandContext) -> np.ndarray:
    """Fisher matrix density at a single ``(mu, k)``.

    Returns shape ``(n_par, n_par)`` where
    ``n_par = n_tracers + 3 (= f sigma8, alpha_perp, alpha_par)``.
    """
    # Kaiser factor b + f * mu^2  (per tracer)
    kaiser = ctx.bias + ctx.f * mu**2

    # BAO damping after reconstruction
    D_par = mu**2 * k**2 * ctx.Sigma_par_sq
    D_perp = (1.0 - mu**2) * k**2 * ctx.Sigma_perp_sq
    D_factor_sq = np.exp(-ctx.recon_sq * (D_par + D_perp))  # (already squared!)

    # Power spectrum at this (z, k)
    pk = float(splev(k, ctx.pk_spline))
    pk_smooth = float(splev(k, ctx.pksmooth_spline))

    # Alpha derivatives at this (k, mu)
    der_alpha_perp = float(ctx.der_p_alpha[0]((k, mu)))
    der_alpha_par = float(ctx.der_p_alpha[1]((k, mu)))
    if not ctx.BAO_only:
        scale = ctx.sigma8_ratio**2
        der_alpha_perp *= scale
        der_alpha_par *= scale

    # Derivatives and (inverse) covariance of the spectra
    der_P = _derivative_matrix(ctx, kaiser, pk, pk_smooth, mu,
                               der_alpha_perp, der_alpha_par)
    cov_inv = _inverse_spectra_covariance(ctx, kaiser, pk)

    return k**2 * (der_P @ cov_inv @ der_P.T) * D_factor_sq


def _inverse_spectra_covariance(
    ctx: _IntegrandContext, kaiser: np.ndarray, pk: float
) -> np.ndarray:
    """Inverse covariance between auto- and cross-spectra at fixed (k, mu).

    For Gaussian fields, Cov(P_{12}, P_{34}) = (P_{13} P_{24} + P_{14} P_{23})
    where shot noise 1/nbar_i is added to P_{ii}.
    """
    n_pk = ctx.n_pk
    cov = np.empty((n_pk, n_pk))
    pk_pairs = list(combinations_with_replacement(range(ctx.n_tracers), 2))

    for ps1, (a, b) in enumerate(pk_pairs):
        for ps2, (c, d) in enumerate(pk_pairs):
            p_ac = kaiser[a] * kaiser[c] * pk + (1.0 / ctx.nbar[a] if a == c else 0.0)
            p_bd = kaiser[b] * kaiser[d] * pk + (1.0 / ctx.nbar[b] if b == d else 0.0)
            p_ad = kaiser[a] * kaiser[d] * pk + (1.0 / ctx.nbar[a] if a == d else 0.0)
            p_bc = kaiser[b] * kaiser[c] * pk + (1.0 / ctx.nbar[b] if b == c else 0.0)
            cov[ps1, ps2] = p_ac * p_bd + p_ad * p_bc

    return np.linalg.inv(cov)


def _derivative_matrix(
    ctx: _IntegrandContext,
    kaiser: np.ndarray,
    pk: float,
    pk_smooth: float,
    mu: float,
    der_alpha_perp: float,
    der_alpha_par: float,
) -> np.ndarray:
    """Derivatives of all auto/cross spectra w.r.t. each parameter.

    Returns shape ``(n_par, n_pk)``, with the parameter order
    ``[b_0 sigma8, ..., b_{n-1} sigma8, f sigma8, alpha_perp, alpha_par]`` and
    spectrum order matching :func:`_inverse_spectra_covariance`.
    """
    n_tracers = ctx.n_tracers
    n_pk = ctx.n_pk
    pair_idx = ctx.pair_idx
    f = ctx.f
    sigma8 = ctx.sigma8

    der_P = np.zeros((n_tracers + 3, n_pk))

    # ----- Derivatives w.r.t. b_i sigma8 -----------------------------------
    # P_ij ≈ b_i b_j P(k) plus RSD, so dP_ij/d(b_i sigma8) ≈ b_j P / sigma8.
    # Auto-spectra get a factor of 2 from b_i^2.
    for i in range(n_tracers):
        for j in range(n_tracers):
            pair = (i, j) if i <= j else (j, i)
            coeff = 2.0 if i == j else 1.0
            der_P[i, pair_idx[pair]] = coeff * kaiser[j] * pk / sigma8

    # ----- Derivative w.r.t. f sigma8 --------------------------------------
    # P_ij = (b_i + f mu^2)(b_j + f mu^2) P. d/d(f s8) = (kaiser_i + kaiser_j) mu^2 P / s8.
    f_row = n_tracers
    for (i, j), idx in pair_idx.items():
        der_P[f_row, idx] = (kaiser[i] + kaiser[j]) * mu**2 * pk / sigma8

    # ----- Derivatives w.r.t. alpha_perp / alpha_par -----------------------
    perp_row, par_row = n_tracers + 1, n_tracers + 2
    if ctx.BAO_only:
        # Only the BAO wiggles carry alpha info; the Kaiser factor is alpha-independent.
        for (i, j), idx in pair_idx.items():
            kk = kaiser[i] * kaiser[j]
            der_P[perp_row, idx] = kk * der_alpha_perp * pk_smooth
            der_P[par_row, idx] = kk * der_alpha_par * pk_smooth
    else:
        # Full-shape: kaiser factor depends on alpha through mu' = mu(alpha).
        # d(mu'^2)/d(alpha_perp) = 2 mu^2 (1 - mu^2);  d(mu'^2)/d(alpha_par) = - same.
        d_mu2 = 2.0 * mu**2 * (1.0 - mu**2)
        for (i, j), idx in pair_idx.items():
            kk = kaiser[i] * kaiser[j]
            ks = kaiser[i] + kaiser[j]
            der_P[perp_row, idx] = ks * f * pk * d_mu2 + kk * der_alpha_perp
            der_P[par_row, idx] = -ks * f * pk * d_mu2 + kk * der_alpha_par

    return der_P


# --------------------------------------------------------------------------- #
# Covariance renormalisation: (alpha_perp, alpha_par) -> (D_A, H)             #
# --------------------------------------------------------------------------- #
def renormalize_covariance(cov: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Convert a covariance from ``[..., f s8, alpha_perp, alpha_par]`` to
    ``[..., f s8, D_A, H]`` via the linear Jacobian J = diag(..., 1, D_A, -H).

    The negative sign is correct: at the fiducial cosmology
    ``alpha_par ∝ 1/H``, so ``dH/dalpha_par|_{alpha=1} = -H``.
    """
    jacobian = np.eye(cov.shape[0])
    jacobian[-2, -2] = means[1]   # D_A
    jacobian[-1, -1] = -means[2]  # -H
    return jacobian @ cov @ jacobian.T
