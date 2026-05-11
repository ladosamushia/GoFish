"""Cosmology evaluation: runs CAMB and packages everything needed downstream.

The :class:`Cosmology` object computes, at a list of redshift bin centres, the
linear matter power spectrum, its dewiggled smooth version, the growth rate
``f``, ``sigma8``, distances, and the BAO damping scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import splrep

if TYPE_CHECKING:
    from .config import ForecastConfig

# Empirical coefficient for BAO damping perpendicular to LOS, calibrated against
# N-body simulations: Sigma_perp = (9.4 Mpc/h) * sigma8(z) / 0.9.
_SIGMA_PERP_COEFF_MPCH = 9.4
_SIGMA_PERP_SIGMA8_NORM = 0.9


@dataclass
class Cosmology:
    """Cosmological quantities at each redshift bin centre.

    Attributes
    ----------
    z : np.ndarray
        Bin centre redshifts.
    volume : np.ndarray
        Comoving bin volume in :math:`({\\rm Mpc}/h)^3`.
    k : np.ndarray
        k grid in h/Mpc for the splined power spectra.
    pk : list[tuple]
        ``scipy.interpolate.splrep`` representations of P(k) at each z.
    pksmooth : list[tuple]
        Dewiggled (BAO-removed) P(k) splines at each z.
    da : np.ndarray
        Angular diameter distance in Mpc.
    h : np.ndarray
        H(z) in km/s/Mpc.
    f : np.ndarray
        Growth rate of structure.
    sigma8 : np.ndarray
        sigma8(z) (Mpc/h).
    growth : np.ndarray
        Linear growth factor normalised to D(z=0) = 1.
    r_d : float
        Sound horizon at the baryon drag epoch (Mpc).
    Sigma_perp, Sigma_par : np.ndarray
        BAO damping scales in Mpc/h.
    kmin, kmax : float
        Wavenumber range for the Fisher integration.
    """

    z: np.ndarray
    volume: np.ndarray
    k: np.ndarray
    pk: list
    pksmooth: list
    da: np.ndarray
    h: np.ndarray
    f: np.ndarray
    sigma8: np.ndarray
    growth: np.ndarray
    r_d: float
    Sigma_perp: np.ndarray = field(init=False)
    Sigma_par: np.ndarray = field(init=False)
    kmin: float = field(default=0.0)
    kmax: float = field(default=0.5)

    def __post_init__(self) -> None:
        self.Sigma_perp = _SIGMA_PERP_COEFF_MPCH * self.sigma8 / _SIGMA_PERP_SIGMA8_NORM
        self.Sigma_par = (1.0 + self.f) * self.Sigma_perp

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        config: "ForecastConfig",
        zlow: np.ndarray,
        zhigh: np.ndarray,
    ) -> "Cosmology":
        """Run CAMB at ``zmid = (zlow + zhigh) / 2`` and package the result."""
        import camb

        zlow = np.atleast_1d(zlow)
        zhigh = np.atleast_1d(zhigh)
        zmid = 0.5 * (zlow + zhigh)

        pars = camb.CAMBparams()
        if config.w0_fld is not None:
            pars.set_dark_energy(w=config.w0_fld, dark_energy_model="fluid")
        pars.InitPower.set_params(As=config.A_s, ns=config.n_s)

        # CAMB wants redshifts in descending order, plus z=0 for normalisation.
        camb_redshifts = np.concatenate([zmid[::-1], [0.0]])
        pars.set_matter_power(redshifts=camb_redshifts, kmax=5.0)
        pars.set_cosmology(
            H0=config.H0,
            omch2=config.omega_cdm,
            ombh2=config.omega_b,
            omk=config.Omega_k,
            tau=config.tau_reio,
            mnu=config.Sum_mnu,
            neutrino_hierarchy=config.nu_hierarchy,
            thetastar=config.thetastar,
        )
        pars.NonLinear = camb.model.NonLinear_none

        results = camb.get_results(pars)
        kin, _, pklin = results.get_matter_power_spectrum(
            minkh=2.0e-5, maxkh=5.0, npoints=2000
        )

        # CAMB returns redshifts in the order we passed them (then sorted),
        # so flip back and drop the z=0 entry.
        pklin_ordered = pklin[::-1][1:]

        # Derived quantities
        area_sr = config.skyarea * (np.pi / 180.0) ** 2
        h_unit = pars.H0 / 100.0
        rmin = results.comoving_radial_distance(zlow) * h_unit
        rmax = results.comoving_radial_distance(zhigh) * h_unit
        volume = area_sr / 3.0 * (rmax**3 - rmin**3)

        da = results.angular_diameter_distance(zmid)
        hubble = results.hubble_parameter(zmid)
        fsigma8 = results.get_fsigma8()[::-1][1:]
        sigma8 = results.get_sigma8()[::-1][1:]
        sigma8_z0 = results.get_sigma8()[-1]
        f = fsigma8 / sigma8
        growth = sigma8 / sigma8_z0
        r_d = results.get_derived_params()["rdrag"]

        pk_splines = [splrep(kin, pklin_ordered[i]) for i in range(len(zmid))]
        pksmooth_splines = [
            splrep(kin, _smooth_hinton2017(kin, pklin_ordered[i]))
            for i in range(len(zmid))
        ]

        cosmo = cls(
            z=zmid,
            volume=volume,
            k=kin,
            pk=pk_splines,
            pksmooth=pksmooth_splines,
            da=da,
            h=hubble,
            f=f,
            sigma8=sigma8,
            growth=growth,
            r_d=r_d,
        )
        cosmo.kmin = max(config.kmin, float(kin[0]))
        cosmo.kmax = config.kmax
        return cosmo


def _smooth_hinton2017(
    k: np.ndarray,
    pk: np.ndarray,
    degree: int = 13,
    sigma: float = 1.0,
    weight: float = 0.5,
) -> np.ndarray:
    """Polynomial dewiggling of P(k) following Hinton+ 2017.

    Fits a polynomial of order ``degree`` in log(k) to log(P), down-weighting
    points near the BAO peak with a Gaussian. The returned array is the
    smooth (no-wiggle) reference power spectrum.
    """
    log_k = np.log(k)
    gauss = np.exp(-0.5 * ((log_k - log_k[np.argmax(pk)]) / sigma) ** 2)
    fit_weights = 1.0 - weight * gauss
    poly = np.poly1d(np.polyfit(log_k, np.log(pk), degree, w=fit_weights))
    return np.exp(poly(log_k))
