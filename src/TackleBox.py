import numpy as np
from findiff import FinDiff
from scipy.integrate import simps
from scipy.interpolate import splrep, splev
from scipy.linalg.lapack import dgesv
from quadpy import quad
from itertools import combinations_with_replacement


def Set_Bait(cosmo, data):

    # Compute the reconstruction factors for each redshift bin. Has shape len(z)
    recon = compute_recon(cosmo, data)

    # Precompute some derivative terms. The derivative of P(k) w.r.t. to alpha_perp/alpha_par
    # only needs doing once and then can be scaled by the ratios of sigma8 values. This works because we
    # ignore the derivatives of Dfactor. Has shape (2, len(k), len(mu))
    derPalpha = compute_deriv_alphas(cosmo)

    return recon, derPalpha


def compute_recon(cosmo, data):

    muconst = 0.6
    kconst = 0.16

    nP = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0]
    r_factor = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5]
    r_spline = splrep(nP, r_factor)

    recon = np.empty(len(cosmo.z))
    kaiser_vec = data.bias + cosmo.f * muconst ** 2
    nbar_comb = np.sum(data.nbar * kaiser_vec ** 2, axis=0)
    for iz in range(len(cosmo.z)):
        nP_comb = nbar_comb[iz] * splev(kconst, cosmo.pk[iz])
        if nP_comb <= nP[0]:
            recon[iz] = r_factor[0]
        elif nP_comb >= nP[-1]:
            recon[iz] = r_factor[-1]
        else:
            recon[iz] = splev(nP_comb, r_spline)
    return recon


def compute_deriv_alphas(cosmo):

    from scipy.interpolate import RegularGridInterpolator

    order = 5
    dalpha = 0.0025

    nmu = 100
    musq = np.linspace(0.0, 1.0, nmu)

    pkarray = np.empty((2 * order + 1, 2 * order + 1, len(cosmo.k), nmu))
    for i in range(-order, order):
        alpha_perp = 1.0 + i * dalpha
        for j in range(-order, order):
            alpha_par = 1.0 + j * dalpha
            kprime = np.outer(cosmo.k, np.sqrt((1.0 - musq) / alpha_perp ** 2 + musq / alpha_par ** 2))
            pkarray[i + order, j + order] = splev(kprime, cosmo.pk[0])

    derPalpha = [FinDiff(i, dalpha, acc=6)(pkarray)[order, order] for i in range(2)]
    derPalpha_interp = [RegularGridInterpolator([cosmo.k, musq], derPalpha[i]) for i in range(2)]

    return derPalpha_interp


def Fish(cosmo, data, iz, recon, derPalpha):
    """ Integrate over mu and k to compute the Fisher matrix for a given redshift bin

    :return:
    """

    npop = np.shape(data.nbar)[0]
    npk = int(npop * (npop + 1) / 2)

    OneFish = lambda *args: quad(CastNet, 0.0, 1.0, args=args, limit=1000)[0]
    ManyFish = quad(
        OneFish, cosmo.kmin, cosmo.kmax, args=(iz, npop, npk, data, cosmo, recon, derPalpha), limit=1000,
    )[0]

    # muvec = np.linspace(0.0, 1.0, 100)
    # kvec = np.linspace(cosmo.kmin, cosmo.kmax, 1000)
    # Fish = simps(
    #    simps(CastNet(muvec, kvec, iz, npop, npk, data, cosmo, recon, derPalpha), x=muvec, axis=-1),
    #    x=kvec,
    #    axis=-1,
    # )
    # print(OldFish / Fish)

    # Multiply by the necessary prefactors
    ManyFish *= cosmo.volume[iz] / (2.0 * np.pi ** 2)

    return ManyFish


def CastNet(mu, k, iz, npop, npk, data, cosmo, recon, derPalpha):
    """ Compute the Fisher matrix for a vector of k and mu at a particular redshift

    :return:
    """

    Shoal = np.empty((npop + 3, npop + 3, len(k), len(mu)))

    kaiser = np.tile(data.bias[:, iz], (len(mu), 1)).T + cosmo.f[iz] * mu ** 2
    Dpar = np.outer(np.outer(mu ** 2, k ** 2), cosmo.Sigma_par[iz] ** 2).reshape(len(mu), len(k))
    Dperp = np.outer(np.outer(1.0 - mu ** 2, k ** 2), cosmo.Sigma_perp[iz] ** 2).reshape(len(mu), len(k))
    Dfactor = np.exp(-(recon ** 2) * (Dpar + Dperp) / 2.0)

    pkval = splev(k, cosmo.pk[iz])
    coords = [[kval, muval] for kval in k for muval in mu]
    derPalphaval = [
        derPalpha[i](coords).reshape(len(k), len(mu)) * (cosmo.sigma8[iz] / cosmo.sigma8[0]) ** 2
        for i in range(2)
    ]
    for i, kval in enumerate(k):
        for j, muval in enumerate(mu):

            derP = compute_full_deriv(
                npop,
                npk,
                kaiser[:, j],
                pkval[i],
                muval,
                [derPalphaval[0][i, j], derPalphaval[1][i, j]],
                cosmo.f[iz],
                cosmo.sigma8[iz],
            )
            derP *= Dfactor[j, i]

            covP, cov_inv = compute_inv_cov(npop, npk, kaiser[:, j], pkval[i], data.nbar[:, iz])

            Shoal[:, :, i, j] = kval ** 2 * (derP @ cov_inv @ derP.T)

    return Shoal


def compute_inv_cov(npop, npk, kaiser, pk, nbar):

    covariance = np.empty((npk, npk))

    # Loop over power spectra of different samples P_12
    for ps1, pair1 in enumerate(combinations_with_replacement(range(npop), 2)):
        n1, n2 = pair1
        # Loop over power spectra of different samples P_34
        for ps2, pair2 in enumerate(combinations_with_replacement(range(npop), 2)):
            n3, n4 = pair2
            # Cov(P_12,P_34)
            pk13, pk24 = kaiser[n1] * kaiser[n3] * pk, kaiser[n2] * kaiser[n4] * pk
            pk14, pk23 = kaiser[n1] * kaiser[n4] * pk, kaiser[n2] * kaiser[n3] * pk
            if n1 == n3:
                pk13 += 1.0 / nbar[n1]
            if n1 == n4:
                pk14 += 1.0 / nbar[n1]
            if n2 == n3:
                pk23 += 1.0 / nbar[n2]
            if n2 == n4:
                pk14 += 1.0 / nbar[n1]
            covariance[ps1, ps2] = pk13 * pk24 + pk14 * pk23

    identity = np.eye(npk)
    cov_lu, pivots, cov_inv, info = dgesv(covariance, identity)

    return covariance, cov_inv


def compute_full_deriv(npop, npk, kaiser, pk, mu, derPalpha, f, sigma8):

    derP = np.zeros((npop + 3, npk))

    # Derivatives of all power spectra w.r.t to the bsigma8 of each population
    for i in range(npop):
        derP[i, int(i * (npop + (1 - i) / 2))] = 2.0 * kaiser[i] * pk / sigma8
        derP[i, int(i * (npop + (1 - i) / 2)) + 1 : int((i + 1) * (npop - i / 2))] = (
            kaiser[i + 1 :] * pk / sigma8
        )
        for j in range(0, i):
            derP[i, i + int(j * (npop - (1 + j) / 2))] = kaiser[j] * pk / sigma8

    # Derivatives of all power spectra w.r.t fsigma8
    derP[npop, :] = [
        (kaiser[i] + kaiser[j]) * mu ** 2 * pk / sigma8 for i in range(npop) for j in range(i, npop)
    ]

    # Derivatives of all power spectra w.r.t the alphas centred on alpha_per = alpha_par = 1.0
    # Derivative of mu'**2 w.r.t alpha_perp. Derivative w.r.t. alpha_par is -dmudalpha
    dmudalpha = 2.0 * mu ** 2 * (1.0 - mu ** 2)

    # We then just need use the product rule as we already precomputed dP(k')/dalpha
    derP[npop + 1, :] = [
        (kaiser[i] + kaiser[j]) * f * pk * dmudalpha + kaiser[i] * kaiser[j] * derPalpha[0]
        for i in range(npop)
        for j in range(i, npop)
    ]
    derP[npop + 2, :] = [
        -(kaiser[i] + kaiser[j]) * f * pk * dmudalpha + kaiser[i] * kaiser[j] * derPalpha[1]
        for i in range(npop)
        for j in range(i, npop)
    ]

    return derP
