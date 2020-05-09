import numpy as np
from findiff import FinDiff
from scipy.interpolate import splrep, splev


def Set_Bait(cosmo, data):

    # Compute the reconstruction factors for each redshift bin. Has shape len(z)
    recon = compute_recon(cosmo, data)

    # Compute the damping factors for all redshifts. Has shape (len(mu), len(k), len(z))
    nk = len(cosmo.k)
    nmu = len(cosmo.mu)
    Dpar = np.outer(np.outer(cosmo.mu ** 2, cosmo.k ** 2), cosmo.Sigma_par ** 2).reshape(nmu, nk, -1)
    Dperp = np.outer(np.outer(1.0 - cosmo.mu ** 2, cosmo.k ** 2), cosmo.Sigma_perp ** 2).reshape(nmu, nk, -1)
    Dfactor = np.exp(-(recon ** 2) * (Dpar + Dperp) / 2.0)

    # Precompute some derivative terms. The derivative of P(k) w.r.t. to alpha_perp/alpha_par
    # only needs doing once and then can be scaled by the ratios of sigma8 values. This works because we
    # ignore the derivatives of Dfactor. Has shape (2, len(k), len(mu))
    derPalpha = compute_deriv_alphas(cosmo)

    return recon, Dfactor, derPalpha


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
        pk_spline = splrep(cosmo.k, cosmo.pk[iz])
        nP_comb = nbar_comb[iz] * splev(kconst, pk_spline)
        if nP_comb <= nP[0]:
            recon[iz] = r_factor[0]
        elif nP_comb >= nP[-1]:
            recon[iz] = r_factor[-1]
        else:
            recon[iz] = splev(nP_comb, r_spline)
    return recon


def compute_deriv_alphas(cosmo):

    order = 4
    dalpha = 0.001
    pkspline = splrep(cosmo.k, cosmo.pk[0])

    pkarray = np.empty((2 * order + 1, 2 * order + 1, len(cosmo.k), len(cosmo.mu)))
    for i in range(-order, order):
        alpha_perp = 1.0 + i * dalpha
        for j in range(-order, order):
            alpha_par = 1.0 + j * dalpha
            kprime = np.outer(
                cosmo.k, np.sqrt((1.0 - cosmo.mu ** 2) / alpha_perp ** 2 + cosmo.mu ** 2 / alpha_par ** 2)
            )
            pkarray[i, j] = splev(kprime, pkspline)

    derPalpha = np.array([FinDiff(i, dalpha, acc=4)(pkarray)[order + 1, order + 1] for i in range(2)])

    return derPalpha


def Fish(cosmo, data, iz, recon, Dfactor, derPalpha):
    """ Loop over each k and mu and compute the Fisher matrix for a given redshift bin

    :return:
    """

    npop = np.shape(data.nbar)[0]
    npk = int(npop * (npop + 1) / 2)
    nk = len(cosmo.k)

    sigma8_ratio = cosmo.sigma8[iz] / cosmo.sigma8[0]
    kaiser_vec = np.tile(data.bias[:, iz], (len(cosmo.mu), 1)).T + cosmo.f[iz] * cosmo.mu ** 2
    for j, k in enumerate(cosmo.k):
        for l, m in enumerate(cosmo.mu):

            # Given we have precomputed derPalpha, the derivatives can be done analytically
            # which should be quite fast and we don't have to worry about numerical accuracy
            compute_full_deriv(
                npop,
                npk,
                kaiser_vec[:, l],
                cosmo.pk[iz][j],
                m,
                derPalpha[:, j, l] * sigma8_ratio ** 2,
                cosmo.f[iz],
                cosmo.sigma8[iz],
            )


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
