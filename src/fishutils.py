import numpy as np
from findiff import FinDiff
from scipy.interpolate import splrep, splev


def Fish(cosmo, data):
    """ Loop over each redshift bin and compute the Fisher matrix

    :return:
    """

    npop = np.shape(data["nbar"])[0]
    npk = int(npop * (npop + 1) / 2)
    nk = len(cosmo["kin"][0])
    mu = np.linspace(0.0, 1.0, 100)

    # Compute the reconstruction factors for each redshift bin
    recon = compute_recon(cosmo, data)

    # Compute all the kaiser factors for the various populations
    kaiser = np.einsum("nz,zm->nzm", cosmo["bias"], np.outer(cosmo["f"], mu ** 2))

    # Compute the damping factors for all redshifts
    Dpar = np.outer(cosmo["Sigma_par"] ** 2, np.outer(mu ** 2, cosmo["k"] ** 2))
    Dperp = np.outer(cosmo["Sigma_perp"] ** 2, np.outer(1.0 - mu ** 2, cosmo["k"] ** 2))
    Dfactor = np.exp(-(recon ** 2) * (Dpar + Dperp) / 2.0)

    # Precompute some derivative terms. The derivative of P(k) w.r.t. to alpha_perp/alpha_par
    # only needs doing once and then can be scaled by the ratios of sigma8 values. This works because we
    # ignore the derivatives of Dfactor.
    derPalpha = compute_deriv_alphas(cosmo, mu)

    for i, z in enumerate(cosmo["zin"]):
        print("z = {0:.2f}, V = {1:.2e} (Gpc/h)^3".format(z, cosmo["vol"][i] / 1e9))
        sigma8_ratio = cosmo["sigma8"][i] / cosmo["sigma8"][0]
        for j, k in enumerate(cosmo["k"]):
            for l, m in enumerate(mu):

                # Given we have precomputed derPalpha, the derivatives can be done analytically
                # which should be quite fast and we don't have to worry about numerical accuracy
                compute_full_deriv(
                    npop, npk, kaiser[:, i, l], cosmo["Pk"][i][j], mu, derPalpha[j, l] * sigma8_ratio ** 2
                )


def compute_recon(cosmo, data):

    muconst = 0.6
    kconst = 0.16

    nP = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 6.0, 10.0]
    r_factor = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5]
    r_spline = splrep(nP, r_factor)

    recon = np.empty(len(cosmo["zin"]))
    kaiser_vec = cosmo["bias"] + cosmo["f"] * muconst ** 2
    for i, (z, pk, nbar, kaiser) in enumerate(zip(cosmo["zin"], cosmo["Pk"], data["nbar"], kaiser_vec)):
        pk_spline = splrep(cosmo["k"], pk)
        nbar_comb = np.sum(nbar * kaiser ** 2) * splev(kconst, pk_spline)
        if nbar_comb <= nP[0]:
            recon[i] = r_factor[0]
        elif nbar_comb >= nP[-1]:
            recon[i] = r_factor[-1]
        else:
            recon[i] = splev(nbar_comb, r_spline)

    return recon


def compute_deriv_alphas(cosmo, mu):

    order = 4
    dalpha = 0.001
    pkspline = splrep(cosmo["k"], cosmo["Pk"][0])

    pkarray = np.empty((2 * order + 1, 2 * order + 1, len(cosmo["k"]), len(mu)))
    for i in range(-order, order):
        alpha_perp = 1.0 + i * dalpha
        for j in range(-order, order):
            alpha_par = 1.0 + j * dalpha
            kprime = cosmo["k"] * np.sqrt((1 - mu ** 2) / alpha_perp ** 2 + mu ** 2 / alpha_par ** 2)
            pkarray[i, j] = splev(kprime, pkspline)

    derPalpha = np.array([FinDiff(i, dalpha, acc=4)(pkarray)[order + 1, order + 1] for i in range(2)])

    return derPalpha


def compute_full_deriv(npop, npk, kaiser, pk, mu, derPalpha):

    derP = np.zeros(npop + 3, npk)

    return derP
