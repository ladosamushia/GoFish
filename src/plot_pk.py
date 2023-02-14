# Simple code to use some GoFish routines to compute a model power spectrum
# with some error bars

import sys
import numpy as np
from configobj import ConfigObj
from ioutils import CosmoResults, InputData
from TackleBox import compute_recon
from scipy.integrate import simps
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Read in the file containing the redshift bins, nz and bias values
    data = InputData(pardict)

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(pardict, data.zmin, data.zmax)
    if np.any(data.volume > 0):
        cosmo.volume = data.volume * 1.0e9

    # Convert the nz to nbar in (h/Mpc)^3
    data.convert_nbar(cosmo.volume, float(pardict["skyarea"]))

    # Scales the bias so that it goes as b/G(z)
    if pardict.as_bool("scale_bias"):
        data.scale_bias(cosmo.growth)
    print("#  Data nbar")
    print(data.nbar)
    print("#  Data bias")
    print(data.bias)

    recon = compute_recon(cosmo, data)
    print("#  Data recon factor")
    print(recon)

    # Produce power spectra for each sample as a function of k and mu
    # Compute the kaiser factors for each galaxy sample at the redshift as a function of mu
    muvec = np.linspace(0.0, 1.0, 100)
    dkvec, dkout = 0.001, 0.005
    ncomb = int(np.round(dkout / dkvec))
    kout = (
        np.linspace(cosmo.kmin, cosmo.kmax, int((cosmo.kmax - cosmo.kmin) / dkout), endpoint=False)
        + dkout / 2.0
    )
    kvec = (
        np.linspace(cosmo.kmin, cosmo.kmax, int((cosmo.kmax - cosmo.kmin) / dkvec), endpoint=False)
        + dkvec / 2.0
    )
    print(kout, kvec)
    labels = [r"$\mathrm{BGS\,BRIGHT}$", r"$\mathrm{LRG}$", r"$\mathrm{ELG\_LOP}$", r"$\mathrm{Quasar}$"]
    labels2 = [r"$0.0 < z < 0.4$", r"$0.4 < z < 1.1$", r"$1.1 < z < 1.6$", r"$1.6 < z < 2.1$"]
    for iz, (label, label2, ymin, ymax) in enumerate(
        zip(
            labels,
            labels2,
            np.array([-500.0, -250.0, -100.0, -200.0]),
            np.array([1900.0, 2800.0, 850.0, 1500.0]),
        )
    ):
        kaiser = np.tile(data.bias[iz, iz], (len(muvec), 1)).T + cosmo.f[iz] * muvec ** 2

        # Compute the BAO damping factor parameter after reconstruction at the redshift of interest
        # as a function of k and mu.
        Dpar = np.outer(muvec ** 2, kvec ** 2) * cosmo.Sigma_par[iz] ** 2
        Dperp = np.outer(1.0 - muvec ** 2, kvec ** 2) * cosmo.Sigma_perp[iz] ** 2
        Dfactor = np.exp(-(recon[iz] ** 2) * (Dpar + Dperp) / 2.0)

        # Compute the model 2D power spectrum
        pkmod = (kaiser.T ** 2 * splev(kvec, cosmo.pk[iz]) * Dfactor).T
        pkmodsmooth = (kaiser.T ** 2 * splev(kvec, cosmo.pksmooth[iz]) * Dfactor).T

        # Integrate the 2D model over mu to get the multipoles
        L2 = 1.0 / 2.0 * (3.0 * muvec ** 2 - 1.0)
        L4 = 1.0 / 8.0 * (35.0 * muvec ** 4 - 30.0 * muvec ** 2 + 3.0)
        pk0 = simps(pkmod, muvec, axis=-1)
        pk2 = 5.0 * simps(pkmod * L2, muvec, axis=-1)
        pk4 = 9.0 * simps(pkmod * L4, muvec, axis=-1)
        pk0smooth = simps(pkmodsmooth, muvec, axis=-1)
        pk2smooth = 5.0 * simps(pkmodsmooth * L2, muvec, axis=-1)
        pk4smooth = 9.0 * simps(pkmodsmooth * L4, muvec, axis=-1)
        pk0cov = simps((pkmod + 1.0 / data.nbar[iz, iz]) ** 2, muvec, axis=-1) / cosmo.volume[iz]
        pk2cov = (
            2.0
            * 25.0
            * simps((pkmod + 1.0 / data.nbar[iz, iz]) ** 2 * L2 ** 2, muvec, axis=-1)
            / cosmo.volume[iz]
        )
        pk4cov = (
            2.0
            * 81.0
            * simps((pkmod + 1.0 / data.nbar[iz, iz]) ** 2 * L4 ** 2, muvec, axis=-1)
            / cosmo.volume[iz]
        )

        Vkout = 4.0 / 3.0 * np.pi * ((kout + dkout / 2.0) ** 3 - (kout - dkout / 2.0) ** 3)
        Vkvec = 4.0 / 3.0 * np.pi * ((kvec + dkvec / 2.0) ** 3 - (kvec - dkvec / 2.0) ** 3)
        pk0new = 4.0 * np.pi * np.sum((pk0 * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        pk2new = 4.0 * np.pi * np.sum((pk2 * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        pk4new = 4.0 * np.pi * np.sum((pk4 * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        pk0smoothnew = (
            4.0 * np.pi * np.sum((pk0smooth * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        )
        pk2smoothnew = (
            4.0 * np.pi * np.sum((pk2smooth * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        )
        pk4smoothnew = (
            4.0 * np.pi * np.sum((pk4smooth * kvec ** 2).reshape((-1, ncomb)), axis=-1) * dkvec / Vkout
        )
        pk0covnew = (
            2.0
            * (2.0 * np.pi) ** 4
            * np.sum((pk0cov * kvec ** 2).reshape((-1, ncomb)), axis=-1)
            * dkvec
            / Vkout ** 2
        )
        pk2covnew = (
            2.0
            * (2.0 * np.pi) ** 4
            * np.sum((pk2cov * kvec ** 2).reshape((-1, ncomb)), axis=-1)
            * dkvec
            / Vkout ** 2
        )
        pk4covnew = (
            2.0
            * (2.0 * np.pi) ** 4
            * np.sum((pk4cov * kvec ** 2).reshape((-1, ncomb)), axis=-1)
            * dkvec
            / Vkout ** 2
        )

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.83, 0.83])
        ax.errorbar(
            kout,
            kout * pk0new,
            yerr=kout * np.sqrt(pk0covnew),
            marker="o",
            markersize=4,
            color="r",
            mfc="w",
            linestyle="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        ax.errorbar(
            kout,
            kout * pk0new,
            yerr=kout * np.sqrt(pk0covnew),
            marker="None",
            color="r",
            linestyle="-",
            zorder=5,
            alpha=0.4,
        )
        ax.errorbar(
            kout,
            kout * pk2new,
            yerr=kout * np.sqrt(pk2covnew),
            marker="s",
            markersize=4,
            color="b",
            mfc="w",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=5,
        )
        ax.errorbar(
            kout,
            kout * pk2new,
            yerr=kout * np.sqrt(pk2covnew),
            marker="None",
            color="b",
            linestyle="-",
            zorder=5,
            alpha=0.4,
        )
        ax.errorbar(
            kout,
            kout * pk4new,
            yerr=kout * np.sqrt(pk4covnew),
            marker="D",
            markersize=4,
            color="g",
            mfc="w",
            linestyle="-",
            markeredgewidth=1.3,
            zorder=5,
        )
        ax.errorbar(
            kout,
            kout * pk4new,
            yerr=kout * np.sqrt(pk4covnew),
            marker="None",
            color="g",
            linestyle="-",
            zorder=5,
            alpha=0.4,
        )
        ax.axvline(x=0.1, color="k", ls="--", lw=1.3)
        ax.set_xlim(0.00, 0.30)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=14, labelpad=2)
        ax.set_ylabel(r"$k\times P(k)\,(h^{-2}\mathrm{Mpc^{2}})$", fontsize=14, labelpad=2)
        ax.tick_params(width=1.3)
        ax.tick_params("both", length=10, which="major")
        ax.tick_params("both", length=5, which="minor")
        for axis in ["top", "left", "bottom", "right"]:
            ax.spines[axis].set_linewidth(1.3)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(12)
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(12)
        ax.text(0.04, 0.93, label, fontsize=14, transform=ax.transAxes)
        ax.text(0.04, 0.86, label2, fontsize=14, transform=ax.transAxes)

        ax2 = fig.add_axes([0.65, 0.65, 0.32, 0.32])
        ax2.errorbar(
            kout,
            pk0new / pk0smoothnew,
            yerr=np.sqrt(pk0covnew) / pk0smoothnew,
            marker="o",
            markersize=4,
            color="r",
            mfc="w",
            ls="None",
            markeredgewidth=1.3,
            zorder=5,
        )
        ax2.errorbar(
            kout,
            pk0new / pk0smoothnew,
            yerr=np.sqrt(pk0covnew) / pk0smoothnew,
            marker="None",
            markersize=4,
            color="r",
            mfc="w",
            ls="-",
            alpha=0.4,
            zorder=5,
        )
        ax2.set_xlim(0.00, 0.50)
        ax2.set_ylim(0.91, 1.09)
        ax2.set_xlabel(r"$k\,(h\,\mathrm{Mpc^{-1}})$", fontsize=12, labelpad=0)
        ax2.set_ylabel(r"$P_{0}(k)/P_{0}^{\mathrm{nw}}(k)$", fontsize=12, labelpad=0)
        ax2.tick_params(width=1.1)
        ax2.tick_params("both", length=10, which="major")
        ax2.tick_params("both", length=5, which="minor")
        for axis in ["top", "left", "bottom", "right"]:
            ax2.spines[axis].set_linewidth(1.1)
        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(10)
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(10)

        plt.show()
