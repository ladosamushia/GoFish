import copy
import numpy as np


class InputData:
    def __init__(self, pardict):

        df = self.read_nbar(pardict)

        self.zmin = df[:, 0]
        self.zmax = df[:, 1]
        self.nz = df[:, 2::2].T
        self.bias = df[:, 3::2].T

        # Sort out any tracers without galaxies in a particular redshift bin
        self.remove_null_tracers()

    def read_nbar(self, pardict):
        """ Reads redshift edges, number density, and bias from an input file

        Parameters
        ----------
        pardict:

        Returns
        -------
        df: np.array
            An array containing the input data

        """
        import pandas as pd

        df = np.array(
            pd.read_csv(
                pardict["inputfile"],
                comment="#",
                delim_whitespace=True,
                dtype="float",
                header=None,
                skiprows=0,
            )
        )

        return df

    def convert_nbar(self, volume, skyarea):
        """ Converts the number of galaxies per sq. deg. per dz into number density in (h/Mpc)^3

        Parameters
        ----------
        data: dict
            A dictionary containing redshift edges, number of galaxy, biases. Gets updated with new nbar values
        volume: np.array
            An array containing the comoving volume of each redshift bin
        skyarea: float
            The skyarea in sq. deg.

        Returns
        -------
        None
        """

        dz = self.zmax - self.zmin
        self.nbar = skyarea * self.nz * dz / volume

    def scale_bias(self, growth):
        self.bias /= growth

    def remove_null_tracers(self):

        """ Sorts out any tracers that have zero number density in a particular redshift bin.
            Does this my setting the bias to 0.0 in that bin and the number density to a very small number.
            This ensures their is no information in that bin from that tracer.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        index = np.where(self.nz == 0)
        self.bias[index] = 0.0
        self.nz[index] = 1.0e-30


# This class contains everything we might need to set up to compute the fisher matrix
class CosmoResults:
    def __init__(self, pardict, zlow, zhigh):

        (
            self.z,
            self.volume,
            self.k,
            self.pk,
            self.pksmooth,
            self.da,
            self.h,
            self.f,
            self.sigma8,
            self.growth,
            self.r_d,
        ) = self.run_camb(pardict, zlow, zhigh)
        self.Sigma_perp, self.Sigma_par = self.get_Sigmas(self.f, self.sigma8)

        self.kmin = np.amax([float(pardict["kmin"]), self.k[0]])
        self.kmax = float(pardict["kmax"])

    def run_camb(self, pardict, zlow, zhigh):

        """ Runs an instance of CAMB given the cosmological parameters in pardict and redshift bins

        Parameters
        ----------
        pardict: configobj.ConfigObj
            A dictionary of parameters read from the config file
        zlow: np.ndarray
            An array containing the lower limits of the redshift bins
        zhigh: np.ndarray
            An array containing the upper limits of the redshift bins

        Returns
        -------
        zmid: np.ndarray
            The midpoint of each redshift bin. The value at which all cosmological quantities are computed
        volume: np.ndarray
            The volume of each redshift bin in Mpc^3/h^3
        kin: np.ndarray
            The k values of the computed linear power spectra in unit h/Mpc
        pk_splines: list
            A list of scipy.intepolate.splrep objects. Each one contains a spline representation of the linear
            power spectrum in units of Mpc^3/h^3 at a particular zmid value. Can be interpolated using scipy.interpolate.splev
        pksmooth_splines: list
            A list of scipy.intepolate.splrep objects. Each one contains a spline representation of the smoothed linear
            power spectrum (the dewiggled spectrum) at a particular zmid value. Can be interpolated using scipy.interpolate.splev
        da: np.ndarray
            The angular diameter distance in Mpc at each zmid
        hubble: np.ndarray
            The Hubble parameter H(z) in km/s/Mpc at each zmid
        f: np.ndarray
            The growth rate of structure at each zmid
        sigma8: np.ndarray
            The square root of the variance in the matter field in spheres of radius 8 Mpc/h at each zmid. Has units Mpc/h.
        r_d: float
            The radius of the sound horizon at the baryon-drag epoch in Mpc
        """

        import camb
        from scipy.interpolate import splrep

        parlinear = copy.deepcopy(pardict)

        zmid = (zhigh + zlow) / 2.0

        # Set the CAMB parameters
        pars = camb.CAMBparams()
        if "A_s" not in parlinear.keys():
            if "ln10^{10}A_s" in parlinear.keys():
                parlinear["A_s"] = np.exp(float(parlinear["ln10^{10}A_s"])) / 1.0e10
            else:
                print("Error: Neither ln10^{10}A_s nor A_s given in config file")
                exit()
        if "H0" in parlinear.keys():
            parlinear["H0"] = float(parlinear["H0"])
            parlinear["thetastar"] = None
        elif "h" in parlinear.keys():
            parlinear["H0"] = 100.0 * float(parlinear["h"])
            parlinear["thetastar"] = None
        elif "thetastar" in parlinear.keys():
            parlinear["thetastar"] = float(parlinear["thetastar"])
            parlinear["H0"] = None
        else:
            print("Error: Neither H0 nor h nor theta_s given in config file")
            exit()
        if "w0_fld" in parlinear.keys():
            pars.set_dark_energy(w=float(parlinear["w0_fld"]), dark_energy_model="fluid")
        pars.InitPower.set_params(As=float(parlinear["A_s"]), ns=float(parlinear["n_s"]))
        pars.set_matter_power(redshifts=np.concatenate([zmid[::-1], [0.0]]), kmax=0.6)
        pars.set_cosmology(
            H0=parlinear["H0"],
            omch2=float(parlinear["omega_cdm"]),
            ombh2=float(parlinear["omega_b"]),
            omk=float(parlinear["Omega_k"]),
            tau=float(parlinear["tau_reio"]),
            mnu=float(parlinear["Sum_mnu"]),
            neutrino_hierarchy=parlinear["nu_hierarchy"],
            thetastar=parlinear["thetastar"],
        )
        pars.NonLinear = camb.model.NonLinear_none

        # Run CAMB
        results = camb.get_results(pars)

        # Get the power spectrum
        kin, zin, pklin = results.get_matter_power_spectrum(minkh=2.0e-5, maxkh=0.6, npoints=500)

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        rmin = results.comoving_radial_distance(zlow) * pars.H0 / 100.0
        rmax = results.comoving_radial_distance(zhigh) * pars.H0 / 100.0
        volume = area / 3.0 * (rmax ** 3 - rmin ** 3)
        da = results.angular_diameter_distance(zmid)
        hubble = results.hubble_parameter(zmid)
        fsigma8 = results.get_fsigma8()[::-1][1:]
        sigma8 = results.get_sigma8()[::-1][1:]
        r_d = results.get_derived_params()["rdrag"]
        f = fsigma8 / sigma8
        growth = sigma8 / results.get_sigma8()[-1]

        pk_splines = [splrep(kin, pklin[i + 1]) for i in range(len(zin[1:]))]
        pksmooth_splines = [
            splrep(kin, self.smooth_hinton2017(kin, pklin[i + 1])) for i in range(len(zin[1:]))
        ]

        return zmid, volume, kin, pk_splines, pksmooth_splines, da, hubble, f, sigma8, growth, r_d

    def get_Sigmas(self, f, sigma8):
        """ Compute the nonlinear degradation of the BAO feature in the perpendicular and parallel direction

        Parameters
        ----------
        f: np.ndarray
            The growth rate of structure in each redshift bin
        sigma8: np.ndarray
            The linear matter variance in each redshift bin

        Returns
        -------
        Sigma_perp: np.ndarray
            The BAO damping perpendicular to the line of sight
        Sigma_par: np.ndarray
            The BAO damping parallel to the line of sight
        """

        # The growth factor G has been absorbed in sigma8(z) already.
        Sigma_perp = 9.4 * sigma8 / 0.9
        Sigma_par = (1.0 + f) * Sigma_perp

        return Sigma_perp, Sigma_par

    def smooth_hinton2017(self, ks, pk, degree=13, sigma=1, weight=0.5):
        """ Smooth power spectrum based on Hinton et. al., 2017 polynomial method

        Parameters
        ----------
        ks: np.ndarray
            The k values of the input power spectrum
        pk: np.ndarray
            The input power spectrum
        degree: int (optional)
            The polynomial order used to fit the power spectrum. Default = 13
        sigma: float (optional)
            The width of the Gaussian weighting scheme used to avoid overfitting
            the BAO wiggles. Default = 1.0
        weight: float (optional)
            The amplitude of the Gaussian weighting scheme used to avoid overfitting
            the BAO wiggles. Default = 0.5. Hence default weights are 1.0 - 0.5*exp(-k^2/2.0)

        Returns
        -------
        pk_smoothed: np.ndarray
            The smooth (dewiggled) power spectrum at the input ks values
        """

        log_ks = np.log(ks)
        gauss = np.exp(-0.5 * ((log_ks - log_ks[np.argmax(pk)]) / sigma) ** 2)
        w = np.ones(pk.size) - weight * gauss
        z = np.polyfit(log_ks, np.log(pk), degree, w=w)
        p = np.poly1d(z)
        polyval = p(log_ks)
        pk_smoothed = np.exp(polyval)

        return pk_smoothed


def write_fisher(pardict, cov_inv, redshift, parameter_means):
    """ Write Fisher predictions to text files

    Parameters
    ---------
    pardict: configobj.ConfigObj
        A dictionary containing input parameters
    cov_inv: np.ndarray
        A 2D covariance matrix. fs8, da, h are the last three columns/rows
    redshift: np.float
        Mean redshift of that sample. Used in the filename
    parameter_means: list
        Contains mean values of fs8, da, h


    Will write a 3x3 covariance matrix of fs8, da, h and the true values of fs8, da, h.
    """

    cov_filename = pardict["outputfile"] + "_cov_" + format(redshift, ".2f") + ".txt"
    data_filename = pardict["outputfile"] + "_data_" + format(redshift, ".2f") + ".txt"

    np.savetxt(cov_filename, cov_inv[-3:, -3:])
    np.savetxt(data_filename, parameter_means)

    return
