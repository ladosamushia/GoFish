import copy
import numpy as np


class InputData:
    def __init__(self, pardict):

        df = self.read_nbar(pardict)

        self.zmin = df[:, 0]
        self.zmax = df[:, 1]
        self.nz = df[:, 2::2].T
        self.bias = df[:, 3::2].T

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


# This class contains everything we might need to set up to compute the fisher matrix
class CosmoResults:
    def __init__(self, pardict, zlow, zhigh):

        (
            self.z,
            self.volume,
            self.k,
            self.pk,
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
        pardict: dict
            A dictionary of parameters read from the config file
        zlow: np.array
            An array containing the lower limits of the redshift bins
        zhigh: np.array
            An array containing the upper limits of the redshift bins

        Returns
        -------
        camb_results: dict
            A dictionary collating all the cosmological parameters we might need
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
        if "H0" not in parlinear.keys():
            if "h" in parlinear.keys():
                parlinear["H0"] = 100.0 * float(parlinear["h"])
            else:
                print("Error: Neither H0 nor h given in config file")
                exit()
        if "w0_fld" in parlinear.keys():
            pars.set_dark_energy(w=float(parlinear["w0_fld"]), dark_energy_model="fluid")
        pars.InitPower.set_params(As=float(parlinear["A_s"]), ns=float(parlinear["n_s"]))
        pars.set_matter_power(redshifts=np.concatenate([zmid[::-1], [0.0]]), kmax=float(parlinear["kmax"]))
        pars.set_cosmology(
            H0=float(parlinear["H0"]),
            omch2=float(parlinear["omega_cdm"]),
            ombh2=float(parlinear["omega_b"]),
            omk=float(parlinear["Omega_k"]),
            tau=float(parlinear["tau_reio"]),
            mnu=float(parlinear["Sum_mnu"]),
            neutrino_hierarchy=parlinear["nu_hierarchy"],
        )
        pars.NonLinear = camb.model.NonLinear_none

        # Run CAMB
        results = camb.get_results(pars)

        # Get the power spectrum
        kin, zin, pklin = results.get_matter_power_spectrum(
            minkh=2.0e-5, maxkh=1.1 * float(parlinear["kmax"]), npoints=200
        )

        # Get some derived quantities
        area = float(pardict["skyarea"]) * (np.pi / 180.0) ** 2
        rmin = results.comoving_radial_distance(zlow) * float(parlinear["H0"]) / 100.0
        rmax = results.comoving_radial_distance(zhigh) * float(parlinear["H0"]) / 100.0
        volume = area / 3.0 * (rmax ** 3 - rmin ** 3)
        da = results.angular_diameter_distance(zmid)
        hubble = results.hubble_parameter(zmid)
        fsigma8 = results.get_fsigma8()[::-1][1:]
        sigma8 = results.get_sigma8()[::-1][1:]
        r_d = results.get_derived_params()["rdrag"]
        f = fsigma8 / sigma8
        growth = sigma8 / results.get_sigma8()[-1]

        pk_splines = [splrep(kin, pklin[i + 1]) for i in range(len(zin[1:]))]

        return zin[1:], volume, kin, pk_splines, da, hubble, f, sigma8, growth, r_d

    def get_Sigmas(self, f, sigma8):
        """ Compute the nonlinear degradation of the BAO feature in the perpendicular and parallel direction

        Parameters
        ----------
        f: np.array
            The growth rate of structure in each redshift bin
        sigma8: np.array
            The linear matter variance in each redshift bin

        Returns
        -------
        Sigma_perp: np.array
            The BAO damping perpendicular to the line of sight
        Sigma_par: np.array
            The BAO damping parallel to the line of sight
        """

        # The growth factor G has been absorbed in sigma8(z) already.
        Sigma_perp = 9.4 * sigma8 / 0.9
        Sigma_par = (1.0 + f) * Sigma_perp

        return Sigma_perp, Sigma_par
