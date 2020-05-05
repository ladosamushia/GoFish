import sys
import copy
import numpy as np
from configobj import ConfigObj


def read_nbar(pardict):

    import pandas as pd

    df = pd.read_csv(
        pardict["inputfile"],
        comment="#",
        delim_whitespace=True,
        names=["zmin", "zmax", "nbar", "bias"],
        dtype="float",
    )
    data = dict(zip(df.keys(), df.T.values))
    print(data)

    return data


def run_camb(pardict, zmid):
    """ Runs an instance of CAMB given the cosmological parameters in pardict and redshifs in data

    Parameters
    ----------
    pardict: dict
        A dictionary of parameters read from the config file
    zmid: np.array
        An array containing the redshifts we want to compute forecasts for

    Returns
    -------
    camb_results: dict
        A dictionary collating all the cosmological parameters we might need
    """

    import camb

    parlinear = copy.deepcopy(pardict)

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
    pars.set_matter_power(redshifts=zmid[::-1], kmax=float(parlinear["kmax"]))
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
    kin, zin, Plin = results.get_matter_power_spectrum(
        minkh=2.0e-5, maxkh=float(parlinear["kmax"]), npoints=2000
    )

    # Get some derived quantities
    Da = results.angular_diameter_distance(zmid)
    H = results.hubble_parameter(zmid)
    fsigma8 = results.get_fsigma8()[::-1]
    sigma8 = results.get_sigma8()[::-1]
    r_d = results.get_derived_params()["rdrag"]

    camb_results = {
        "z": zin,
        "k": kin,
        "Pk": Plin,
        "D_A": Da,
        "H": H,
        "f": fsigma8 / sigma8,
        "sigma8": sigma8,
        "r_d": r_d,
    }

    return camb_results


if __name__ == "__main__":

    # Read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Read in the file containing the redshift bins, nz and bias values
    data = read_nbar(pardict)
    zmid = (data["zmin"] + data["zmax"]) / 2.0

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = run_camb(pardict, zmid)

    # Compute the derivatives

    # Compute the covariance matrix

    # Compute the fisher matrix

    # Output the fisher matrix for each bin
