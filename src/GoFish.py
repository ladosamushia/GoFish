import sys
import numpy as np
import importlib
from configobj import ConfigObj
from TackleBox import Set_Bait, Fish
from ioutils import CosmoResults, InputData, write_fisher
from scipy.linalg.lapack import dgesv

if __name__ == "__main__":

    # Read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Read in the file containing the redshift bins, nz and bias values
    data = InputData(pardict)

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = CosmoResults(pardict, data.zmin, data.zmax)

    # Convert the nz to nbar in (h/Mpc)^3
    data.convert_nbar(cosmo.volume, float(pardict["skyarea"]))

    # Scales the bias so that it goes as b/G(z)
    data.scale_bias(cosmo.growth)

    # Precompute some things we might need for the Fisher matrix
    recon, derPalpha = Set_Bait(cosmo, data, BAO_only=pardict.as_bool("BAO_only"))

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix
    identity = np.eye(len(data.nbar) + 3)
    for iz in range(len(cosmo.z)):
        print("z = {0:.2f}, V = {1:.2e} (Gpc/h)^3".format(cosmo.z[iz], cosmo.volume[iz] / 1e9))
        Catch = Fish(cosmo, data, iz, recon[iz], derPalpha, BAO_only=pardict.as_bool("BAO_only"))
        cov_lu, pivots, cov, info = dgesv(Catch, identity)
        print(100.0 * np.sqrt(np.diag(cov)[-3:]) / np.array([cosmo.f[iz] * cosmo.sigma8[iz], 1.0, 1.0]))
        parameter_means = [cosmo.f[iz] * cosmo.sigma8[iz], cosmo.da[iz], cosmo.h[iz]]
        print(parameter_means)

        # Output the fisher matrix for each bin
        write_fisher(pardict, cov, cosmo.z[iz], parameter_means)
