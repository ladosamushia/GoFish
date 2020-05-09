import sys
import numpy as np
from configobj import ConfigObj
from src.TackleBox import Set_Bait, Fish
from src.ioutils import CosmoResults, InputData


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

    # Precompute some things we might need for the Fisher matrix
    recon, Dfactor, derPalpha = Set_Bait(cosmo, data)

    # Loop over redshifts and compute the Fisher matrix and output the 3x3 matrix
    for iz in range(len(cosmo.z)):
        print("z = {0:.2f}, V = {1:.2e} (Gpc/h)^3".format(cosmo.z[iz], cosmo.volume[iz] / 1e9))
        Fish(cosmo, data, iz, recon[iz], Dfactor[:, :, iz], derPalpha)

    # Output the fisher matrix for each bin
