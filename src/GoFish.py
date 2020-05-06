import sys
from configobj import ConfigObj
import ioutils

if __name__ == "__main__":

    # Read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Read in the file containing the redshift bins, nz and bias values
    data = ioutils.read_nbar(pardict)
    zmid = (data["zmin"] + data["zmax"]) / 2.0

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = ioutils.run_camb(pardict, zmid)

    # Compute the derivatives

    # Compute the covariance matrix

    # Compute the fisher matrix

    # Output the fisher matrix for each bin
