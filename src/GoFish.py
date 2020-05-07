import sys
from configobj import ConfigObj
from src.ioutils import read_nbar, run_camb, convert_nbar


if __name__ == "__main__":

    # Read in the config file
    configfile = sys.argv[1]
    pardict = ConfigObj(configfile)

    # Read in the file containing the redshift bins, nz and bias values
    data = read_nbar(pardict)

    # Set up the linear power spectrum and derived parameters based on the input cosmology
    cosmo = run_camb(pardict, data["zmin"], data["zmax"])

    # Convert the nz to nbar in (h/Mpc)^3
    convert_nbar(data, cosmo["vol"], float(pardict["skyarea"]))

    # Compute the derivatives

    # Compute the covariance matrix

    # Compute the fisher matrix

    # Output the fisher matrix for each bin
