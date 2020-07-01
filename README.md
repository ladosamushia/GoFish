# GoFish
A simple and lightweight but user friendly Fisher forecast tool for DESI galaxy clustering.

You can specify your cosmology and the properties of your tracers in the configuration file.
You can take the 'config/test.ini' and 'input_files/DESI_BGS_nbar.txt'  that come with this package 
and modify them to suit your needs.

The code will create a set of two output text files for each redshift: one for covariance and one for 
the mean values (data).
The order of parameters in both the data and covaraince files is fs8, DA, H.
DA is in Megaparsecs and H is in Megaparsecs/km/s.

If you have any questions, comments, or complaints feel free to get in touch with
Cullan Howlett (chowlett@uq.edu.au)
Lado Samushia (lado.samushia.office@gmail.com)

