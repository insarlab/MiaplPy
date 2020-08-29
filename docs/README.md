[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)

## MiNoPy ##

A python package of InSAR processing with Non Linear inversion method working with ISCE and MintPy.
This package depends on MintPy for time series corrections and MinSAR for job submission.
It works with all data based on ISCE outputs format.


### 1. [Installation](./installation.md) ###


examples:

- minopy_wrapper.py template_file
- minopy_wrapper.py template_file --submit
- minopy_wrapper.py template_file --start crop --stop unwrap
- minopy_wrapper.py template_file --dostep ifgrams


Use 'minopy_wrapper.py -H' for a complete list of required options.

Use 'minopy_wrapper.py -h' for a help on the steps you need to run 

-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. 


### Citation ###

Sara. Mirzaee , Falk. Amelung, Heresh Fattahi, "Non-linear phase inversion package for time series analysis (MiNoPy)", AGUFM, 2019, pp.G13C0572M


