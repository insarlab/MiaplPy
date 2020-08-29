[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)

## MiNoPy ##
*MIami NOn linear phase inversion in PYthon*

An open source python package of InSAR processing with Non Linear phase inversion in full resolution. It reads a stack of coregistered SLCs and
produces time series of surface deformation. This package depends on MintPy for time series corrections and MinSAR for job submission.
It works with all data based on ISCE outputs format. The inversion is based on wrapped phase time series and it includes PTA, EMI and EVD techniques.
It also supports sequential inversion.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.


### 1. [Installation](./installation.md) ###

### 2. Workflow ###

The workflow starts with reading coregistered SLCs, then performs an inversion to get wrapped phase time series.
Interferograms are then unwrapped and different corrections are applied on the final time series.
Everything is followed by defined steps in the `minopy_wrapper.py` and the input is a text file containing adjustable options (template_file)

examples:

- minopy_wrapper.py template_file
- minopy_wrapper.py template_file --submit
- minopy_wrapper.py template_file --start crop --stop unwrap
- minopy_wrapper.py template_file --dostep ifgrams


Use 'minopy_wrapper.py -H' for a complete list of required options.

Use 'minopy_wrapper.py -h' for a help on the steps you need to run 

-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. 


**Example:** Here is an example of SAR stack pre-processed using ISCE:

Area: Pichincha volcano, Ecuador\
Dataset: Sentinel-1 Ascending Track 18, 23 acquisitions, 2019.01.01 - 2019.06.12\
Size: ~340 Mb\
```
wget https://zenodo.org/record/4007068/files/PichinchaSenAT18.zip
tar -xvJf PichinchaSenAT18.zip
cd PichinchaSenAT18
minopy_wrapper.py PichinchaSenAT18.template 
```


### 3. Contribution ###
Please follow the [guidlines](./CONTRIBUTING.md) for contributing to the code

### 4. Citation ###

Sara. Mirzaee , Falk. Amelung, Heresh Fattahi, "Non-linear phase inversion package for time series analysis (MiNoPy)", AGUFM, 2019, pp.G13C0572M


