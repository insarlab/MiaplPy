[![Language](https://img.shields.io/badge/python-3.5%2B-blue.svg)](https://www.python.org/)


## MiNoPy ##
*MIami NOn linear phase inversion in PYthon*

An open source python package of InSAR processing with Non Linear phase inversion in full resolution. It reads a stack of coregistered SLCs and
produces time series of surface deformation. This package depends on MintPy for time series corrections.
It works with all data based on ISCE outputs format. The inversion is based on wrapped phase time series and it includes PTA, EMI and EVD techniques.
It also supports sequential inversion.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.


### 1. [Installation](./installation.md) ###

### 2. Workflow ###

The workflow starts with reading coregistered SLCs, then performs an inversion to get wrapped phase time series.
Interferograms are then unwrapped and different corrections are applied on the final time series.
Everything is followed by defined steps in the `minopyApp.py` and the input is a text file containing adjustable options (template_file)
Starting the software will create two templates in minopy folder: `minopyApp.cfg` and `smallbaselineApp.cfg`
Use `smallbaselineApp.cfg` only for mintpy corrections. 

examples:

```
- minopyApp.py template_file
- minopyApp.py template_file --start load_slc --stop unwrap
- minopyApp.py template_file --dostep ifgram
```

Use `minopyApp.py template_file --runfiles` to create run files, you may then run them one by one manually

Use `minopyApp.py -H` for a complete list of required options.

Use `minopyApp.py -h` for a help on the steps you need to run 

-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. 


**Example:** Here is an example of SAR stack pre-processed using ISCE:

Area: Pichincha volcano, Ecuador\
Dataset: Sentinel-1 Descending Track 142, 46 acquisitions, 2016.04.19 - 2017.12.28\
Size: ~318 Mb\
```
wget https://zenodo.org/record/5167298/files/PichinchaSenDT142.zip
unzip PichinchaSenDT142.zip
cd PichinchaSenDT142
minopyApp.py PichinchaSenDT142.template --dir ./minopy
```

#### Example tutorial in jupyter notebook [nbviewer](https://nbviewer.jupyter.org/github/geodesymiami/MiNoPy/blob/main/tutorial/minopyApp.ipynb)

[Guide for University of Miami HPC users](./HPC_Users.md)

### 3. Contribution ###
Please follow the [guidelines](./CONTRIBUTING.md) for contributing to the code

### 4. Citation ###

S. Mirzaee, F. Amelung, and H. Fattahi, Non-linear phase inversion package for time series
analysis, in AGU Fall Meeting Abstracts, Dec. 2019, vol. 2019, pp. G13C-0572.

S. Mirzaee and F. Amelung, Volcanic Activity Change Detection Using SqueeSAR-InSAR and
Backscatter Analysis, in AGU Fall Meeting Abstracts, Dec. 2018, vol. 2018, pp. G41B-0707.



