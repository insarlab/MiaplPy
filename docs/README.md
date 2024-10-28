[![Language](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-yellow.svg)](https://github.com/insarlab/MiaplPy/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-v0.2.0-yellowgreen.svg)](https://github.com/insarlab/MiaplPy/releases)


## MiaplPy ##
*MIAmi Phase Linking in PYthon*

An open source python package of InSAR processing with Non Linear phase inversion in full resolution. It reads a stack of coregistered SLCs and
produces time series of surface deformation. This package depends on MintPy for time series corrections.
It works with all data based on ISCE or GAMMA outputs format. The inversion is based on wrapped phase time series and it includes PTA, EMI and EVD techniques.
It also supports sequential inversion.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS. USE AT YOUR OWN RISK.


### 1. [Installation](./installation.md) ###

### 2. Workflow ###

The workflow starts with reading coregistered SLCs, then performs an inversion to get wrapped phase time series.
Interferograms are then unwrapped and different corrections are applied on the final time series.
Everything is followed by defined steps in the `miaplpyApp.py` and the input is a text file containing adjustable options (configuration file)
Starting the software will create two configuration files in miaplpy folder: `miaplpyApp.cfg` and `custom_smallbaselineApp.cfg`
Use `custom_smallbaselineApp.cfg` only for mintpy corrections. 

examples:

```
- miaplpyApp.py config_file
- miaplpyApp.py config_file --start load_data --stop unwrap_ifgram
- miaplpyApp.py config_file --dostep generate_ifgram
```

Use `miaplpyApp.py config_file --runfiles` to create run files, you may then run them one by one manually

Use `miaplpyApp.py -H` for a complete list of required options.

Use `miaplpyApp.py -h` for a help on the steps you need to run 

-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. 


**Example:** 

Here are examples of SAR stack pre-processed using **ISCE**:

Area: Pichincha volcano, Ecuador\
Dataset: Sentinel-1 Descending Track 142, 46 acquisitions, 2016.04.19 - 2017.12.28\
Size: ~318 MB\
```
wget https://zenodo.org/record/6539952/files/PichinchaSenDT142.zip
unzip PichinchaSenDT142.zip
cd PichinchaSenDT142
miaplpyApp.py PichinchaSenDT142.txt --dir ./miaplpy
```


Area: Miami, USA\
Dataset: Sentinel-1 Ascending Track 48, 147 acquisitions, 2015.09.21 - 2021.11.12\
Size: ~1.3 GB\
```
wget https://zenodo.org/record/7470050/files/Miami_Sentinel1_data.tar.xz
tar -xvJf Miami_Sentinel1_data.tar.xz
cd Miami_Sentinel1_data
miaplpyApp.py Miami_Sentinel1_data.template --start phase_linking --dir ./miaplpy
```

Here are examples of SAR stack pre-processed using **GAMMA**:

Area: Pichincha volcano, Ecuador\
Dataset: Sentinel-1 Descending Track 142, 52 acquisitions, 2016.04.19 - 2017.12.28\
Size: ~544 MB\
```
wget https://zenodo.org/records/14001005/files/PichinchaSenDT142_gamma.zip
unzip PichinchaSenDT142_gamma.zip
cd PichinchaSenDT142_gamma
miaplpyApp.py gamma_parameters.txt --dir ./miaplpy
```

#### Example tutorial in jupyter notebook [nbviewer](https://nbviewer.org/github/insarlab/MiaplPy_notebooks/blob/main/miaplpyApp.ipynb)

#### [Brief description of the steps](https://github.com/insarlab/MiaplPy/blob/main/docs/steps_guide.md)

#### [Guide for HPC users](./HPC_Users.md)

### 3. Contribution ###
Please follow the [guidelines](./CONTRIBUTING.md) for contributing to the code

### 4. Citation ###

Mirzaee, S., Amelung, F., Fattahi, H., 2023. Non-linear phase linking using joined distributed and persistent scatterers. Comput Geosci 171, 105291. https://doi.org/10.1016/j.cageo.2022.105291.

S. Mirzaee, F. Amelung, and H. Fattahi, Non-linear phase inversion package for time series
analysis, in AGU Fall Meeting Abstracts, Dec. 2019, vol. 2019, pp. G13C-0572.

S. Mirzaee and F. Amelung, Volcanic Activity Change Detection Using SqueeSAR-InSAR and
Backscatter Analysis, in AGU Fall Meeting Abstracts, Dec. 2018, vol. 2018, pp. G41B-0707.



