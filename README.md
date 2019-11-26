# MiNoPy

A python package of InSAR processing with Non Linear inversion method working with ISCE and MintPy.
This package depends on MintPy and MinSAR now but will be independent from MinSAR in the near future.
It can only process Sentinel tops Satck (Isce outputs).


examples:

- minopy_wrapper.py template_file
- minopy_wrapper.py template_file --submit
- minopy_wrapper.py template_file --start crop --stop unwrap
- minopy_wrapper.py template_file --step ifgrams


Use 'minopy_wrapper.py -H' for a complete list of required options.
Use 'minopy_wrapper.py -h' for a help on the steps you need to run 

-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. Around 3 hours for 40 images covering an area of 1000*6000 pixels





