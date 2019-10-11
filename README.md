# MiNoPy

A python package of InSAR processing with Non Linear inversion method working with ISCE and MintPy.
This package depends on MintPy and MinSAR now but will be independent from MinSAR in the near future.
It can only process Sentinel tops Satck (Isce outputs).

To start working you need to specify following parameters in your template file:

- minopy.plmethod                       = auto         # [EVD, EMI, PTA, sequential_EVD, sequential_EMI, sequential_PTA] auto: EMI
- minopy.patch_size                     = auto         # patch size to divide the image for parallel processing, auto for 200
- minopy.range_window                   = auto         # range window size for synthetic multilook, auto for 15
- minopy.azimuth_window                 = auto         # azimuth window size for synthetic multilook, auto for 21
- minopy.subset                         = None         # [ -1 0.15 -91.7 -90.9] required [S N W E]
- minopy.shp_test                       = auto         # [ks, ad, ttest] auto for ks  
- processingMethod                      = minopy


After Processing with ISCE through SLC or interferogram workflow, you will have coregistered SLCs. Then you can follow the workflow given by minopy_wrapper.py

It has 7 steps:
- 'crop',
- 'patch',
- 'inversion',
- 'ifgrams',
- 'unwrap',
- 'mintpy',
- 'email']

examples:

- minopy_wrapper.py template_file
- minopy_wrapper.py template_file --submit
- minopy_wrapper.py template_file --start crop --stop unwrap
- minopy_wrapper.py template_file --step ifgrams


step 'mintpy' is the time series corrections from mintpy and is calling the script 'timeseries_corrections.py'. It has similar 
structure as smallbaselineApp.py with slightly modification to remove smallbaseline network inversion.


-- Note:
Inversion step may take long time depending on the number of pixels in the subset area you are processing and number of images. Around 3 hours for 40 images covering an area of 1000*6000 pixels





