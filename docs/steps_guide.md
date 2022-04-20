### Brief description of the steps: ###

miaplpyApp.py runs 9 steps sequentially from reading data to time series analysis. It uses [ISCE](https://github.com/isce-framework/isce2), [MintPy](https://github.com/insarlab/MintPy) and [PyAPS](https://github.com/AngeliqueBenoit/pyaps3) as extrenal modules and for correction steps, it uses MintPy.

You need to have a configuration text file with the options for each step like the [sample](https://github.com/insarlab/MiaplPy/blob/main/sample_input/PichinchaSenDT142.txt). For a complete list of options, run `miaplpyApp.py -H`

Run `miaplpyApp.py -h` for a quick help on steps.
For more details refer to the [example](https://nbviewer.jupyter.org/github/geodesymiami/MiaplPy/blob/main/tutorial/miaplpyApp.ipynb) tutorial.

1. The first step is to read/load the coregistered SLC data and geometry files in full resolution. For that, 
it is recommended to set the options with `miaplpy.load.*` in your template. The ones related to interferograms 
are not required at this step. You only need SLC and geometry files. If your directory is set up following ISCE 
convention, you may set `miaplpy.load.autoPath` to `yes` and it will automatically read the data. 
Also you need to set subset area by specifying bounding box in `miaplpy.subset.lalo`. 
Processing time would be a matter if large subset is selected. 
After setting up your template file, run following command to load data. It will call the `load_slc.py` script. 
```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep load_data --dir $PWD/miaplpy
```

2. Second step would be the phase linking. 
In this step full network of wrapped phase series will be inverted using non-linear 
phase linking methods including EVD, EMI, PTA, sequential_EVD, sequential_EMI (default) and 
sequential_PTA. It will process the data in parallel by dividing the subset into patches. 
You may set the number of workers in configuration file `miaplpy.multiprocessing.numProcessor` depending on 
your processing system availability which will be the number of parallel jobs. 
All options begining with `miaplpy.inversion.*` are used in this step. Patch size is the dimension
of your patches, for example 200 by 200 as default. ministack size is the number of images used for inverting 
each mini stack. Range and Azimuth window are the size of searching window to find SHPs. 
Statistical test to find SHPs can be selected among KS (default), AD and ttest. Following command will call `phase_inversion.py` script

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep phase_linking --dir $PWD/miaplpy
```

3. Third step is to concatenate the patches created in previous step. 

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep concatenate_patch --dir $PWD/miaplpy
```

4. After phase linking you have the single reference interferograms in a stack called `phase_series.h5`. You need to unwrap the interferograms for time series analysis but unwrapping is not easy specially when you have large temporal baselines. We like to unwrap minimum number of interferograms but the most correlated ones. In MiaplPy you can select which pairs to unwrap and for that you write selected pairs from the stack to separate ifgram directories. Use options starting with `miaplpy.interferograms.*` in template to select your network of interferograms to unwrap. The available options are: single reference, mini_stacks and sequential. You may also write your own selected list in a text file and set the path to `miaplpy.interferograms.list`. For sequential pairs (more than 2 connections), you can later perform both `bridging` and `phase_closure` unwrap error corrections of MintPy but for other pair networks you may only run `bridging`. Following command will call `generate_ifgram.py` script.

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep generate_ifgram --dir $PWD/miaplpy
```

5. The next step would be to unwrap the selected pairs. We use [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) for unwrapping and you can set some options starting with `miaplpy.unwrap.*` in template. Following command will call `unwrap_ifgram.py` script.

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep unwrap_ifgram --dir $PWD/miaplpy
```

6. After unwrapping, The interferograms will be loaded to a stack in HDF5 format to be ready for mintpy time series analysis and correction steps.
You can now use `miaplpy.load.*` options in template specified for interferograms or set `miaplpy.load.autoPath = yes` to read them automatically. Following command will call `load_ifgram.py` script.

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep load_ifgram --dir $PWD/miaplpy
```
s
7. At this step you will run the modify network, reference point selection and correct unwrap error using MintPy. Use the corresponding mintpy template options `mintpy.unwrapError.*`. Following command will call `smallbaselineApp.py` script from MintPy.

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep ifgram_correction --dir $PWD/miaplpy
```

8. Now you need to convert phase to range change (time series). The temporal coherence threshold can be set for this step using `miaplpy.timeseries.minTempCoh` and you can use water mask by setting `miaplpy.timeseries.waterMask`. Following command will call `network_inversion.py` script.

```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep invert_network --dir $PWD/miaplpy
```s

9. Finally, the time series is ready for different corrections including, tropospheric and topographic corrections. At this step you can use MintPy starting `correct_LOD` or run the following. It will call `smallbaselineApp.py` script from MintPy.


```
miaplpyApp.py $PWD/PichinchaSenDT142.txt --dostep timeseries_correction --dir $PWD/miaplpy
```


#### Post processing (Optional) ####
You can correct geolocation by running the following command but you need to do it after topographic residual correction step in MintPy. Also geocoding must be done after this post processing.

```
correct_geolocation.py -g ./miaplpy/inputs/geometryRadar.h5 -d ./miaplpy/demErr.h5
```

#### Geocoding (resampling) ####
It is worth noting that MiaplPy products are in full resolution and the storage is usually a matter. 
The geocoding step of MintPy helps to reduce the size of final geocoded products by resampling the 
data to a grid of lower resolution. You can choose one of 'linear' or 'nearest' options for interpolation, 
you can change the output resolution in degrees and also there is an option to limit the subset area: 
Try playing with the following options to match your needs.

```
########## 11.1 geocode (post-processing)
# for input dataset in radar coordinates only
# commonly used resolution in meters and in degrees (on equator)
# 100,         60,          50,          30,          20,          10
# 0.000925926, 0.000555556, 0.000462963, 0.000277778, 0.000185185, 0.000092593
mintpy.geocode              = auto  #[yes / no], auto for yes
mintpy.geocode.SNWE         = auto  #[-1.2,0.5,-92,-91 / none ], auto for none, output extent in degree
mintpy.geocode.laloStep     = auto  #[-0.000555556,0.000555556 / None], auto for None, output resolution in degree
mintpy.geocode.interpMethod = auto  #[nearest], auto for nearest, interpolation method
mintpy.geocode.fillValue    = auto  #[np.nan, 0, ...], auto for np.nan, fill value for outliers.
```
