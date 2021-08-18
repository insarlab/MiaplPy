### Brief description of the steps: ###

minopyApp.py runs 8 steps sequentially from reading data to time series analysis. It uses [ISCE](https://github.com/isce-framework/isce2), [MintPy](https://github.com/insarlab/MintPy) and [PyAPS](https://github.com/AngeliqueBenoit/pyaps3) as extrenal modules and for correction steps, it uses MintPy.

You need to have a template with the options for each step like the [sample](https://github.com/geodesymiami/MiNoPy/blob/main/sample_input/PichinchaSenDT142.template). For a complete list of options, run `minopyApp.py -H`

Run `minopyApp.py -h` for a quick help on steps.
For more details refer to the [example](https://nbviewer.jupyter.org/github/geodesymiami/MiNoPy/blob/main/tutorial/minopyApp.ipynb) tutorial.

1. The first step is to read/load the coregistered SLC data and geometry files in full resolution. For that, it is recommended to set the options with `minopy.load.*` in your template. The ones related to interferograms are not required at this step. You only need SLC and geometry files. If your directory is set up following ISCE convention, you may set `minopy.load.autoPath` to `yes` and it will automatically read the data. Also you need to set subset area by specifying bounding box in `minopy.subset.lalo`. Processing time would be a matter if large subset is selected. 
After setting up your template file, run following command to load data:
```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep load_slc --dir $PWD/minopy
```

2. Second step would be the phase inversion (phase linking). In this step full network of wrapped phase series will be inverted using non linear phase linking methods including EVD, EMI, PTA, sequential_EVD, sequential_EMI and sequential_PTA. The default is sequential_EMI. 
It will process the data in parallel by dividing the subset into patches. You may set the number of workers in template `minopy.compute.numWorker` depending on your processing system availability which will be the number of parallel jobs. All options begining with `minopy.inversion.*` are used in this step. 

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep inversion --dir $PWD/minopy
```

3. After inversion you have the inverted single reference interferograms in a stack called `phase_series.h5`. You need to unwrap the interferograms for time series analysis but unwrapping is not easy specially when you have large temporal baselines. We like to unwrap minimum number of interferograms but the most correlated ones. In MiNoPy you can select which pairs to unwrap and for that you write selected pairs from the stack to separate ifgram directories. Use options starting with `minopy.interferograms.*` in template to select your network of interferograms to unwrap. The available options are: single reference, mini_stacks and delaunay (triangles). You may also write your own selected list in a text file and set the path to `minopy.interferograms.list`. For delaunay pairs, you can later perform both `bridging` and `phase_closure` unwrap error corrections of MintPy but for other pair networks you may only run `bridging`.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep ifgram --dir $PWD/minopy
```

4. The next step would be to unwrap the selected pairs. We use [SNAPHU](https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/) for unwrapping and you can set some options starting with `minopy.unwrap.*` in template.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep unwrap --dir $PWD/minopy
```

5. After unwrapping, The interferograms will be loaded to a stack in HDF5 format to be ready for mintpy time series analysis and correction steps.
You can now use `minopy.load.*` options in template specified for interferograms or set `minopy.load.autoPath = yes` to read them automatically.

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep load_ifgram --dir $PWD/minopy
```

6. At this step you will run the reference point selection and correct unwrap error using MintPy. Use the corresponding mintpy template options `mintpy.unwrapError.*`

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep correct_unwrap_error --dir $PWD/minopy
```

7. Now you need to convert phase to range change (time series). The temporal coherence threshold can be set for this step using `minopy.timeseries.minTempCoh` and you can use water mask by setting `minopy.timeseries.waterMask`

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep phase_to_range --dir $PWD/minopy
```

8. Finally, the time series is ready for different corrections including, tropospheric and topographic corrections. At this step you can use MintPy starting `correct_LOD` or run the following:

```
minopyApp.py $PWD/PichinchaSenDT142.template --dostep mintpy_corrections --dir $PWD/minopy
```


#### Post processing (Optional) ####
You can correct geolocation by running the following command but you need to do it after topographic residual correction step in MintPy. Also geocoding must be done after this post processing.

```
correct_geolocation.py -g ./minopy/inputs/geometryRadar.h5 -d ./minopy/demErr.h5
```
