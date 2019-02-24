#!/usr/bin/env python3
######################################################################
# Project: part of PySAR                                             #
# Purpose: InSAR Time Series Analysis in Python using phase linking  #                                             #
# Author: Sara Mirzaee                                               #
# Copyright (c) 2018, Sara Mirzaee, Yunjun Zhang, Heresh Fattahi     #
######################################################################

import os
import re
import glob
import time
from datetime import datetime as dt
import argparse
import warnings
import shutil
import subprocess
import numpy as np
import gdal
from pysar.objects import sensor
from pysar.defaults.auto_path import autoPath
from pysar.utils import readfile, utils as ut
from pysar import version
from pysar.pysarApp import *


#####################

EXAMPLE = """example:
  plApp.py                                             #Run / Rerun
  plApp.py  SanAndreasT356EnvD.template  --load-data   #Exit after loading data into HDF5 files

  # Template options
  plApp.py -H                               #Print    default template
  plApp.py -g                               #Generate default template
  plApp.py -g SanAndreasT356EnvD.template   #Generate default template considering input custom template
"""


def create_parser():
    parser = argparse.ArgumentParser(description='PySAR Phase Linking and Time Series Analysis',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('--template', dest='customTemplateFile', type=str, help='custom template with option settings.\n' +
                             "It's equivalent to None if default pysarApp_template.txt is input.")
    parser.add_argument('--dir', dest='workDir',
                        help='PySAR working directory, default is:\n' +
                             'a) current directory, or\n' +
                             'b) $SCRATCHDIR/projectName/PYSAR, if meets the following 3 requirements:\n' +
                             '    1) autoPath = True in pysar/defaults/auto_path.py\n' +
                             '    2) environmental variable $SCRATCHDIR exists\n' +
                             '    3) input custom template with basename same as projectName\n')
    parser.add_argument('-g', dest='generate_template', action='store_true',
                        help='Generate default template (and merge with custom template), then exit.')
    parser.add_argument('-H', dest='print_auto_template', action='store_true',
                        help='Print/Show the example template file for routine processing.')
    parser.add_argument('--version', action='store_true', help='print version number')
    parser.add_argument('--reset', action='store_true',
                        help='Reset files attributes to re-run plApp.py after loading data by:\n' +
                             '    1) removing ref_y/x/lat/lon for unwrapIfgram.h5 and coherence.h5\n' +
                             '    2) set DROP_IFGRAM=no for unwrapIfgram.h5 and coherence.h5')
    parser.add_argument('--load-data', dest='load_dataset', action='store_true',
                        help='Step 1. Load/check dataset, then exit')

    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    inps.autoTemplateFile = os.path.join(os.path.dirname(__file__), '../docs/pysarApp_template.txt')

    if inps.print_auto_template:
        with open(inps.autoTemplateFile, 'r') as f:
            print(f.read())
        raise SystemExit()

    if (inps.customTemplateFile
            and os.path.basename(inps.customTemplateFile) == 'pysarApp_template.txt'):
        inps.customTemplateFile = None
    return inps



def get_phase_linking_coherence_mask(inps, template):
    """Generate mask from temporal coherence"""

    inps.maskFile = 'maskTempCoh.h5'




    maskCmd = 'generate_mask.py {} -m {} -M {} -o {} --shadow {}'.format(inps.tempCohFile,
                                                                   inps.vmin,
                                                                   inps.vmax,
                                                                   inps.maskFile,
                                                                   inps.geomFile)
    print(maskCmd)

    # update mode checking
    # run if 1) output file exists; 2) newer than input file and 3) all config keys are the same
    run = False
    if ut.run_or_skip(out_file=inps.maskFile, in_file=inps.tempCohFile, print_msg=False) == 'run':
        run = True
    else:
        print('  1) output file: {} already exists and newer than input file: {}'.format(inps.maskFile,
                                                                                         inps.tempCohFile))
        meta_dict = readfile.read_attribute(inps.maskFile)
        if any(str(template[i]) != meta_dict.get(i, 'False') for i in configKeys):
            run = True
            print('  2) NOT all key configration parameters are the same --> run.\n\t{}'.format(configKeys))
        else:
            print('  2) all key configuration parameters are the same:\n\t{}'.format(configKeys))
    # result
    print('run this step:', run)
    if run:
        status = subprocess.Popen(maskCmd, shell=True).wait()
        if status is not 0:
            raise Exception('Error while generating mask file from temporal coherence.')

    # check number of pixels selected in mask file for following analysis
    min_num_pixel = float(template['pysar.networkInversion.minNumPixel'])
    msk = readfile.read(inps.maskFile)[0]
    num_pixel = np.sum(msk != 0.)
    print('number of pixels selected: {}'.format(num_pixel))
    if num_pixel < min_num_pixel:
        msg = "Not enought coherent pixels selected (minimum of {}). ".format(int(min_num_pixel))
        msg += "Try the following:\n"
        msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
        msg += "2) Check the network and make sure it's fully connected without subsets"
        raise RuntimeError(msg)
    del msk
    return



#####################################################

def main(iargs=None):
    start_time = time.time()
    inps = cmd_line_parse(iargs)
    if inps.version:
        raise SystemExit(version.version_description)

    #########################################
    # Initiation
    #########################################
    print(version.logo)

    # Project Name
    inps.projectName = None
    if inps.customTemplateFile:
        inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)
        inps.projectName = os.path.splitext(os.path.basename(inps.customTemplateFile))[0]
        print('Project name:', inps.projectName)

    # Work directory
    if not inps.workDir:
        if autoPath and 'SCRATCHDIR' in os.environ and inps.projectName:
            inps.workDir = os.path.join(os.getenv('SCRATCHDIR'), inps.projectName, 'PYSAR')
        else:
            inps.workDir = os.getcwd()
    inps.workDir = os.path.abspath(inps.workDir)
    if not os.path.isdir(inps.workDir):
        os.makedirs(inps.workDir)
    os.chdir(inps.workDir)
    print("Go to work directory:", inps.workDir)

    copy_aux_file(inps)
    inps.templateFile = os.path.join(os.getenv('PYSAR_HOME'),'docs/pysarApp_template.txt')
    shutil.copy2(inps.templateFile, inps.workDir)
    inps, template, customTemplate = read_template(inps)

    #########################################
    # Loading Data
    #########################################
    print('\n**********  Load Data  **********')
    loadCmd = 'load_data.py --template {}'.format(inps.templateFile)
    if inps.customTemplateFile:
        loadCmd += ' {}'.format(inps.customTemplateFile)
    if inps.projectName:
        loadCmd += ' --project {}'.format(inps.projectName)
    print(loadCmd)
    status = subprocess.Popen(loadCmd, shell=True).wait()
    os.chdir(inps.workDir)

    print('-'*50)
    inps, atr = ut.check_loaded_dataset(inps.workDir, inps)

    # Add template options into HDF5 file metadata
    if inps.customTemplateFile:
        #metaCmd = 'add_attribute.py {} {}'.format(inps.stackFile, inps.customTemplateFile)
        #print(metaCmd)
        #status = subprocess.Popen(metaCmd, shell=True).wait()
        # better control of special metadata, such as SUBSET_X/YMIN
        print('updating {} metadata based on custom template file: {}'.format(
            os.path.basename(inps.stackFile), inps.customTemplateFile))
        ut.add_attribute(inps.stackFile, customTemplate)

    if inps.load_dataset:
        if inps.plot:
            plot_pysarApp(inps)
        raise SystemExit('Exit as planned after loading/checking the dataset.')

    if inps.reset:
        print('Reset dataset attributtes for a fresh re-run.\n'+'-'*50)
        # Reset reference pixel
        refPointCmd = 'reference_point.py {} --reset'.format(inps.stackFile)
        print(refPointCmd)
        status = subprocess.Popen(refPointCmd, shell=True).wait()
        # Reset network modification
        networkCmd = 'modify_network.py {} --reset'.format(inps.stackFile)
        print(networkCmd)
        status = subprocess.Popen(networkCmd, shell=True).wait()

    #########################################
    # Referencing Interferograms in Space
    #########################################
    print('\n**********  Select Reference Point  **********')
    # Initial mask (pixels with valid unwrapPhase or connectComponent in ALL interferograms)
    inps.maskFile = 'maskConnComp.h5'
    maskCmd = 'generate_mask.py {} --nonzero -o {} --update'.format(inps.stackFile, inps.maskFile)
    print(maskCmd)
    status = subprocess.Popen(maskCmd, shell=True).wait()

    # Average spatial coherence
    inps.avgSpatialCohFile = 'avgSpatialCoh.h5'
    avgCmd = 'temporal_average.py {i} --dataset coherence -o {o} --update'.format(i=inps.stackFile,
                                                                                  o=inps.avgSpatialCohFile)
    print(avgCmd)
    status = subprocess.Popen(avgCmd, shell=True).wait()

    # Select reference point
    refPointCmd = 'reference_point.py {} -t {} -c {}'.format(inps.stackFile,
                                                             inps.templateFile,
                                                             inps.avgSpatialCohFile)
    print(refPointCmd)
    status = subprocess.Popen(refPointCmd, shell=True).wait()
    if status is not 0:
        if inps.plot:
            plot_pysarApp(inps)
        raise Exception('Error while finding reference pixel in space.\n')


    ###############################################
    # Average velocity from interferogram stacking
    ###############################################
    print('\n**********  Quick assessment with interferogram stacking  **********')
    inps.waterMaskFile = 'waterMask.h5'
    if not os.path.isfile(inps.waterMaskFile):
        inps.waterMaskFile = None

    # Average phase velocity - Stacking
    inps.avgPhaseVelFile = 'avgPhaseVelocity.h5'
    avgCmd = 'temporal_average.py {i} --dataset unwrapPhase -o {o} --update'.format(i=inps.stackFile,
                                                                                    o=inps.avgPhaseVelFile)
    print(avgCmd)
    status = subprocess.Popen(avgCmd, shell=True).wait()

    # mask based on average spatial coherence
    inps.maskSpatialCohFile = 'maskSpatialCoh.h5'
    if ut.run_or_skip(out_file=inps.maskSpatialCohFile, in_file=inps.avgSpatialCohFile) == 'run':
        maskCmd = 'generate_mask.py {i} -m 0.7 -o {o}'.format(i=inps.avgSpatialCohFile,
                                                              o=inps.maskSpatialCohFile)
        if inps.waterMaskFile:
            maskCmd += ' --base {}'.format(inps.waterMaskFile)
        print(maskCmd)
        status = subprocess.Popen(maskCmd, shell=True).wait()

    ############################################
    # Unwrapping Error Correction (Optional)
    #    based on the consistency of triplets
    #    of interferograms
    ############################################
    #correct_unwrap_error(inps, template)

    #########################################
    # Writing into time series format
    ########################################

    from pysar.objects import ifgramStack, timeseries
    from pysar.ifgram_inversion import write2hdf5_file, read_unwrap_phase, mask_unwrap_phase

    inps.timeseriesFile = 'timeseries.h5'
    inps.tempCohFile = 'temporalCoherence.h5'
    inps.timeseriesFiles = ['timeseries.h5']       #all ts files
    inps.outfile = ['timeseries.h5', 'temporalCoherence.h5']

    ifgram_file = './INPUTS/ifgramStack.h5'
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    date_list = stack_obj.get_date_list(dropIfgram=True)
    num_date = len(date_list)

    metadata = dict(stack_obj.metadata)
    metadata['REF_DATE'] = date_list[0]
    metadata['FILE_TYPE'] = 'timeseries'
    metadata['UNIT'] = 'm'

    num_row = stack_obj.length
    num_col = stack_obj.width
    num_pixel = num_row * num_col
    ts_std = np.zeros((num_date, num_pixel), np.float32)
    ts_std = ts_std.reshape(num_date, num_row, num_col)

    phase2range = -1 * float(stack_obj.metadata['WAVELENGTH']) / (4. * np.pi)

    box = None
    ref_phase = stack_obj.get_reference_phase(dropIfgram=False)
    unwDatasetName = 'unwrapPhase'
    skip_zero_phase = True
    mask_dataset_name = None
    mask_threshold = 0.4

    pha_data = read_unwrap_phase(stack_obj,
                                 box,
                                 ref_phase,
                                 unwDatasetName=unwDatasetName,
                                 dropIfgram=True,
                                 skip_zero_phase=skip_zero_phase)

    pha_data = mask_unwrap_phase(pha_data,
                                 stack_obj,
                                 box,
                                 dropIfgram=True,
                                 mask_ds_name=mask_dataset_name,
                                 mask_threshold=mask_threshold)



    ts = pha_data * phase2range
    ts0 = ts.reshape(num_date-1,num_row,num_col)
    ts = np.zeros((num_date, num_row, num_col), np.float32)
    ts[1::,:,:] = ts0
    num_inv_ifg = np.zeros((num_row, num_col), np.int16) + num_date - 1


    gfilename = os.path.join(os.getenv('SCRATCHDIR'), inps.projectName,'merged/geom_master/Quality.rdr')
    ds = gdal.Open(gfilename, gdal.GA_ReadOnly)
    quality_map = ds.GetRasterBand(1).ReadAsArray()
    inps.plmethod = ds.GetMetadata()['plmethod']

    if 'EMI' in inps.plmethod:
        inps.vmin = 0.9
        inps.vmax = 1.012
    else:
        inps.vmin = 0.55
        inps.vmax = 1.5


    write2hdf5_file(ifgram_file, metadata, ts, quality_map, ts_std, num_inv_ifg, suffix='', inps=inps)

    get_phase_linking_coherence_mask(inps, template)


    ##############################################
    # Tropospheric Delay Correction (Optional)
    ##############################################
    print('\n**********  Tropospheric Delay Correction  **********')
    correct_tropospheric_delay(inps, template)

    ##############################################
    # Phase Ramp Correction (Optional)
    ##############################################
    print('\n**********  Remove Phase Ramp  **********')
    inps.derampMaskFile = template['pysar.deramp.maskFile']
    inps.derampMethod = template['pysar.deramp']
    if inps.derampMethod:
        print('Phase Ramp Removal method: {}'.format(inps.derampMethod))
        ramp_list = ['linear', 'quadratic',
                     'linear_range', 'quadratic_range',
                     'linear_azimuth', 'quadratic_azimuth']
        if inps.derampMethod in ramp_list:
            outName = '{}_ramp.h5'.format(os.path.splitext(inps.timeseriesFile)[0])
            derampCmd = 'remove_ramp.py {} -s {} -m {} -o {}'.format(inps.timeseriesFile,
                                                                     inps.derampMethod,
                                                                     inps.derampMaskFile,
                                                                     outName)
            print(derampCmd)
            if ut.run_or_skip(out_file=outName, in_file=inps.timeseriesFile) == 'run':
                status = subprocess.Popen(derampCmd, shell=True).wait()
                if status is not 0:
                    if inps.plot:
                        plot_pysarApp(inps)
                    raise Exception('Error while removing phase ramp for time-series.\n')
            inps.timeseriesFile = outName
            inps.timeseriesFiles.append(outName)
        else:
            msg = 'un-recognized phase ramp method: {}'.format(inps.derampMethod)
            msg += '\navailable ramp types:\n{}'.format(ramp_list)
            raise ValueError(msg)
    else:
        print('No phase ramp removal.')

    ##############################################
    # Topographic (DEM) Residuals Correction (Optional)
    ##############################################
    print('\n**********  Topographic Residual (DEM error) Correction  **********')
    outName = os.path.splitext(inps.timeseriesFile)[0]+'_demErr.h5'
    topoCmd = 'dem_error.py {i} -t {t} -o {o} --update '.format(i=inps.timeseriesFile,
                                                                t=inps.templateFile,
                                                                o=outName)
    #if not inps.fast:
    topoCmd += ' -g {}'.format(inps.geomFile)
    print(topoCmd)

    inps.timeseriesResFile = None
    if template['pysar.topographicResidual']:
        status = subprocess.Popen(topoCmd, shell=True).wait()
        if status is not 0:
            if inps.plot:
                plot_pysarApp(inps)
            raise Exception('Error while correcting topographic phase residual.\n')
        inps.timeseriesFile = outName
        inps.timeseriesResFile = 'timeseriesResidual.h5'
        inps.timeseriesFiles.append(outName)
    else:
        print('No correction for topographic residuals.')

    ##############################################
    # Phase Residual for Noise Evaluation
    ##############################################
    # Timeseries Residual RMS
    print('\n**********  Timeseries Residual Root Mean Square  **********')
    if inps.timeseriesResFile:
        rmsCmd = 'timeseries_rms.py {} -t {}'.format(inps.timeseriesResFile,
                                                     inps.templateFile)
        print(rmsCmd)
        status = subprocess.Popen(rmsCmd, shell=True).wait()
        if status is not 0:
            if inps.plot:
                plot_pysarApp(inps)
            raise Exception('Error while calculating RMS of residual phase time-series.\n')
    else:
        print('No timeseries residual file found! Skip residual RMS analysis.')

    # Reference in Time
    print('\n**********  Select Reference Date  **********')
    if template['pysar.reference.date']:
        refCmd = 'reference_date.py -t {} '.format(inps.templateFile)
        for fname in inps.timeseriesFiles:
            refCmd += ' {}'.format(fname)
        print(refCmd)
        status = subprocess.Popen(refCmd, shell=True).wait()
        if status is not 0:
            if inps.plot:
                plot_pysarApp(inps)
            raise Exception('Error while changing reference date.\n')
    else:
        print('No reference change in time.')

    #############################################
    # Velocity and rmse maps
    #############################################
    print('\n**********  Estimate Velocity  **********')
    inps.velFile = 'velocity.h5'
    velCmd = 'timeseries2velocity.py {} -t {} -o {} --update'.format(inps.timeseriesFile,
                                                                     inps.templateFile,
                                                                     inps.velFile)
    print(velCmd)
    status = subprocess.Popen(velCmd, shell=True).wait()
    if status is not 0:
        if inps.plot:
            plot_pysarApp(inps)
        raise Exception('Error while estimating linear velocity from time-series.\n')

    # Velocity from Tropospheric delay
    if inps.tropFile:
        suffix = os.path.splitext(os.path.basename(inps.tropFile))[0].title()
        inps.tropVelFile = '{}{}.h5'.format(os.path.splitext(inps.velFile)[0], suffix)
        velCmd = 'timeseries2velocity.py {} -t {} -o {} --update'.format(inps.tropFile,
                                                                         inps.templateFile,
                                                                         inps.tropVelFile)
        print(velCmd)
        status = subprocess.Popen(velCmd, shell=True).wait()

    ############################################
    # Post-processing
    # Geocodeing --> Masking --> KMZ & HDF-EOS5
    ############################################
    print('\n**********  Post-processing  **********')
    if template['pysar.save.hdfEos5'] is True and template['pysar.geocode'] is False:
        print('Turn ON pysar.geocode to be able to save to HDF-EOS5 format.')
        template['pysar.geocode'] = True

    # Geocoding
    if not inps.geocoded:
        if template['pysar.geocode'] is True:
            print('\n--------------------------------------------')
            geo_dir = os.path.join(inps.workDir, 'GEOCODE')
            geocode_script = os.path.join(os.path.dirname(__file__), 'geocode.py')
            geoCmd = ('{scp} {v} {c} {t} {g} -l {l} -t {e}'
                      ' --outdir {d} --update').format(scp=geocode_script,
                                                       v=inps.velFile,
                                                       c=inps.tempCohFile,
                                                       t=inps.timeseriesFile,
                                                       g=inps.geomFile,
                                                       l=inps.lookupFile,
                                                       e=inps.templateFile,
                                                       d=geo_dir)
            print(geoCmd)
            status = subprocess.Popen(geoCmd, shell=True).wait()
            if status is not 0:
                if inps.plot:
                    plot_pysarApp(inps)
                raise Exception('Error while geocoding.\n')
            else:
                inps.velFile        = os.path.join(geo_dir, 'geo_'+os.path.basename(inps.velFile))
                inps.tempCohFile    = os.path.join(geo_dir, 'geo_'+os.path.basename(inps.tempCohFile))
                inps.timeseriesFile = os.path.join(geo_dir, 'geo_'+os.path.basename(inps.timeseriesFile))
                inps.geomFile       = os.path.join(geo_dir, 'geo_'+os.path.basename(inps.geomFile))
                inps.geocoded = True

            # generate mask based on geocoded temporal coherence
            print('\n--------------------------------------------')
            outName = os.path.join(geo_dir, 'geo_maskTempCoh.h5')
            genCmd = 'generate_mask.py {} -m {} -o {}'.format(inps.tempCohFile,
                                                              inps.minTempCoh,
                                                              outName)
            print(genCmd)
            if ut.run_or_skip(out_file=outName, in_file=inps.tempCohFile) == 'run':
                status = subprocess.Popen(genCmd, shell=True).wait()
            inps.maskFile = outName

    # mask velocity file
    if inps.velFile and inps.maskFile:
        outName = '{}_masked.h5'.format(os.path.splitext(inps.velFile)[0])
        maskCmd = 'mask.py {} -m {} -o {}'.format(inps.velFile,
                                                  inps.maskFile,
                                                  outName)
        print(maskCmd)
        if ut.run_or_skip(out_file=outName, in_file=[inps.velFile, inps.maskFile]) == 'run':
            status = subprocess.Popen(maskCmd, shell=True).wait()
        try:
            inps.velFile = glob.glob(outName)[0]
        except:
            inps.velFile = None

    # Save to Google Earth KML file
    if inps.geocoded and inps.velFile and template['pysar.save.kml'] is True:
        print('\n--------------------------------------------')
        print('creating Google Earth KMZ file for geocoded velocity file: ...')
        outName = '{}.kmz'.format(os.path.splitext(os.path.basename(inps.velFile))[0])
        kmlCmd = 'save_kml.py {} -o {}'.format(inps.velFile, outName)
        print(kmlCmd)
        try:
            outFile = [i for i in [outName, 'PIC/{}'.format(outName)] if os.path.isfile(i)][0]
        except:
            outFile = None
        if ut.run_or_skip(out_file=outFile, in_file=inps.velFile, check_readable=False) == 'run':
            status = subprocess.Popen(kmlCmd, shell=True).wait()
            if status is not 0:
                if inps.plot:
                    plot_pysarApp(inps)
                raise Exception('Error while generating Google Earth KMZ file.')

    #############################################
    # Save Timeseries to HDF-EOS5 format
    #############################################
    if template['pysar.save.hdfEos5'] is True:
        print('\n**********  Save Time-series in HDF-EOS5 Format  **********')
        save_hdfeos5(inps, customTemplate)

    #############################################
    # Plot Figures
    #############################################
    if inps.plot:
        plot_pysarApp(inps)

    #############################################
    # Timing                                    #
    #############################################
    m, s = divmod(time.time()-start_time, 60)
    print('\n###############################################')
    print('End of PySAR Routine Processing Workflow!')
    print('###############################################\n')
    print('time used: {:02.0f} mins {:02.1f} secs'.format(m, s))


###########################################################################################
if __name__ == '__main__':
    main()

