#!/usr/bin/env python3
######################################################################
# Project: part of PySAR                                             #
# Purpose: InSAR Time Series Analysis in Python using phase linking  #                                             #
# Author: Sara Mirzaee                                               #
# Copyright (c) 2018, Sara Mirzaee, Yunjun Zhang, Heresh Fattahi     #
######################################################################

import os
import glob
import time
import argparse
import numpy as np
from pysar import pysarApp


##########################################################################

EXAMPLE = """example:
  timeseries_corrections.py                                             #Run / Rerun
  
  # Template options
  timeseries_corrections.py -H                               #Print    default template
  timeseries_corrections.py -g                               #Generate default template
  timeseries_corrections.py -g SanAndreasT356EnvD.template   #Generate default template considering input custom template
"""


def create_parser():
    parser = argparse.ArgumentParser(description='PySAR Phase Linking and Time Series Analysis',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings\n')
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


    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    inps.autoTemplateFile = os.path.join(os.getenv('PYSAR_HOME'), 'docs/pysarApp_template.txt')

    if inps.print_auto_template:
        with open(inps.autoTemplateFile, 'r') as f:
            print(f.read())
        raise SystemExit()

    if (inps.customTemplateFile
            and os.path.basename(inps.customTemplateFile) == 'pysarApp_template.txt'):
        inps.customTemplateFile = None
    return inps

###############################

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

###############################

def write_to_timeseries(inps):

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

    return



##########################

def main(iargs=None):
    start_time = time.time()
    inps = cmd_line_parse(iargs)

    app = pysarApp.TimeSeriesAnalysis(inps.customTemplateFile, inps.workDir)
    app.startup()

    if app.template['pysar.unwrapError.method']:
        app.template['pysar.unwrapError.method'] = 'bridging'

    inps.runSteps = ['load_data',
    'reference_point',
    'stack_interferograms',
    'correct_unwrap_error',
    'correct_troposphere',
    'deramp',
    'correct_topography',
    'residual_RMS',
    'reference_date',
    'velocity',
    'geocode',
    'google_earth',
    'hdfeos5']

    app.run(steps=inps.runSteps[0:4])

    write_to_timeseries(inps)

    app.run(steps=inps.runSteps[5::])

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('\nTotal time: {:02.0f} mins {:02.1f} secs'.format(m, s))
    return


###########################################################################################
if __name__ == '__main__':
    main()