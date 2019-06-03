#!/usr/bin/env python3
######################################################################
# Project: part of MintPy                                             #
# Purpose: InSAR Time Series Analysis in Python using phase linking  #                                             #
# Author: Sara Mirzaee                                               #
# Copyright (c) 2018, Sara Mirzaee, Yunjun Zhang, Heresh Fattahi     #
######################################################################

import os
import glob
import time
import argparse
import numpy as np
import gdal
import mintpy
from mintpy import smallbaselineApp
from minsar.utils.process_utilities import get_project_name, get_work_directory
from minsar.objects.auto_defaults import PathFind
from mintpy.utils import readfile, utils as ut

pathObj = PathFind()

##########################################################################

EXAMPLE = """example:
  timeseries_corrections.py                                             #Run / Rerun
  
  # Template options
  timeseries_corrections.py -H                               #Print    default template
  timeseries_corrections.py -g                               #Generate default template
  timeseries_corrections.py -g SanAndreasT356EnvD.template   #Generate default template considering input custom template
"""


def create_parser():
    parser = argparse.ArgumentParser(description='MintPy Phase Linking and Time Series Analysis',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings\n')
    parser.add_argument('--dir', dest='work_dir',
                        help='MintPy working directory, default is:\n' +
                             'a) current directory, or\n' +
                             'b) $SCRATCHDIR/projectName/mintpy, if meets the following 3 requirements:\n' +
                             '    1) autoPath = True in mintpy/defaults/auto_path.py\n' +
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
    inps.autoTemplateFile = os.path.join(os.getenv('MINTPY_HOME'), 'defaults/smallbaselineApp_auto.cfg')

    if inps.print_auto_template:
        with open(inps.autoTemplateFile, 'r') as f:
            print(f.read())
        raise SystemExit()

    if (inps.customTemplateFile
            and os.path.basename(inps.customTemplateFile) == 'smallbaselineApp_auto.cfg'):
        inps.customTemplateFile = None
    else:
        inps.customTemplateFile = os.path.abspath(inps.customTemplateFile)

    inps.project_name = get_project_name(inps.customTemplateFile)
    inps.work_dir = os.path.join(get_work_directory(None, inps.project_name), pathObj.mintpydir)

    return inps

###############################


def get_phase_linking_coherence_mask(inps, template):
    """Generate reliable pixel mask from temporal coherence"""

    geom_file = ut.check_loaded_dataset(inps.work_dir, print_msg=False)[2]
    tcoh_file = os.path.join(inps.work_dir, 'temporalCoherence.h5')
    mask_file = os.path.join(inps.work_dir, 'maskTempCoh.h5')
    tcoh_min = 0.4

    scp_args = '{} -m {} -o {} --shadow {}'.format(tcoh_file, tcoh_min, mask_file, geom_file)
    print('generate_mask.py', scp_args)

    # update mode: run only if:
    # 1) output file exists and newer than input file, AND
    # 2) all config keys are the same

    print('update mode: ON')
    flag = 'skip'
    if ut.run_or_skip(out_file=mask_file, in_file=tcoh_file, print_msg=False) == 'run':
        flag = 'run'

    print('run or skip: {}'.format(flag))

    if flag == 'run':
        mintpy.generate_mask.main(scp_args.split())
        # update configKeys
        atr = {}
        atr['mintpy.networkInversion.minTempCoh'] = tcoh_min
        ut.add_attribute(mask_file, atr)

    # check number of pixels selected in mask file for following analysis
    num_pixel = np.sum(readfile.read(mask_file)[0] != 0.)
    print('number of reliable pixels: {}'.format(num_pixel))

    min_num_pixel = float(template['mintpy.networkInversion.minNumPixel'])
    if num_pixel < min_num_pixel:
        msg = "Not enough reliable pixels (minimum of {}). ".format(int(min_num_pixel))
        msg += "Try the following:\n"
        msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
        msg += "2) Check the network and make sure it's fully connected without subsets"
        raise RuntimeError(msg)
    return

###############################


def write_to_timeseries(inps, template):

    from mintpy.objects import ifgramStack, timeseries
    from mintpy.ifgram_inversion import write2hdf5_file, read_unwrap_phase, mask_unwrap_phase

    inps.timeseriesFile = os.path.join(inps.work_dir, 'timeseries.h5')
    inps.tempCohFile = os.path.join(inps.work_dir, 'temporalCoherence.h5')
    inps.timeseriesFiles = [os.path.join(inps.work_dir, 'timeseries.h5')]       #all ts files
    inps.outfile = [os.path.join(inps.work_dir, 'timeseries.h5'),
                    os.path.join(inps.work_dir, 'temporalCoherence.h5')]

    ifgram_file = os.path.join(inps.work_dir, 'inputs/ifgramStack.h5')
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
    mask_dataset_name = None
    mask_threshold = 0.4

    pha_data = read_unwrap_phase(stack_obj,
                                 box,
                                 ref_phase,
                                 unwDatasetName=unwDatasetName,
                                 dropIfgram=True)

    pha_data = mask_unwrap_phase(pha_data,
                                 stack_obj,
                                 box,
                                 dropIfgram=True,
                                 mask_ds_name=mask_dataset_name,
                                 mask_threshold=mask_threshold)

    ts = pha_data * phase2range
    ts0 = ts.reshape(num_date-1, num_row, num_col)
    ts = np.zeros((num_date, num_row, num_col), np.float32)
    ts[1::, :, :] = ts0
    num_inv_ifg = np.zeros((num_row, num_col), np.int16) + num_date - 1

    gfilename = os.path.join(os.getenv('SCRATCHDIR'), inps.project_name, 'merged/geom_master/Quality.rdr')
    ds = gdal.Open(gfilename, gdal.GA_ReadOnly)
    quality_map = ds.GetRasterBand(1).ReadAsArray()
    inps.plmethod = ds.GetMetadata()['plmethod']

    if 'SUBSET_XMIN' in metadata:
        first_row = int(metadata['SUBSET_YMIN'])
        last_row = int(metadata['SUBSET_YMAX'])
        first_col = int(metadata['SUBSET_XMIN'])
        last_col = int(metadata['SUBSET_XMAX'])

        quality_map = quality_map[first_row:last_row, first_col:last_col]

    write2hdf5_file(ifgram_file, metadata, ts, quality_map, ts_std, num_inv_ifg, suffix='', inps=inps)

    get_phase_linking_coherence_mask(inps, template)

    return

##########################


def main(iargs=None):
    start_time = time.time()
    inps = cmd_line_parse(iargs)
    app = smallbaselineApp.TimeSeriesAnalysis(inps.customTemplateFile, inps.work_dir)
    app.startup()

    if app.template['mintpy.unwrapError.method']:
        app.template['mintpy.unwrapError.method'] = 'bridging'

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

    write_to_timeseries(inps, app.template)

    os.chdir(inps.work_dir)

    app.run(steps=inps.runSteps[5::])

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('\nTotal time: {:02.0f} mins {:02.1f} secs'.format(m, s))
    return


###########################################################################################
if __name__ == '__main__':
    main()