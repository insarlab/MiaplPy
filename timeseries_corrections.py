#!/usr/bin/env python3
######################################################################
# Project: part of MintPy                                             #
# Purpose: InSAR Time Series Analysis in Python using phase linking  #                                             #
# Author: Sara Mirzaee                                               #
# Copyright (c) 2018, Sara Mirzaee, Yunjun Zhang, Heresh Fattahi     #
######################################################################

import os
import sys
import time
import numpy as np
import gdal
import mintpy
import mintpy.workflow
from mintpy.smallbaselineApp import TimeSeriesAnalysis
from mintpy.utils import readfile, utils as ut
import minsar.job_submission as js
from minsar.utils.process_utilities import add_pause_to_walltime, get_config_defaults
from minsar.objects.auto_defaults import PathFind
from minsar.objects import message_rsmas
import minopy.minopy_utilities as mnp

pathObj = PathFind()
##########################################################################


def main(iargs=None):

    start_time = time.time()
    inps = mnp.cmd_line_parse(iargs, script='timeseries_corrections')

    config = get_config_defaults(config_file='job_defaults.cfg')

    job_file_name = 'timeseries_corrections'
    job_name = job_file_name

    if inps.wall_time == 'None':
        inps.wall_time = config[job_file_name]['walltime']

    wait_seconds, new_wall_time = add_pause_to_walltime(inps.wall_time, inps.wait_time)

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:

        work_dir = os.getcwd()

        js.submit_script(job_name, job_file_name, sys.argv[:], work_dir, new_wall_time)
        sys.exit(0)

    time.sleep(wait_seconds)

    message_rsmas.log(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    inps.autoTemplateFile = os.path.join(os.getenv('MINTPY_HOME'), 'defaults/smallbaselineApp_auto.cfg')

    if inps.print_auto_template:
        with open(inps.autoTemplateFile, 'r') as f:
            print(f.read())
        raise SystemExit()

    if (inps.custom_template_file
            and os.path.basename(inps.custom_template_file) == 'smallbaselineApp_auto.cfg'):
        inps.custom_template_file = None
    else:
        inps.custom_template_file = os.path.abspath(inps.custom_template_file)

    inps.mintpy_dir = os.path.join(inps.work_dir, pathObj.mintpydir)

    app = TimeSeriesAnalysis(inps.custom_template_file, inps.mintpy_dir)
    app.startup()

    if app.template['mintpy.unwrapError.method']:
        app.template['mintpy.unwrapError.method'] = 'bridging'

    inps.runSteps = pathObj.minopy_corrections()

    app.run(steps=inps.runSteps[0:5])

    write_to_timeseries(inps, app.template)

    os.chdir(inps.mintpy_dir)

    app.run(steps=inps.runSteps[5::])

    # Timing
    m, s = divmod(time.time()-start_time, 60)
    print('\nTotal time: {:02.0f} mins {:02.1f} secs'.format(m, s))
    return

###########################################################################################


def get_phase_linking_coherence_mask(inps, template):
    """Generate reliable pixel mask from temporal coherence"""

    tcoh_file = os.path.join(inps.mintpy_dir, 'temporalCoherence.h5')
    mask_file = os.path.join(inps.mintpy_dir, 'maskTempCoh.h5')

    tcoh_min = 0.25

    scp_args = '{} --nonzero -o {} --update'.format(tcoh_file, mask_file)
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

    inps.timeseriesFile = os.path.join(inps.mintpy_dir, 'timeseries.h5')
    inps.tempCohFile = os.path.join(inps.mintpy_dir, 'temporalCoherence.h5')
    inps.timeseriesFiles = [os.path.join(inps.mintpy_dir, 'timeseries.h5')]       #all ts files
    inps.outfile = [os.path.join(inps.mintpy_dir, 'timeseries.h5'),
                    os.path.join(inps.mintpy_dir, 'temporalCoherence.h5')]

    ifgram_file = os.path.join(inps.mintpy_dir, 'inputs/ifgramStack.h5')

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

    phase2range = -1 * float(stack_obj.metadata['WAVELENGTH']) / (4. * np.pi)

    box = None
    ref_phase = stack_obj.get_reference_phase(dropIfgram=False)
    unwDatasetName = 'unwrapPhase'
    mask_dataset_name = None
    mask_threshold = 0.25

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

    if 'SUBSET_XMIN' in metadata:
        first_row = int(metadata['SUBSET_YMIN'])
        last_row = int(metadata['SUBSET_YMAX'])
        first_col = int(metadata['SUBSET_XMIN'])
        last_col = int(metadata['SUBSET_XMAX'])

        quality_map = quality_map[first_row:last_row, first_col:last_col]

    os.chdir(inps.mintpy_dir)

    write2hdf5_file(ifgram_file, metadata, ts, quality_map, num_inv_ifg, suffix='', inps=inps)

    get_phase_linking_coherence_mask(inps, template)

    return

##########################


if __name__ == '__main__':
    main()
