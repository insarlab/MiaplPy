#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
import numpy as np
import minopy_utilities as mnp
import dask
import glob
import minsar.utils.process_utilities as putils
from minsar.objects.auto_defaults import PathFind
import minsar.job_submission as js

pathObj = PathFind()


#################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given bounding box in lat/lon')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('customTemplateFile', nargs='?', help='custom template with option settings.\n')

    return parser


def command_line_parse(iargs=None):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(iargs)

    return inps


if __name__ == '__main__':
    '''
    Phase linking process.
    '''

    inps = command_line_parse()
    inps = putils.create_or_update_template(inps)
    pathObj.minopydir = os.path.join(inps.work_dir, pathObj.minopydir)
    patch_list = glob.glob(pathObj.minopydir + '/PATCH*')

    run_minopy_inversion = os.path.join(inps.work_dir, pathObj.rundir, 'run_minopy_inversion')

    with open(run_minopy_inversion, 'w') as f:
        for item in patch_list:
            cmd = 'patch_inversion.py {a0} -p {a1} \n'.format(a0=inps.customTemplateFile, a1=item)
            f.write(cmd)

    config = putils.get_config_defaults(config_file='job_defaults.cfg')

    step_name = 'phase_linking'
    try:
        memorymax = config[step_name]['memory']
    except:
        memorymax = config['DEFAULT']['memory']

    try:
        if config[step_name]['adjust'] == 'True':
            walltimelimit = putils.walltime_adjust(config[step_name]['walltime'])
        else:
            walltimelimit = config[step_name]['walltime']
    except:
        walltimelimit = config['DEFAULT']['walltime']

    queuename = os.getenv('QUEUENAME')

    putils.remove_last_job_running_products(run_file=run_minopy_inversion)

    jobs = js.submit_batch_jobs(batch_file=run_minopy_inversion,
                                out_dir=os.path.join(inps.work_dir, 'run_files'),
                                memory=memorymax, walltime=walltimelimit, queue=queuename)

    putils.remove_zero_size_or_length_error_files(run_file=run_minopy_inversion)
    putils.raise_exception_if_job_exited(run_file=run_minopy_inversion)
    putils.concatenate_error_files(run_file=run_minopy_inversion)
    putils.move_out_job_files_to_stdout(run_file=run_minopy_inversion)


#################################################
