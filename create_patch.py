#!/usr/bin/env python3
# Author: Sara Mirzaee


import os
import sys
import shutil
import numpy as np
import dask
from datetime import datetime
import time
import minsar.job_submission as js
from minsar.objects import message_rsmas
import minopy_utilities as mnp
from minsar.utils.process_utilities import add_pause_to_walltime, get_config_defaults
from minsar.objects.auto_defaults import PathFind


pathObj = PathFind()
#######################


def main(iargs=None):
    """
        Divides the whole scene into patches for parallel processing
    """

    inps = mnp.cmd_line_parse(iargs)

    config = get_config_defaults(config_file='job_defaults.cfg')

    job_file_name = 'create_patch'
    job_name = job_file_name

    if inps.wall_time == 'None':
        inps.wall_time = config[job_file_name]['walltime']

    wait_seconds, new_wall_time = add_pause_to_walltime(inps.wall_time, inps.wait_time)

    #########################################
    # Submit job
    #########################################

    if inps.submit_flag:
        js.submit_script(job_name, job_file_name, sys.argv[:], inps.work_dir, new_wall_time)
        sys.exit(0)

    time.sleep(wait_seconds)

    message_rsmas.log(inps.work_dir, os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::]))

    inps.minopy_dir = os.path.join(inps.work_dir, pathObj.minopydir)
    pathObj.patch_dir = inps.minopy_dir + '/PATCH'

    pathObj.int_dir = os.path.join(inps.work_dir, pathObj.mergedintdir)

    if os.path.exists(pathObj.int_dir):
        shutil.rmtree(pathObj.int_dir)
        os.mkdir(pathObj.int_dir)
    else:
        os.mkdir(pathObj.int_dir)

    pathObj.slave_dir = os.path.join(inps.work_dir, pathObj.mergedslcdir)

    pathObj.list_slv = os.listdir(pathObj.slave_dir)
    pathObj.list_slv = [datetime.strptime(x, '%Y%m%d') for x in pathObj.list_slv]
    pathObj.list_slv = np.sort(pathObj.list_slv)
    pathObj.list_slv = [x.strftime('%Y%m%d') for x in pathObj.list_slv]

    inps.range_window = int(inps.template['minopy.range_window'])
    inps.azimuth_window = int(inps.template['minopy.azimuth_window'])

    if not os.path.isdir(inps.minopy_dir):
        os.mkdir(inps.minopy_dir)

    slc = mnp.read_image(pathObj.slave_dir + '/' + pathObj.list_slv[0] + '/' + pathObj.list_slv[0] + '.slc')  #
    pathObj.n_image = len(pathObj.list_slv)
    pathObj.lin = slc.shape[0]
    pathObj.sam = slc.shape[1]
    del slc

    pathObj.patch_rows, pathObj.patch_cols, inps.patch_list = \
        mnp.patch_slice(pathObj.lin, pathObj.sam, inps.azimuth_window, inps.range_window, np.int(inps.template['minopy.patch_size']))

    np.save(inps.minopy_dir + '/rowpatch.npy', pathObj.patch_rows)
    np.save(inps.minopy_dir + '/colpatch.npy', pathObj.patch_cols)

    submit_dask_job(inps.patch_list, inps.minopy_dir)

    return


def create_patch(name):
    patch_row, patch_col = name.split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))
    patch_name = pathObj.patch_dir + str(patch_row) + '_' + str(patch_col)

    line = pathObj.patch_rows[1][0][patch_row] - pathObj.patch_rows[0][0][patch_row]
    sample = pathObj.patch_cols[1][0][patch_col] - pathObj.patch_cols[0][0][patch_col]

    if not os.path.isfile(patch_name + '/count.npy'):
        if not os.path.isdir(patch_name):
            os.mkdir(patch_name)

        rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='w+', shape=(pathObj.n_image, line, sample))

        count = 0

        for dirs in pathObj.list_slv:
            data_name = pathObj.slave_dir + '/' + dirs + '/' + dirs + '.slc'
            slc = np.memmap(data_name, dtype=np.complex64, mode='r', shape=(pathObj.lin, pathObj.sam))

            rslc[count, :, :] = slc[pathObj.patch_rows[0][0][patch_row]:pathObj.patch_rows[1][0][patch_row],
                                pathObj.patch_cols[0][0][patch_col]:pathObj.patch_cols[1][0][patch_col]]
            count += 1
            del slc

        del rslc

        np.save(patch_name + '/count.npy', [pathObj.n_image, line, sample])

    else:
        print('Next patch...')
    return print("PATCH" + str(patch_row) + '_' + str(patch_col) + " is created")


def submit_dask_job(patch_list, minopy_dir):

    if os.path.isfile(minopy_dir + '/flag.npy'):
        print('patchlist exist')
    else:

        futures = []
        start_time = time.time()

        for patch in patch_list:
            futures.append(dask.delayed(create_patch)(patch))

        results = dask.compute(*futures)

    np.save(minopy_dir + '/flag.npy', 'patchlist_created')
    timep = time.time() - start_time

    print('All patches created in {} seconds'.format(timep))

    return


if __name__ == '__main__':
    '''
    Divides the whole scene into patches for parallel processing.

    '''
    main()
