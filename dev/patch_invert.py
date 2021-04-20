#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import logging
import warnings

warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

mpl_logger = logging.getLogger('asyncio')
mpl_logger.setLevel(logging.WARNING)

import os
import numpy as np
import pickle
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects import cluster_minopy
import minopy.objects.inversion_utils as iut
from osgeo import gdal
import time


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='patch_invert')
    inps = Parser.parse()

    # --cluster and --num-worker option
    inps.numWorker = str(cluster_minopy.cluster.DaskCluster.format_num_worker(inps.cluster, inps.numWorker))
    if inps.cluster != 'no' and inps.numWorker == '1':
       print('WARNING: number of workers is 1, turn OFF parallel processing and continue')
       inps.cluster = 'no'

    with open(inps.data_kwargs, 'rb') as handle:
        data_kwargs = pickle.load(handle)

    patch_dir = data_kwargs['patch_dir']
    quality_file = os.path.join(patch_dir, 'quality')
    ds = gdal.Open(quality_file + '.vrt', gdal.GA_ReadOnly)
    quality = ds.GetRasterBand(1).ReadAsArray()
    quality[:, :] = -1

    if not -1 in quality:
        np.save(patch_dir + '/flag.npy', '{} is done inverting'.format(os.path.basename(patch_dir)))
        return

    #inps.cluster = 'no'
    if inps.cluster == 'no':
        t0 = time.time()
        iut.parallel_invertion(**data_kwargs)
        print('time spent: {} s'.format(time.time() - t0))
    else:
        # parallel
        print('\n\n------- start processing {} using Dask -------'.format(os.path.basename(patch_dir)))

        # initiate dask cluster and client
        cluster_obj = cluster_minopy.MDaskCluster(inps.cluster, inps.numWorker, config_name=inps.config)
        cluster_obj.open()

        cluster_obj.run(iut.parallel_invertion, func_data=data_kwargs)

        # close dask cluster and client
        cluster_obj.close()

        print('------- finished processing {} -------\n\n'.format(os.path.basename(patch_dir)))
        
    np.save(patch_dir + '/flag.npy', '{} is done inverting'.format(os.path.basename(patch_dir)))

    return


if __name__ == '__main__':
    main()

