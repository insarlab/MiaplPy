#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import numpy as np
import minopy_utilities as mut
from skimage.measure import label
import h5py
import pickle
from mintpy.utils import ptime
from minopy.objects.arg_parser import MinoPyParser
from minopy.objects import cluster_minopy
import minopy.objects.inversion_utils as iut


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

    # inps.cluster = 'no'
    if inps.cluster == 'no':
        iut.parallel_invertion(**data_kwargs)
    else:
        # parallel
        print('\n\n------- start parallel processing using Dask -------')

        # initiate dask cluster and client
        cluster_obj = cluster_minopy.MDaskCluster(inps.cluster, inps.numWorker, config_name=inps.config)
        cluster_obj.open()

        cluster_obj.run(iut.parallel_invertion, func_data=data_kwargs)

        # close dask cluster and client
        cluster_obj.close()

        print('------- finished parallel processing -------\n\n')

    return


if __name__ == '__main__':
    main()

