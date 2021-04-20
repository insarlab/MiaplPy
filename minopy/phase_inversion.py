#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

#mpl_logger = logging.getLogger('matplotlib')
#mpl_logger.setLevel(logging.WARNING)

fiona_logger = logging.getLogger('fiona')
fiona_logger.propagate = False

import time
import numpy as np

from minopy.objects.arg_parser import MinoPyParser
#from mpi4py import MPI
from minopy.lib import utils as iut
from minopy.lib import invert as iv
from math import ceil
import multiprocessing as mp
from functools import partial
import signal

#################################

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main(iargs=None):
    '''
        Phase linking process.
    '''


    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    inversionObj = iv.CPhaseLink(inps)

    box_list = []
    for box in inversionObj.box_list:
        index = box[4]
        out_dir = inversionObj.out_dir.decode('UTF-8')
        out_folder = out_dir + '/PATCHES/PATCH_{}'.format(index)
        os.makedirs(out_folder, exist_ok=True)
        if not os.path.exists(out_folder + '/quality.npy'):
            box_list.append(box)

    num_workers = int(inps.num_worker)
    cpu_count = mp.cpu_count()
    if num_workers > cpu_count:
        print('Maximum number of Workers is {}\n'.format(cpu_count))
        num_cores = cpu_count
    else:
        num_cores = num_workers

    pool = mp.Pool(num_cores, init_worker)
    data_kwargs = inversionObj.get_datakwargs()
    os.makedirs(data_kwargs['out_dir'].decode('UTF-8') + '/PATCHES', exist_ok=True)

    func = partial(iut.process_patch_c, range_window=data_kwargs['range_window'],
                   azimuth_window=data_kwargs['azimuth_window'], width=data_kwargs['width'],
                   length=data_kwargs['length'], n_image=data_kwargs['n_image'],
                   slcStackObj=data_kwargs['slcStackObj'], distance_threshold=data_kwargs['distance_threshold'],
                   reference_row=data_kwargs['reference_row'],reference_col=data_kwargs['reference_col'],
                   phase_linking_method=data_kwargs['phase_linking_method'],
                   total_num_mini_stacks=data_kwargs['total_num_mini_stacks'],
                   default_mini_stack_size=data_kwargs['default_mini_stack_size'],
                   shp_test=data_kwargs['shp_test'],
                   def_sample_rows=data_kwargs['def_sample_rows'],
                   def_sample_cols=data_kwargs['def_sample_cols'],
                   out_dir=data_kwargs['out_dir'])

    try:
        pool.map(func, box_list)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()

    inversionObj.unpatch()

    return None


#################################################


if __name__ == '__main__':
    main()
