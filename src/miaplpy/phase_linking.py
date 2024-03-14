#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import os
import sys
import logging
import warnings
import datetime

warnings.filterwarnings("ignore")

fiona_logger = logging.getLogger('fiona')
fiona_logger.propagate = False

from miaplpy.objects.arg_parser import MiaplPyParser
from miaplpy.lib import utils as iut
from miaplpy.lib import invert as iv
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

    Parser = MiaplPyParser(iargs, script='phase_linking')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        #sysargv = ['./inputs/slcStack.h5' if x == '/tmp/slcStack.h5' else x for x in sys.argv[1::]]
        sysargv = [x for x in sys.argv[1::]]
        msg = os.path.basename(__file__) + ' ' + ' '.join(sysargv)
        string = dateStr + " * " + msg
        print(string)

    inversionObj = iv.CPhaseLink(inps)

    if inps.do_concatenate:
        phase_invert(inps, inversionObj)
    else:
        concatenate_patches(inversionObj)

    return None


def phase_invert(inps, inversionObj):

    if not inps.sub_index is None:
        inps.sub_index = int(inps.sub_index)
        indx1 = int(inps.sub_index * inps.num_worker)
        indx2 = int((inps.sub_index + 1) * inps.num_worker) #+ 1
        if indx2 > len(inversionObj.box_list):
            indx2 = len(inversionObj.box_list)
        print('Total number of PATCHES/tasks for job {} : {}'.format(inps.sub_index, indx2-indx1))
    else:
        indx1 = 0
        indx2 = len(inversionObj.box_list)
        print('Total number of PATCHES/tasks: {}'.format(len(inversionObj.box_list)))

    box_list = []
    for box in inversionObj.box_list[indx1:indx2]:
        index = box[4]
        out_dir = inversionObj.out_dir.decode('UTF-8')
        out_folder = out_dir + '/PATCHES/PATCH_{:04.0f}'.format(index)
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.exists(out_folder + '/flag.npy'):
            box_list.append(box)

    #print('Total number of PATCHES: {}'.format(len(inversionObj.box_list)))
    print('Remaining number of PATCHES/tasks: {}'.format(len(box_list)))

    num_workers = int(inps.num_worker)
    cpu_count = mp.cpu_count()
    if num_workers > cpu_count:
        print('Maximum number of Workers is {}\n'.format(cpu_count))
        num_cores = cpu_count
    elif num_workers < len(box_list) < cpu_count:
        num_cores = len(box_list)
    else:
        num_cores = num_workers

    print('Number of parallel tasks: {}'.format(num_cores))
    pool = mp.Pool(num_cores, init_worker)
    data_kwargs = inversionObj.get_datakwargs()
    os.makedirs(data_kwargs['out_dir'].decode('UTF-8') + '/PATCHES', exist_ok=True)

    if int(data_kwargs['n_image']) < 10 and 'sequential' in data_kwargs['phase_linking_method'].decode('UTF-8'):
        new_plmethod = data_kwargs['phase_linking_method'].decode('UTF-8').split('sequential_')[1].encode('UTF-8')
        print('Number of images less than 10, phase linking method switched to "{}"'.format(new_plmethod))
        data_kwargs['phase_linking_method'] = new_plmethod

    func = partial(iut.process_patch_c, range_window=data_kwargs['range_window'],
                   azimuth_window=data_kwargs['azimuth_window'], width=data_kwargs['width'],
                   length=data_kwargs['length'], n_image=data_kwargs['n_image'],
                   slcStackObj=data_kwargs['slcStackObj'], distance_threshold=data_kwargs['distance_threshold'],
                   reference_row=data_kwargs['reference_row'], reference_col=data_kwargs['reference_col'],
                   phase_linking_method=data_kwargs['phase_linking_method'],
                   total_num_mini_stacks=data_kwargs['total_num_mini_stacks'],
                   default_mini_stack_size=data_kwargs['default_mini_stack_size'],
                   ps_shp=data_kwargs['ps_shp'],
                   shp_test=data_kwargs['shp_test'],
                   def_sample_rows=data_kwargs['def_sample_rows'],
                   def_sample_cols=data_kwargs['def_sample_cols'],
                   out_dir=data_kwargs['out_dir'],
                   lag=data_kwargs['time_lag'],
                   mask_file=data_kwargs['mask_file'])

    print('Reading SLC data from {} and inverting patches in parallel ...'.format(inps.slc_stack))

    try:
        pool.map(func, box_list)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()

    return

def concatenate_patches(inversionObj):
    completed = True
    for box in inversionObj.box_list:
        index = box[4]
        out_dir = inversionObj.out_dir.decode('UTF-8')
        out_folder = out_dir + '/PATCHES/PATCH_{:04.0f}'.format(index)
        while not os.path.exists(out_folder + '/flag.npy'):
            completed = False
            # print('Error: PATCH_{:04.0f} is not inverted, run previous step (phase_linking) to complete'.format(index))
            raise RuntimeError('Error: PATCH_{:04.0f} is not inverted, run previous step (phase_linking) to complete'.format(index))

    if completed:
        inversionObj.unpatch()
        print('Successfully concatenated')
    else:
        print('Exit without concatenating')
        sys.exit(0)
    return


#################################################


if __name__ == '__main__':
    main()
