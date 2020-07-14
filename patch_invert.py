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

    #inps.cluster = 'no'
    if inps.cluster == 'no':
        iut.inversion(**data_kwargs)
    else:
        # parallel
        print('\n\n------- start parallel processing using Dask -------')

        # initiate dask cluster and client
        cluster_obj = cluster_minopy.MDaskCluster(inps.cluster, inps.numWorker, config_name=inps.config)
        cluster_obj.open()

        cluster_obj.run(iut.inversion, func_data=data_kwargs)

        # close dask cluster and client
        cluster_obj.close()

        print('------- finished parallel processing -------\n\n')

    return


if __name__ == '__main__':
    main()

'''
args_file = patch_dir + '/data_kwargs.pkl'
with open(args_file, 'wb') as handle:
    pickle.dump(data_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

cmd = 'patch_invert.py --dataArg {a1} --cluster {a2} --num-worker {a3} --config-name {a4}\n'.format(
    a1=args_file, a2=self.cluster, a3=self.numWorker, a4=self.config)
run_commands.append(cmd)

if len(run_commands) > 0:
    run_dir = self.work_dir + '/run_file'
    os.makedirs(run_dir, exist_ok=True)
    run_file_inversion = os.path.join(run_dir, 'run_minopy_inversion')
    with open(run_file_inversion, 'w+') as f:
        f.writelines(run_commands)

inps_args = self.inps
inps_args.work_dir = run_dir
inps_args.out_dir = run_dir
job_obj = JOB_SUBMIT(inps_args)

putils.remove_last_job_running_products(run_file=run_file_inversion)
job_status = job_obj.submit_batch_jobs(batch_file=run_file_inversion)
if job_status:

    putils.remove_zero_size_or_length_error_files(run_file=run_file_inversion)
    putils.rerun_job_if_exit_code_140(run_file=run_file_inversion, inps_dict=inps_args)
    putils.raise_exception_if_job_exited(run_file=run_file_inversion)
    putils.concatenate_error_files(run_file=run_file_inversion, work_dir=inps_args.work_dir)
    putils.move_out_job_files_to_stdout(run_file=run_file_inversion)

'''