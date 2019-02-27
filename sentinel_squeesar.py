#!/usr/bin/env python3

# Author: Sara Mirzaee


import numpy as np
import argparse
import os
import subprocess
import sys
import time
import argparse
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from rsmas_logging import loglevel
from dataset_template import Template
import _pysqsar_utilities as pysq
#from dask import compute, delayed

logger_ph_lnk  = pysq.send_logger_squeesar()

######################################################

EXAMPLE = """example:
  sentinel_squeesar.py LombokSenAT156VV.template 
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
        help='custom template with option settings.\n')
    
    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps

    
def create_patch(inps, name):

    patch_row, patch_col = name.split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))
    patch_name = inps.patch_dir + str(patch_row) + '_' + str(patch_col)

    line = inps.patch_rows[1][0][patch_row] - inps.patch_rows[0][0][patch_row]
    sample = inps.patch_cols[1][0][patch_col] - inps.patch_cols[0][0][patch_col]

    if  not os.path.isfile(patch_name + '/count.npy'):
        if not os.path.isdir(patch_name):
            os.mkdir(patch_name)
        logger_ph_lnk.log(loglevel.INFO, "Making PATCH" + str(patch_row) + '_' + str(patch_col))

        rslc = np.memmap(patch_name + '/RSLC', dtype=np.complex64, mode='w+', shape=(inps.n_image, line, sample))

        count = 0
        for dirs in inps.list_slv:
            data_name = inps.slave_dir + '/' + dirs + '/' + dirs + '.slc.full'
            slc = np.memmap(data_name, dtype=np.complex64, mode='r', shape=(inps.lin, inps.sam))
            
            rslc[count, :, :] = slc[inps.patch_rows[0][0][patch_row]:inps.patch_rows[1][0][patch_row],
                                            inps.patch_cols[0][0][patch_col]:inps.patch_cols[1][0][patch_col]]
            count += 1
            del slc
        del rslc

        np.save(patch_name + '/count.npy', inps.n_image)
    else:
        print('Next patch...')
    return "PATCH" + str(patch_row) + '_' + str(patch_col)+" is created"
                                            
######################################################################

def main(iargs=None):
    """
        Pre-process and wrapper for phase linking and phase filtering of Distributed scatterers
    """

    inps = command_line_parse(iargs)
    logger_ph_lnk.log(loglevel.INFO, os.path.basename(sys.argv[0]) + " " + sys.argv[1])
    inps.project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    inps.project_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name

    inps.slave_dir = inps.project_dir + '/merged/SLC'
    inps.sq_dir = inps.project_dir + '/SqueeSAR'
    inps.patch_dir = inps.sq_dir+'/PATCH'
    inps.list_slv = os.listdir(inps.slave_dir)
    #A = [inps.list_slv[0]+'_'+x for x in inps.list_slv]
    #with open('A.txt','w') as f:
    #    for t in A[1::]:
    #        f.write(t+'\n')
    

    
    inps.range_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizerange'])
    inps.azimuth_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizeazimuth'])
  
    if not os.path.isdir(inps.sq_dir):
        os.mkdir(inps.sq_dir)
    
    
    try:
      jobqueue = Template(inps.custom_template_file).get_options()['job_queue']
    except:
      jobqueue = 'general'
    

    slc = pysq.read_image(inps.slave_dir + '/' + inps.list_slv[0] + '/' + inps.list_slv[0] + '.slc.full')  #
    inps.n_image = len(inps.list_slv)
    inps.lin = slc.shape[0]
    inps.sam = slc.shape[1]
    del slc
    
    inps.patch_rows, inps.patch_cols, inps.patch_list = \
        pysq.patch_slice(inps.lin,inps.sam,inps.azimuth_win,inps.range_win)
                                            
    np.save(inps.sq_dir+'/rowpatch.npy',inps.patch_rows)
    np.save(inps.sq_dir+'/colpatch.npy',inps.patch_cols)
    
    time0 = time.time()                                        
    if os.path.isfile(inps.sq_dir + '/flag.npy'):
        print('patchlist exist')
    else:
        for patch in inps.patch_list:
            create_patch(inps,patch)    

        #values = [delayed(create_patch)(inps, x) for x in inps.patch_list]
        #compute(*values, scheduler='processes')
    np.save(inps.sq_dir + '/flag.npy', 'patchlist_created')
    timep = time.time() - time0
    logger_ph_lnk.log(loglevel.INFO, "Done Creating PATCH. time:{}".format(timep))

###########################################      
 
    flag = np.load(inps.sq_dir + '/flag.npy')
    if flag == 'patchlist_created':
        run_PSQ_sentinel = inps.sq_dir + "/run_PSQ_sentinel"
        with open(run_PSQ_sentinel, 'w') as f:
            for patch in inps.patch_list:
                if not os.path.isfile(inps.sq_dir+'/'+patch+'/num_processed.npy'):
                    cmd = 'PSQ_sentinel.py ' + inps.custom_template_file + ' -p ' + 'PATCH' + patch + ' \n'
                    f.write(cmd)
        ## cmd = 'createBatch.pl ' + inps.sq_dir + '/run_PSQ_sentinel' + ' memory=' + '3700' + ' walltime=' + '10:00'

        cmd = 'submit_jobs.py -f ' + inps.sq_dir + '/run_PSQ_sentinel -w 3:00 -r 3000 -q ' + jobqueue
        status = subprocess.Popen(cmd, shell=True).wait()
        if status is not 0:
            #logger_ph_lnk.log(loglevel.ERROR, 'ERROR running PSQ_sentinel.py')
            raise Exception('ERROR running PSQ_sentinel.py')


 ###########################################   


    run_write_slc = inps.project_dir + '/merged/run_write_SLC'

    with open(run_write_slc, 'w') as f:
        for date in inps.list_slv:
            cmd = 'writeSQ_sentinel.py ' + inps.custom_template_file + ' -s ' + date + '/' + date + '.slc.full' + ' \n'
            f.write(cmd)

    print ("job file created: " + " run_write_SLC")

    #cmd = '$INT_SCR/split_jobs.py -f ' + inps.project_dir + '/merged/run_write_SLC -w 1:00 -r 5000 -q '+ jobqueue 
    cmd = 'createBatch.pl ' + inps.project_dir + '/merged/run_write_SLC' + ' memory=' + '5000' + ' walltime=' + '1:00' + ' QUEUENAME=bigmem'
    status = subprocess.Popen(cmd, shell=True).wait()
    if status is not 0:
        logger_ph_lnk.log(loglevel.ERROR, 'ERROR writing SLCs')
        raise Exception('ERROR writing SLCs')


if __name__ == '__main__':
    '''
    Creates patches from the data and calls phase linking to process. 
    
    Process for each patch is done sequentially.
    '''
    main()
