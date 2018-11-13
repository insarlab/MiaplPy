#!/usr/bin/env python3

# Author: Sara Mirzaee


import isce
import isceobj
import numpy as np
import argparse
import os
import copy

import gdal
import subprocess
import sys
import glob
import argparse
from rsmas_logging import rsmas_logger, loglevel
import pysar
from pysar.utils import utils
from pysar.utils import readfile
import _pysqsar_utilities as psq
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
import _process_utilities as putils
from dask import compute, delayed
import dask.multiprocessing

################
EXAMPLE = """example:
  sentinel_squeesar.py LombokSenAT156VV.template 
"""

inps = None

logfile_name = os.getenv('SCRATCHDIR') + '/LOGS/pysqsar_rsmas.log'
logger = rsmas_logger(file_name=logfile_name)


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
        help='custom template with option settings.\n')
    
    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    global inps

    parser = create_parser()
    inps = parser.parse_args(args)

    
def createpatch(name):
    n1, n2 = name.split('_')
    n1, n2 = int(n1), int(n2)
    patn = inps.patch_dir + str(n1) + '_' + str(n2)
    lin1 = inps.pr[1][0][n1] - inps.pr[0][0][n1]
    sam1 = inps.pc[1][0][n1] - inps.pc[0][0][n1]
    if not os.path.isdir(patn) or not os.path.isfile(patn + '/count.npy'):
        os.mkdir(patn)
        logger.info("Making PATCH" + str(n1) + '_' + str(n2))
        amp1 = np.empty((inps.nimage, lin1, sam1))
        ph1 = np.empty((inps.nimage, lin1, sam1))  
        count = 0
        for dirs in inps.listslv:
            dname = inps.slave_dir + '/' + dirs + '/' + dirs + '.slc.full'
            slc = np.memmap(dname, dtype=np.complex64, mode='r', shape=(inps.lin, inps.sam))
            amp1[count, :, :] = np.abs(slc[inps.pr[0][0][n1]:inps.pr[1][0][n1], 
                                       inps.pc[0][0][n2]:inps.pc[1][0][n2])
            ph1[count, :, :] = np.angle(slc[inps.pr[0][0][n1]:inps.pr[1][0][n1], 
                                       inps.pc[0][0][n2]:inps.pc[1][0][n2])
            count += 1
            del slc
        np.save(patn + '/' + 'Amplitude.npy', amp1)
        np.save(patn + '/' + 'Phase.npy', ph1)
        np.save(patn + '/count.npy', inps.nimage)
    else:
        print('Next patch...')                              
    return "PATCH" + str(n1) + '_' + str(n2)+" is created"  
                                            
######################################################################

if __name__ == "__main__":
                                            
    command_line_parse(sys.argv[1:])
    inps.project_name = putils.get_project_name(custom_template_file=inps.custom_template_file)
    inps.work_dir = putils.get_work_directory(None, inps.project_name)
    templateContents = readfile.read_template(inps.custom_template_file)
    inps.proj_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name
    inps.slave_dir = inps.work_dir + '/merged/SLC'
    inps.sqdir = inps.work_dir + '/SqueeSAR'
    inps.patch_dir = sqdir+'/PATCH'
    inps.listslv = os.listdir(slave_dir)  
    
    inps.wra = int(templateContents['squeesar.wsizerange'])
    inps.waz = int(templateContents['squeesar.wsizeazimuth'])
    
    if not os.path.isdir(inps.sqdir):
        os.mkdir(inps.sqdir)

    slc = psq.readim(inps.slave_dir + '/' + inps.listslv[0] + '/' + inps.listslv[0] + '.slc.full')  #
    inps.nimage = len(inps.listslv)
    inps.lin = slc.shape[0]
    inps.sam = slc.shape[1]
    del slc
    
    inps.pr, inps.pc, inps.patchlist = psq.patch_slice(inps.lin,inps.sam,inps.waz,inps.wra)
                                            
    np.save(inps.sqdir+'/rowpatch.npy',inps.pr)
    np.save(inps.sqdir+'/colpatch.npy',inps.pc) 
    np.save(inps.sqdir + '/patchlist.npy', inps.patchlist)
    
    time0 = time.time()                                        
    if os.path.isfile(inps.sqdir + '/flag.npy'):
        print('patchlist exist')
    else:
        values = [delayed(createpatch)(x) for x in inps.patchlist]
        results = compute(*values, scheduler='processes')
        np.save(inps.sqdir + '/flag.npy', 'patchlist_created')
    timep = time.time() - time0
    logger.info("Done Creating PATCH. time:{}".format(timpep))


    run_PSQ_sentinel = inps.sqdir + "/run_PSQ_sentinel"

    with open(run_PSQ_sentinel, 'w') as f:
        for patch in inps.patchlist:
            cmd_coreg = 'PSQ_sentinel.py ' + templateFileString + '\t' + patch + ' \n'
            f.write(cmd_coreg)
    

           
###########################################
    flag = np.load(inps.sqdir + '/flag.npy')

    if flag == 'patchlist_created':
        cmd = '$INT_SCR/split_jobs.py -f ' + inps.sqdir + '/run_PSQ_sentinel -w 40:00 -r 3700'
        status = subprocess.Popen(cmd, shell=True).wait()
        if status is not 0:
            logger.error('ERROR running PSQ_sentinel.py')
            raise Exception('ERROR running PSQ_sentinel.py')

    for d in inps.patchlist:
        ff0 = np.str(d)
        d = ff0[2:-1]
        if os.path.isfile(inps.sqdir + '/' + d + '/endflag.npy'):
            count = 'True'
        else:
            print(str(d))
            count = 'False'
    if count == 'True':
        cmd = '$SQUEESAR/wrslclist_sentinel.py ' + inps.custom_template_file
        status = subprocess.Popen(cmd, shell=True).wait()
        if status is not 0:
            logger.error('ERROR making run_writeSQ list')
            raise Exception('ERROR making run_writeSQ list')

        run_write = inps.proj_dir + '/merged/run_writeSLC'
        cmd = '$INT_SCR/split_jobs.py -f ' + inps.proj_dir + '/merged/run_writeSLC -w 1:00 -r 5000'
        status = subprocess.Popen(cmd, shell=True).wait()
        if status is not 0:
            logger.error('ERROR writing SLCs')
            raise Exception('ERROR writing SLCs')


if __name__ == '__main__':
  main(sys.argv[:])    

