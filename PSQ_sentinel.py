#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import glob
import time
import matplotlib.pyplot as plt
from rsmas_logging import rsmas_logger, loglevel
import pysar
from pysar.utils import utils
from pysar.utils import readfile
import subprocess
import cmath
from scipy import linalg
import numpy as np
from numpy.linalg import pinv
import multiprocessing
import _pysqsar_utilities as psq
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
import _process_utilities as putils
import pandas as pd
from dask import dataframe as dd 
from dask import compute, delayed
import dask.multiprocessing

#################################
EXAMPLE = """example:
  sentinel_squeesar.py LombokSenAT156VV.template PATCH5_11
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
    parser.add_argument('Patch_file', dest='patchDir', action='store_true', help='patch file directory')
    
    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    global inps

    parser = create_parser()
    inps = parser.parse_args(args)
    
    
def shp_func(data):
    data = psq.shp_loc(data, pixelsdict=pixelsdict)
    return data


def phaselink_func(data):
    data = psq.phase_link(data, pixelsdict=pixelsdict)
    return data

###################################
if __name__ == "__main__":
                                            
    command_line_parse(sys.argv[1:])
    inps.project_name = putils.get_project_name(custom_template_file=inps.custom_template_file)
    inps.work_dir = putils.get_work_directory(None, inps.project_name)
    templateContents = readfile.read_template(inps.custom_template_file)
    inps.scratch_dir = os.getenv('SCRATCHDIR')
    inps.slave_dir = inps.work_dir + '/merged/SLC'
    inps.sqdir = inps.work_dir + '/SqueeSAR'
    inps.listslv = os.listdir(slave_dir)  
    inps.nimage = len(inps.listslv)
    inps.work_dir = inps.sqdir + inps.patchDir
        
    RSLCamp = np.load(inps.work_dir + '/Amplitude.npy')
    RSLCphase = np.load(inps.work_dir + '/Phase.npy')

    inps.lin = np.size(RSLCamp, axis=1)
    inps.sam = np.size(RSLCamp, axis=2)

    inps.wra = int(templateContents['squeesar.wsizerange'])
    inps.waz = int(templateContents['squeesar.wsizeazimuth'])
    
###################

    global pixelsdict
    pixelsdict = {'amp':RSLCamp}

    shp_df = pd.DataFrame(np.zeros(shape=[inps.lin, inps.sam]))    
    psq.shpobj(shp_df)
    shp_df = shp_df.apply(np.vectorize(psq.win_loc), wra=inps.wra, waz=inps.waz, nimage=inps.nimage, lin=inps.lin, sam=inps.sam)

    time0 = time.time()
    xl = np.arange(inps.lin)
    mydf = [shp_df.loc[y] for y in xl]
    values = [delayed(shp_func)(x) for x in mydf]
    results = compute(*values, scheduler='processes')
    timep = time.time() - time0
    logger.info('time spent to find SHPs: {}'.format(timep))
    
    for t in range(lin):
        for q in range(sam):
            shp_df.at[t,q] = results[t][q]
    del results
    shp_df.to_pickle(inps.work_dir + '/shp.pkl')

    print('SHP created ...')
    
#####################################################
    RSLCamp_ref = np.zeros([nimage, lin, sam])
    RSLCamp_ref[:,:,:] = RSLCamp[:,:,:]
    RSLCphase_ref = np.zeros([nimage, lin, sam])
    RSLCphase_ref[:,:,:] = RSLCphase[:,:,:]
    pixelsdict = {'amp':RSLCamp,'ph':RSLCphase,'amp_ref':RSLCamp_ref,'ph_ref':RSLCphase_ref,'work_dir':inps.work_dir}
    
    if os.path.isfile(inps.work_dir + '/Phase_ref.npy'):
        print(inps.patchDir+' is already done' )
    else:
        time0 = time.time()
        shp_df = pd.read_pickle(inps.work_dir + '/shp.pkl')
        mydf = [shp_df.loc[y] for y in xl]
        values = [delayed(phaselink_func)(x) for x in mydf]
        results = compute(*values, scheduler='processes')
        timep = time.time() - time0
        logger.info('time spent to find SHPs in {}: {}'.format(inps.patchDir, timep))
        
    np.save(inps.work_dir + '/endflag.npy', 'True')


#################################################


if __name__ == '__main__':
    main(sys.argv[:])








