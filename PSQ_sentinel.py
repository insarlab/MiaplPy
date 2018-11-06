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
from numpy import linalg as LA
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

  
def sequential_process(mydf,seq_df):
    ns = seq_df.refp[0]*10
    values = [delayed(phaselink_func)(x) for x in mydf]
    results = compute(*values, scheduler='processes')
    squeezed = np.zeros([lin,sam])+0j
    for t in range(lin):
        for q in range(sam):
            try:
                RSLCamp_ref[ns:ns:10,t:t+1,q:q+1] = results[t][q].ampref[seq_df.refp[0]::,1,1]
                RSLCphase_ref[ns:ns:10,t:t+1,q:q+1] = results[t][q].phref[seq_df.refp[0]::,1,1]
            except:
                print('not DS')
            Z = np.multiply(pixelsdict['amp'][ns:ns:10,t,q],np.exp(1j*pixelsdict['ph'][ns:ns:10,t,q])).(10,1)
            Pmap = np.exp(1j*results[t][q].phref[seq_df.refp[0]::,1,1]).reshape(10,1)
            Pmap = np.matrix(Pmap/LA.norm(Pmap))
            squeezed(t,q) = np.matmul(Pmap.getH(),Z)           
    return squeezed
  
  
###################################
if __name__ == "__main__":
    
    global pixelsdict, RSLCamp_ref, RSLCamp_ref
    
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

    pixelsdict = {'amp':RSLCamp,'ph':RSLCphase}
    num_seq = np.floor(inps.nimage/10)
    sequential_df = pd.DataFrame(np.zeros(shape=[num_seq, 1]))
    psq.seqobj(sequential_df)
    
    for t in range(num_seq):
        seq_df = sequential_df.at[t,0]
        if t == 0:
            
        else:
          
    
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
        
    RSLCamp_ref = np.zeros([nimage, lin, sam])
    RSLCamp_ref[:,:,:] = RSLCamp[:,:,:]
    RSLCphase_ref = np.zeros([nimage, lin, sam])
    RSLCphase_ref[:,:,:] = RSLCphase[:,:,:]
    
    for t in range(lin):
      for q in range(sam):
          try:
              RSLCamp_ref[:,t:t+1,q:q+1] = results[t][q].ampref
              RSLCphase_ref[:,t:t+1,q:q+1] = results[t][q].phref
          except:
              print('not DS')
    
    np.save(inps.work_dir + '/endflag.npy', 'True')           
    np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
    np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)

#################################################








