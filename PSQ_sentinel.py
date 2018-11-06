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
    ns = seq_df.refp[0]
    nlines = np.shape(pixelsdictref['amp'])[0]
    values = [delayed(phaselink_func)(x) for x in mydf]
    results = compute(*values, scheduler='processes')
    squeezed = np.zeros([inps.lin,inps.sam])+0j
    for t in range(inps.lin):
        for q in range(inps.sam):
            try:
                pixelsdictref['amp'][:,t:t+1,q:q+1] = results[t][q].ampref[ns::,1,1]
                pixelsdictref['ph'][:,t:t+1,q:q+1] = results[t][q].phref[ns::,1,1]
            except:
                print('pixel({}, {}) is not DS'.format(t,q))
            Z = np.multiply(pixelsdict['amp'][ns::,t,q],np.exp(1j*pixelsdict['ph'][ns::,t,q])).(10,1)
            Pmap = np.exp(1j*results[t][q].phref[ns::,1,1]).reshape(nlines,1)
            Pmap = np.matrix(Pmap/LA.norm(Pmap))
            squeezed(t,q) = np.matmul(Pmap.getH(),Z)           
    return squeezed  

  
###################################
if __name__ == "__main__":
    
    global pixelsdict, pixelsdictref
    
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
    
################### Finding Statistical homogeneous pixels ################

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
    
    for t in range(inps.lin):
        for q in range(inps.sam):
            shp_df.at[t,q] = results[t][q]
    del results
    shp_df.to_pickle(inps.work_dir + '/shp.pkl')

    print('SHP created ...')
    
###################### Sequential Phase linking ###############################
    
    RSLCamp_ref = np.zeros([inps.nimage, inps.lin, inps.sam])
    RSLCamp_ref[:,:,:] = RSLCamp[:,:,:]
    RSLCphase_ref = np.zeros([inps.nimage, inps.lin, inps.sam])
    RSLCphase_ref[:,:,:] = RSLCphase[:,:,:]

    num_seq = np.floor(inps.nimage/10)
    sequential_df = pd.DataFrame(np.zeros(shape=[num_seq, 1]))
    psq.seqobj(sequential_df)
    shp_df = pd.read_pickle(inps.work_dir + '/shp.pkl')
    mydf = [shp_df.loc[y] for y in xl]
    
    time0 = time.time()
    for t in range(num_seq):
        if pixelsdictref:
            pixelsdictref = None
        seq_df = sequential_df.at[t,0]
        sl = t*10
        if seq_df = num_seq:
            el = inps.nimage
        else:
            el = sl+10
        if t == 0:
            AMP = RSLCamp[sl:el,:,:]
            PHAS = RSLCphase[sl:el,:,:]
            pixelsdict = {'amp':AMP,'ph':PHAS}
            pixelsdictref = {'amp':RSLCamp_ref[sl:el,:,:],'ph':RSLCphase_ref[sl:el,:,:]}
            squeezed_image = sequential_process(mydf,seq_df)
            sequential_df.at[t,0].squeezed = squeezed_image
        else:
            AMP = np.zeros([t+10,lin,sam])
            AMP[0:t,:,:] = np.abs(sequential_df.at[t-1,0].squeezed)
            AMP[t::,:,:] = RSLCamp[sl:el,:,:]
            PHAS = np.zeros([t+10,lin,sam])
            PHAS[0:t,:,:] = np.angle(sequential_df.at[t-1,0].squeezed)
            PHAS[t::,:,:] = RSLCphase[sl:el,:,:]
            pixelsdict = {'amp':AMP,'ph':PHAS}
            pixelsdictref = {'amp':RSLCamp_ref[sl:el,:,:],'ph':RSLCphase_ref[sl:el,:,:]}
            squeezed_image = np.dstack(sequential_df.at[t-1,0].squeezed.T,sequential_process(mydf,seq_df).T).T
            sequential_df.at[t,0].squeezed = squeezed_image
        RSLCamp_ref[sl:el,:,:] = pixelsdictref['amp']
        RSLCphase_ref[sl:el,:,:] = pixelsdictref['ph']
    
    ############## Datum Connection ##############################
    pixelsdict = {'amp':np.abs(sequential_df.at[num_seq-1,0].squeezed),
                  'ph':np.angle(sequential_df.at[num_seq-1,0].squeezed)}
    
    values = [delayed(phaselink_func)(x) for x in mydf]
    results = compute(*values, scheduler='processes')
    datum_connect = np.zeros([num_seq,inps.lin,inps.sam])
    for t in range(inps.lin):
        for q in range(inps.sam):
            datum_connect[:,t,q] = results[t][q].phref[:,1,1].reshape(num_seq,1,1)
            
    for t in range(num_seq):
        sl = t*10
        if seq_df = num_seq:
            el = inps.nimage
        else:
            el = sl+10
        RSLCphase_ref[sl:el,:,:] = RSLCphase_ref[sl:el,:,:] + datum_connect[t,:,:]
              
    timep = time.time() - time0
    logger.info('time spent to do sequential phase linking {}: {}'.format(timep))      
    
    np.save(inps.work_dir + '/endflag.npy', 'True')           
    np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
    np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)
    sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
    
   
    
    

#################################################








