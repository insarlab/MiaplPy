#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
from numpy import linalg as LA
import numpy as np
import _pysqsar_utilities as pysq
import pandas as pd
from dask import compute, delayed
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from dataset_template import Template

global rslc, sequential_df, rslc_ref

#################################
EXAMPLE = """example:
  PSQ_sentinel.py LombokSenAT156VV.template -p PATCH5_11
"""


def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=EXAMPLE)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('custom_template_file', nargs='?',
                        help='custom template with option settings.\n')
    parser.add_argument('-p','--patchdir', dest='patch_dir', type=str, required=True, help='patch file directory')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def sequential_process(ccg_sample, stepp, method):

    coh_mat = pysq.est_corr(ccg_sample)
    if method == 'PTA':
        ph_EMI = pysq.EMI_phase_estimation(coh_mat)
        xm = np.zeros([len(ph_EMI),len(ph_EMI)+1])+0j
        xm[:,0:1] = np.reshape(ph_EMI,[len(ph_EMI),1])
        xm[:,1::] = coh_mat[:,:]
        res = pysq.PTA_L_BFGS(xm)
    elif method == 'EMI':
        res = pysq.EMI_phase_estimation(coh_mat)
    elif method == 'EVD':
        res = pysq.EVD_phase_estimation(coh_mat)
        
    res = res.reshape(len(res),1)
    
    vm = np.matrix(np.exp(1j*res[stepp::,0])/LA.norm(np.exp(1j*res[stepp::,0])))
    
    squeezed = np.matmul(np.conjugate(vm),ccg_sample[stepp::,:])

    return res, squeezed


def sequential_phase_linking(CCG, ref_row, ref_col, rows, cols, method):
    step_0 = np.uint32(sequential_df.at[0,'step_n'])
    mat_shape = np.shape(sequential_df.at[0,'squeezed'])
    n_image = CCG.shape[0]
    num_seq = np.int(np.floor(n_image / 10))
    phase_ref = np.float32(np.zeros([n_image,1]))
    phase_ref[:,0:1] = np.angle(rslc_ref[:,ref_row, ref_col]).reshape(n_image,1)  

    squeezed_image = np.matrix(sequential_df.at[0,'squeezed'][:,rows,cols])
    
    for stepp in range(step_0, num_seq):
        
        first_line = stepp  * 10
        if stepp == num_seq-1:
            last_line = n_image
        else:
            last_line = first_line + 10
        num_lines = last_line - first_line
        if stepp == 0:
          
            ccg_sample = CCG[first_line:last_line,:]
            res, squeezed_image = sequential_process(ccg_sample, stepp, method)
            phase_ref[first_line:last_line,0:1] = res[stepp::].reshape(num_lines,1)
            
        else:
            
            ccg_sample = np.zeros([1 + num_lines, CCG.shape[1]])+1j
            ccg_sample[0, :] = np.complex64(squeezed_image[-1,:])
            ccg_sample[1::, :] = CCG[first_line:last_line, :]
            res, squeezed_im = sequential_process(ccg_sample, 1, method)
            phase_ref[first_line:last_line,0:1] = res[1::].reshape(num_lines,1)
            squeezed_image = np.complex64(np.vstack([squeezed_image,squeezed_im]))
            
    sequential_df.at[0,'step_n'] = np.uint32(num_seq)
    sequential_df.at[0,'squeezed'] = np.complex64(np.zeros([squeezed_image.shape[0],mat_shape[1],mat_shape[2]]))
    for t in range(len(rows)):
        sequential_df.at[0,'squeezed'][:,rows[t]:rows[t]+1,cols[t]:cols[t]+1] = np.array(squeezed_image[:,t]).reshape(squeezed_image.shape[0],1,1)
        
    
    ccg_datum = squeezed_image    
    datumshift = sequential_df.at[0, 'datum_shift'][:,ref_row, ref_col]
    res_d, squeezed_d = sequential_process(ccg_datum, 0, method)
    del squeezed_d
    for stepp in range(len(res_d)):
        first_line = stepp * 10
        if stepp == num_seq-1:
            last_line = n_image
        else:
            last_line = first_line + 10
        
        if  step_0 == 0:
            phase_ref[first_line:last_line, 0:1] = phase_ref[first_line:last_line, 0] + np.array(res_d[int(stepp)])
        else:
            phase_ref[first_line:last_line, 0:1] = phase_ref[first_line:last_line, 0] + np.array(res_d[int(stepp)]) - datumshift
            
    sequential_df.at[0, 'datum_shift'][:,ref_row+1, ref_col+1] = np.float32(res_d).reshape(len(res_d),1,1)
    
    
    phase_init = np.triu(np.angle(np.matmul(CCG, CCG.getH()) / (len(rows))),1)
    phase_optimized = np.triu(np.angle(np.matmul(np.exp(-1j * phase_ref), (np.exp(-1j * phase_ref)).getH())), 1)
    gam_pta = pysq.gam_pta_f(phase_init, phase_optimized)
    
    if 0.4 < gam_pta <= 1:
        out = phase_ref
    else:
        out = np.angle(rslc_ref[:,ref_row, ref_col]).reshape(n_image,1)    
            
    return  out 
  
  
def shp_locate(mydf,method):
    n_image = rslc.shape[0]
    rr = mydf.at['rows'].astype(int)
    cc = mydf.at['cols'].astype(int)
    ref_row, ref_col = (mydf.at['ref_pixel'][0],mydf.at['ref_pixel'][1])
    CCG = np.matrix(1.0 * np.arange(n_image * len(rr)).reshape(n_image, len(rr)))
    CCG = np.exp(1j * CCG)
    CCG[:,:] = np.matrix(rslc[:, rr, cc]) 
    amp_ref = np.mean(np.abs(CCG),axis=0)
    ph_ref = sequential_phase_linking(CCG, ref_row, ref_col, rr, cc, method)
    rslc_ref[:,ref_row, ref_col] = np.complex64(np.multiply(amp_ref,np.exp(1j*ph_ref)).reshape(len(ph_ref),1,1))
    
    return None
  

###################################

def main(iargs=None):
    inps = command_line_parse(iargs)

    inps.project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    inps.project_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name
    inps.scratch_dir = os.getenv('SCRATCHDIR')
    
    inps.slave_dir = inps.project_dir + '/merged/SLC'
    inps.sq_dir = inps.project_dir + '/SqueeSAR'
    inps.list_slv = os.listdir(inps.slave_dir)
    inps.n_image = len(inps.list_slv)
    inps.work_dir = inps.sq_dir +'/'+ inps.patch_dir
    
    inps.patch_rows = np.load(inps.sq_dir + '/rowpatch.npy')
    inps.patch_cols = np.load(inps.sq_dir + '/colpatch.npy')
    patch_row, patch_col = inps.patch_dir.split('PATCH')[1].split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))

    inps.lin = inps.patch_rows[1][0][patch_row] - inps.patch_rows[0][0][patch_row]
    inps.sam = inps.patch_cols[1][0][patch_col] - inps.patch_cols[0][0][patch_col]
    
    inps.range_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizerange'])
    inps.azimuth_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizeazimuth'])

    global rslc, sequential_df, rslc_ref
    ###################### Sequential Phase linking ###############################
    
    time0 = time.time()
    num_seq = np.int(np.floor(inps.n_image / 10))
    if os.path.isfile(inps.work_dir + '/sequential_df.pkl'):
        sequential_df = pd.read_pickle(inps.work_dir + '/sequential_df.pkl')
    else:
        sequential_df = pd.DataFrame(columns=['step_n', 'squeezed','datum_shift','n_images_done'])
        sequential_df = sequential_df.append({'step_n':np.uint32(0), 
                                              'squeezed':np.complex64(np.zeros([1,inps.lin,inps.sam])), 
                                              'datum_shift':np.float32(np.zeros([num_seq,inps.lin,inps.sam]))}, ignore_index=True)
                                              
    
    rslc = np.memmap(inps.work_dir + '/RSLC', dtype=np.complex64, mode='r', shape=(inps.n_image, inps.lin, inps.sam))
    
    step_0 = np.uint32(sequential_df.at[0,'step_n'])
    if step_0 == 0:
        rslc_ref = np.memmap(inps.work_dir + '/RSLC_ref', dtype='complex64', mode='w+', shape=(inps.n_image, inps.lin, inps.sam))
        rslc_ref[:,:,:] = rslc[:,:,:]
    else:
        rslc_ref = np.memmap(inps.work_dir + '/RSLC_ref', dtype='complex64', mode='r+', shape=(inps.n_image, inps.lin, inps.sam))
        rslc_ref[step_0*10::,:,:] = rslc[step_0*10::,:,:]
    
    method = 'EMI'                                          
    shp_df = pd.read_pickle(inps.work_dir + '/SHP.pkl')
    shp_df_chunk = [shp_df.loc[y] for y in range(len(shp_df))]  
    values = [delayed(shp_locate)(x,method) for x in shp_df_chunk]
    compute(*values, scheduler='processes')
                                              
    sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
    
    np.save(inps.work_dir + '/endflag.npy', 'True')
    
    del rslc_ref, rslc
        
    timep = time.time() - time0
    print('time spent to do sequential phase linking {}: min'.format(timep/60))

if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

#################################################
