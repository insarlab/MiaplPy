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
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from dataset_template import Template
from rsmas_logging import loglevel
import pandas as pd
from dask import compute, delayed


logger_PSQ = pysq.send_logger_squeesar()

#################################
EXAMPLE = """example:
  PSQ_sentinel.py LombokSenAT156VV.template PATCH5_11
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


#def shp_func(data):
#    data = pysq.shp_loc(data, pixelsdict=pixelsdict)
#    return data


#def phaselink_func(data):
#    data = pysq.phase_link(data, pixelsdict=pixelsdict)
#    return data


def sequential_process(df_chunk, sequential_df_chunk, inps, pixels_dict={}, pixels_dict_ref={}):
    seq_n = sequential_df_chunk.ref_pixel[0]   # sequence number
    n_lines = np.shape(pixels_dict_ref['amp'])[0]
    values = [delayed(pysq.phase_link)(x,pixelsdict=pixels_dict) for x in df_chunk]
    results = compute(*values, scheduler='processes')
    squeezed = np.zeros([inps.lin, inps.sam]) + 0j
    pixels_dict_ref_new = pixels_dict_ref
    for lin in range(inps.lin):
        for sam in range(inps.sam):
            try:
                pixels_dict_ref_new['amp'][:, lin:lin + 1, sam:sam + 1] = results[lin][sam].ampref[seq_n::, 1, 1]
                pixels_dict_ref_new['ph'][:, lin:lin + 1, sam:sam + 1] = results[lin][sam].phref[seq_n::, 1, 1]
            except:
                print('pixel({}, {}) is not DS'.format(lin, sam))
            org_pixel = np.multiply(pixels_dict['amp'][seq_n::, lin, sam],
                            np.exp(1j * pixels_dict['ph'][seq_n::, lin, sam])).reshape(10, 1)
            map_pixel = np.exp(1j * results[lin][sam].phref[seq_n::, 1, 1]).reshape(n_lines, 1)
            map_pixel = np.matrix(map_pixel / LA.norm(map_pixel))
            squeezed[lin, sam] = np.matmul(map_pixel.getH(), org_pixel)
    return squeezed, pixels_dict_ref_new


###################################
def main(iargs=None):
    inps = command_line_parse(iargs)

    logger_PSQ.log(loglevel.INFO, os.path.basename(sys.argv[0]) + " " + sys.argv[1]+ " " + sys.argv[2])

    inps.project_name = os.path.basename(inps.custom_template_file).partition('.')[0]
    inps.project_dir = os.getenv('SCRATCHDIR') + '/' + inps.project_name
    inps.template = Template(inps.custom_template_file).get_options()
    inps.scratch_dir = os.getenv('SCRATCHDIR')
    inps.slave_dir = inps.project_dir + '/merged/SLC'
    inps.sq_dir = inps.project_dir + '/SqueeSAR'
    inps.list_slv = os.listdir(inps.slave_dir)
    inps.n_image = len(inps.list_slv)
    inps.work_dir = inps.sq_dir +'/'+ inps.patch_dir

    RSLCamp = np.load(inps.work_dir + '/Amplitude.npy')
    RSLCphase = np.load(inps.work_dir + '/Phase.npy')

    inps.lin = np.size(RSLCamp, axis=1)
    inps.sam = np.size(RSLCamp, axis=2)

    inps.range_win = int(inps.template['squeesar.wsizerange'])
    inps.azimuth_win = int(inps.template['squeesar.wsizeazimuth'])

    ################### Finding Statistical homogeneous pixels ################
    xl = np.arange(inps.lin)
    
    if not os.path.isfile(inps.work_dir + '/shp.pkl'):
        pixels_dict = {'amp': RSLCamp[0:20,:,:]}

        shp_df = pd.DataFrame(np.zeros(shape=[inps.lin, inps.sam]))
        pysq.shpobj(shp_df)
        shp_df = shp_df.apply(np.vectorize(pysq.win_loc), wra=inps.range_win,
                              waz=inps.azimuth_win, lin=inps.lin, sam=inps.sam)

        time0 = time.time()
    
        shp_df_chunk = [shp_df.loc[y] for y in xl]
        values = [delayed(pysq.shp_loc)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
        results = compute(*values, scheduler='processes')
        timep = time.time() - time0
        logger_PSQ.log(loglevel.INFO, 'time spent to find SHPs: {}'.format(timep))

        for lin in range(inps.lin):
            for sam in range(inps.sam):
                shp_df.at[lin, sam] = results[lin][sam]
        del results
        shp_df.to_pickle(inps.work_dir + '/shp.pkl')

        print('SHP created ...')
    else:
        print('SHP Exists...')

    ###################### Sequential Phase linking ###############################

    RSLCamp_ref = np.zeros([inps.n_image, inps.lin, inps.sam])
    RSLCamp_ref[:, :, :] = RSLCamp[:, :, :]
    RSLCphase_ref = np.zeros([inps.n_image, inps.lin, inps.sam])
    RSLCphase_ref[:, :, :] = RSLCphase[:, :, :]

    num_seq = np.int(np.floor(inps.n_image / 10))
    sequential_df = pd.DataFrame(np.zeros(shape=[num_seq, 1]))
    pysq.shpobj(sequential_df)
    shp_df = pd.read_pickle(inps.work_dir + '/shp.pkl')
    shp_df_chunk = [shp_df.loc[y] for y in xl]

    time0 = time.time()
    for step in range(num_seq):
        if pixels_dict:
            del pixels_dict
        seq_df = sequential_df.at[step, 0]
        first_line = step  * 10
        if seq_df == num_seq:
            last_line = inps.n_image
        else:
            last_line = first_line + 10
        if step == 0:
            AMP = RSLCamp[first_line:last_line, :, :]
            PHAS = RSLCphase[first_line:last_line, :, :]
            pixels_dict = {'amp': AMP, 'ph': PHAS}
            pixels_dict_ref = {'amp': RSLCamp_ref[first_line:last_line, :, :], 'ph': RSLCphase_ref[first_line:last_line, :, :]}
            squeezed_image, pixels_dict_ref = \
                sequential_process(shp_df_chunk, seq_df,inps,
                                   pixels_dict=pixels_dict, pixels_dict_ref=pixels_dict_ref)
            sequential_df.at[step, 0].squeezed = squeezed_image
        else:
            AMP = np.zeros([step + 10, lin, sam])
            AMP[0:step, :, :] = np.abs(sequential_df.at[step - 1, 0].squeezed)
            AMP[step::, :, :] = RSLCamp[first_line:last_line, :, :]
            PHAS = np.zeros([t + 10, lin, sam])
            PHAS[0:step, :, :] = np.angle(sequential_df.at[step - 1, 0].squeezed)
            PHAS[step::, :, :] = RSLCphase[first_line:last_line, :, :]
            pixels_dict = {'amp': AMP, 'ph': PHAS}
            pixels_dict_ref = {'amp': RSLCamp_ref[first_line:last_line, :, :], 'ph': RSLCphase_ref[first_line:last_line, :, :]}
            squeezed_image, pixels_dict_ref = \
                np.dstack((sequential_df.at[step - 1, 0].squeezed.T,
                           sequential_process(shp_df_chunk, seq_df,
                                              inps,pixels_dict=pixels_dict,pixels_dict_ref=pixels_dict_ref).T)).T
            sequential_df.at[step, 0].squeezed = squeezed_image
            
        np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
        np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)
        sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
    
        RSLCamp_ref[first_line:last_line, :, :] = pixels_dict_ref['amp']
        RSLCphase_ref[first_line:last_line, :, :] = pixels_dict_ref['ph']

    ############## Datum Connection ##############################
    pixels_dict = {'amp': np.abs(sequential_df.at[num_seq - 1, 0].squeezed),
                  'ph': np.angle(sequential_df.at[num_seq - 1, 0].squeezed)}

    values = [delayed(pysq.phase_link)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
    results = compute(*values, scheduler='processes')
    datum_connect = np.zeros([num_seq, inps.lin, inps.sam])
    for lin in range(inps.lin):
        for sam in range(inps.sam):
            datum_connect[:, lin, sam] = results[lin][sam].phref[:, 1, 1].reshape(num_seq, 1, 1)

    for step in range(num_seq):
        first_line = step * 10
        if seq_df == num_seq:
            last_line = inps.n_image
        else:
            last_line = first_line + 10
        RSLCphase_ref[first_line:last_line, :, :] = RSLCphase_ref[first_line:last_line, :, :] + datum_connect[step, :, :]

    timep = time.time() - time0
    logger_PSQ.log(loglevel.INFO, 'time spent to do sequential phase linking {}: {}'.format(timep))

    np.save(inps.work_dir + '/endflag.npy', 'True')
    np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
    np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)
    sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')

if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

#################################################
