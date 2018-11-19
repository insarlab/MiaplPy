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
from scipy.stats import anderson_ksamp
from skimage.measure import label
import _pysqsar_utilities as pysq
import pandas as pd
from dask import compute, delayed
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from dataset_template import Template



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


def sequential_process(shp_df_chunk, sequential_df_chunk, inps, pixels_dict={}, pixels_dict_ref={}):
    seq_n = sequential_df_chunk['step']   # sequence number
    n_lines = np.shape(pixels_dict_ref['amp'])[0]
    values = [delayed(pysq.phase_link)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
    results = pd.DataFrame(list(compute(*values, scheduler='processes')))
    squeezed = np.zeros([inps.lin, inps.sam]) + 0j
    pixels_dict_ref_new = pixels_dict_ref
    mydf = [results.loc[y] for y in range(len(results))]
    for item in mydf:
        lin,sam = item['ref_pixel'][0],item['ref_pixel'][1]
        try:
            pixels_dict_ref_new['amp'][:, lin:lin + 1, sam:sam + 1] = item['amp_ref'][seq_n::, 0, 0]
            pixels_dict_ref_new['ph'][:, lin:lin + 1, sam:sam + 1] = item['phase_ref'][seq_n::, 0, 0]
        except:
            print('pixel({}, {}) is not DS'.format(lin, sam))
        org_pixel = np.multiply(pixels_dict['amp'][seq_n::, lin, sam],
                                np.exp(1j * pixels_dict['ph'][seq_n::, lin, sam])).reshape(n_lines, 1)
        map_pixel = np.exp(1j * item['phase_ref'][seq_n::, 0, 0]).reshape(n_lines, 1)
        map_pixel = np.matrix(map_pixel / LA.norm(map_pixel))
        squeezed[lin, sam] = np.matmul(map_pixel.getH(), org_pixel)
    return squeezed, pixels_dict_ref_new


###################################
def main(iargs=None):
    inps = command_line_parse(iargs)

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
    
    if not os.path.isfile(inps.work_dir + '/SHP.pkl'):
        
        time0 = time.time()
        shp_df = pd.DataFrame(columns=['ref_pixel','rows','cols','amp_ref','phase_ref'])
        lin = 0
        while lin < inps.lin:  # rows
            r = np.ogrid[lin - ((inps.azimuth_win - 1) / 2):lin + ((inps.azimuth_win - 1) / 2) + 1]
            ref_row = np.array([(inps.azimuth_win - 1) / 2])
            r = r[r >= 0]
            r = r[r < inps.lin]
            ref_row = ref_row - (inps.azimuth_win - len(r))
            sam = 0
            while sam < inps.sam:
                c = np.ogrid[sam - ((inps.range_win - 1) / 2):sam + ((inps.range_win - 1) / 2) + 1]
                ref_col = np.array([(inps.range_win - 1) / 2])
                c = c[c >= 0]
                c = c[c < inps.sam]
                ref_col = ref_col - (inps.range_win - len(c))
                x, y = np.meshgrid(r.astype(int), c.astype(int), sparse=True)
                win = RSLCamp[:, x, y]
                win = psq.trwin(win)
                test_vec = win.reshape(inps.n_image, len(r) * len(c))
                ks_res = np.zeros(len(r) * len(c))
                ref_scatterer = RSLCamp[:, lin, sam]
                ref_scatterer = ref_scatterer.reshape(inps.n_image, 1)
                for pixel in range(len(test_vec[0])):
                    sample_scatterer = test_vec[:, pixel]
                    sample_scatterer = sample_scatterer.reshape(inps.n_image, 1)

                    try:
                        test = anderson_ksamp([ref_scatterer, sample_scatterer])
                        if test.significance_level > 0.05:
                            ks_res[pixel] = 1
                        else:
                            ks_res[pixel] = 0
                    except:
                        ks_res[pixel] = 0

                ks_result = ks_res.reshape(len(r), len(c))
                ks_label = label(ks_result, background=False, connectivity=2)
                ref_label = ks_label[ref_row.astype(int), ref_col.astype(int)]

                rr, cc = np.where(ks_label == ref_label)

                rr = rr + r[0]
                cc = cc + c[0]
                if len(rr) > 20:
                    shp = {'ref_pixel': [lin, sam], 'rows': rr, 'cols': cc}
                    shp_df = shp_df.append(shp, ignore_index=True)
                sam = sam + 1

            lin = lin + 1

        shp_df.to_pickle(inps.work_dir + '/SHP.pkl')
        
        timep = time.time() - time0
        print('time spent to find SHPs {}: min'.format(timep/60))
    else:
        print('SHP Exists...')

    ###################### Sequential Phase linking ###############################
    if not os.path.isfile(inps.work_dir + '/sequential_df.pkl'):
      
        RSLCamp_ref = np.zeros([inps.n_image, inps.lin, inps.sam])
        RSLCamp_ref[:, :, :] = RSLCamp[:, :, :]
        RSLCphase_ref = np.zeros([inps.n_image, inps.lin, inps.sam])
        RSLCphase_ref[:, :, :] = RSLCphase[:, :, :]

        num_seq = np.int(np.floor(inps.n_image / 10))
        sequential_df = pd.DataFrame(columns=['step', 'squeezed'])
        for seq in range(num_seq):
            sequential_df = sequential_df.append({'step': seq}, ignore_index=True)

    
        shp_df = pd.read_pickle(inps.work_dir + '/SHP.pkl')
        shp_df_chunk = [shp_df.loc[y] for y in range(len(shp_df))]

        time0 = time.time()
        for step in range(num_seq):
      
            try: 
                del pixels_dict
                print('Next Sequence...')
            except:
                print('Next Sequence...')

            first_line = step  * 10
            if step == num_seq-1:
                last_line = inps.n_image
            else:
                last_line = first_line + 10
            if step == 0:
                AMP = RSLCamp[first_line:last_line, :, :]
                PHAS = RSLCphase[first_line:last_line, :, :]
                pixels_dict = {'amp': AMP, 'ph': PHAS}
                pixels_dict_ref = {'amp': RSLCamp_ref[first_line:last_line, :, :], 
                                   'ph': RSLCphase_ref[first_line:last_line, :, :]}
                squeezed_image, pixels_dict_ref = \
                    sequential_process(shp_df_chunk=shp_df_chunk,
                                       sequential_df_chunk=sequential_df.loc[step],
                                       inps=inps, pixels_dict=pixels_dict,
                                       pixels_dict_ref=pixels_dict_ref)
                sequential_df.loc[step]['squeezed'] = squeezed_image
            else:
                AMP = np.zeros([step + 10, inps.lin, inps.sam])
                AMP[0:step, :, :] = np.abs(sequential_df.at[step - 1, 0].squeezed)
                AMP[step::, :, :] = RSLCamp[first_line:last_line, :, :]
                PHAS = np.zeros([step + 10, inps.lin, inps.sam])
                PHAS[0:step, :, :] = np.angle(sequential_df.at[step - 1, 0].squeezed)
                PHAS[step::, :, :] = RSLCphase[first_line:last_line, :, :]
                pixels_dict = {'amp': AMP, 'ph': PHAS}
                pixels_dict_ref = {'amp': RSLCamp_ref[first_line:last_line, :, :], 'ph': RSLCphase_ref[first_line:last_line, :, :]}
                squeezed_im, pixels_dict_ref = \
                    sequential_process(shp_df_chunk=shp_df_chunk,
                                       sequential_df_chunk=sequential_df.loc[step],
                                       inps=inps, pixels_dict=pixels_dict,
                                       pixels_dict_ref=pixels_dict_ref)
                squeezed_image = np.dstack((sequential_df.loc[step - 1]['squeezed'].T,squeezed_im.T)).T
                sequential_df.loc[step]['squeezed'] = squeezed_image
        
            RSLCamp_ref[first_line:last_line, :, :] = pixels_dict_ref['amp']
            RSLCphase_ref[first_line:last_line, :, :] = pixels_dict_ref['ph']
        
            np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
            np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)
            sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
    
        

    ############## Datum Connection ##############################
        pixels_dict = {'amp': np.abs(sequential_df.at[num_seq - 1, 0].squeezed),
                      'ph': np.angle(sequential_df.at[num_seq - 1, 0].squeezed)}

        values = [delayed(pysq.phase_link)(x,pixels_dict=pixels_dict) for x in shp_df_chunk]
        results = pd.DataFrame(list(compute(*values, scheduler='processes')))
        datum_connect = np.zeros([num_seq, inps.lin, inps.sam])
        mydf = [results.loc[y] for y in range(len(results))]
    
        for item in mydf:
            lin,sam = item.ref_pixel[0],item.ref_pixel[1]
            datum_connect[:, lin:lin+1, sam:sam+1] = item.phref[:, 0, 0].reshape(num_seq, 1, 1)
    
        for step in range(num_seq):
            first_line = step * 10
            if step == num_seq-1:
                last_line = inps.n_image
            else:
                last_line = first_line + 10
            RSLCphase_ref[first_line:last_line, :, :] = RSLCphase_ref[first_line:last_line, :, :] + datum_connect[step, :, :]

        np.save(inps.work_dir + '/endflag.npy', 'True')
        np.save(inps.work_dir + '/Amplitude_ref.npy', RSLCamp_ref)
        np.save(inps.work_dir + '/Phase_ref.npy', RSLCphase_ref)
        sequential_df.to_pickle(inps.work_dir + '/sequential_df.pkl')
        
        timep = time.time() - time0
        print('time spent to do sequential phase linking {}: min'.format(timep/60))

if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

#################################################
