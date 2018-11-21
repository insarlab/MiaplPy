#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys
import time

import argparse
import numpy as np
from scipy.stats import anderson_ksamp
from skimage.measure import label
from _pysqsar_utilities import trwin
import pandas as pd
sys.path.insert(0, os.getenv('RSMAS_ISCE'))
from dataset_template import Template

#################################
EXAMPLE = """example:
  find_shp.py LombokSenAT156VV.template -p PATCH5_11
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
    
    inps.range_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizerange'])
    inps.azimuth_win = int(Template(inps.custom_template_file).get_options()['squeesar.wsizeazimuth'])
    
    inps.patch_rows = np.load(inps.sq_dir + '/rowpatch.npy')
    inps.patch_cols = np.load(inps.sq_dir + '/colpatch.npy')
    patch_row, patch_col = inps.patch_dir.split('PATCH')[1].split('_')
    patch_row, patch_col = (int(patch_row), int(patch_col))

    inps.lin = inps.patch_rows[1][0][patch_row] - inps.patch_rows[0][0][patch_row]
    inps.sam = inps.patch_cols[1][0][patch_col] - inps.patch_cols[0][0][patch_col]

    rslc = np.memmap(inps.work_dir + '/RSLC', dtype=np.complex64, mode='r', shape=(inps.n_image, inps.lin, inps.sam))


    ################### Finding Statistical homogeneous pixels ################
    num_slc = 20    # to find SHPs only
    if not os.path.isfile(inps.work_dir + '/SHP.pkl'):
        
        time0 = time.time()
        shp_df = pd.DataFrame(columns=['ref_pixel','rows','cols','amp_ref','phase_ref'])
        lin = np.ogrid[0:inps.lin]
        sam = np.ogrid[0:inps.sam]
        lin, sam = np.meshgrid(lin, sam)
        coords = list(map(lambda x, y: (int(x), int(y)),
                     lin.T.reshape(inps.lin*inps.sam, 1), sam.T.reshape(inps.lin*inps.sam, 1)))
        del lin, sam

        for coord in coords:

            lin,sam = coord
            r = np.ogrid[lin - ((inps.azimuth_win - 1) / 2):lin + ((inps.azimuth_win - 1) / 2) + 1]
            ref_row = np.array([(inps.azimuth_win - 1) / 2])
            r = r[r >= 0]
            r = r[r < inps.lin]
            ref_row = ref_row - (inps.azimuth_win - len(r))
            c = np.ogrid[sam - ((inps.range_win - 1) / 2):sam + ((inps.range_win - 1) / 2) + 1]
            ref_col = np.array([(inps.range_win - 1) / 2])
            c = c[c >= 0]
            c = c[c < inps.sam]
            ref_col = ref_col - (inps.range_win - len(c))

            x, y = np.meshgrid(r.astype(int), c.astype(int), sparse=True)
            win = np.abs(rslc[0:num_slc, x, y])
            win = trwin(win)

            test_vec = win.reshape(num_slc, len(r) * len(c))
            ks_res = np.zeros(len(r) * len(c))
            ref_scatterer = np.abs(rslc[0:num_slc, lin, sam])
            ref_scatterer = ref_scatterer.reshape(num_slc, 1)
            for pixel in range(len(test_vec[0])):
                sample_scatterer = test_vec[:, pixel]
                sample_scatterer = sample_scatterer.reshape(num_slc, 1)

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
                shp = {'ref_pixel': [lin, sam], 'rows': rr, 'cols': cc, 'amp_ref':{}, 'phase_ref':{}}
                shp_df = shp_df.append(shp, ignore_index=True)

        shp_df.to_pickle(inps.work_dir + '/SHP.pkl')
        
        timep = time.time() - time0
        print('time spent to find SHPs {}: min'.format(timep/60))
    else:
        print('SHP Exists...')
        
########################################        
if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

