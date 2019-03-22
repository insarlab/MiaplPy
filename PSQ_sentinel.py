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
import pysqsar_utilities as pysq
import pandas as pd
from scipy.stats import anderson_ksamp
from skimage.measure import label
from dask import compute, delayed

#################################
def create_parser():
    """ Creates command line argument parser object. """

    parser = argparse.ArgumentParser(description='Crops the scene given bounding box in lat/lon')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--patch_dir', dest='patch_dir', type=str, required=True, help='patch file directory')
    parser.add_argument('-r', '--range_window', dest='range_win', type=str, default='21'
                        , help='SHP searching window size in range direction. -- Default : 21')
    parser.add_argument('-a', '--azimuth_window', dest='azimuth_win', type=str, default='15'
                        , help='SHP searching window size in azimuth direction. -- Default : 15')
    parser.add_argument('-m', '--plmethod', dest='phase_linking_method', type=str, default='sequential_EMI', help='Phase linking method')

    return parser


def command_line_parse(args):
    """ Parses command line agurments into inps variable. """

    parser = create_parser()
    inps = parser.parse_args(args)
    return inps


def main(iargs=None):

    inps = command_line_parse(iargs)
    count_dim = np.load(inps.patch_dir+'/count.npy')
    inps.n_image = count_dim[0]
    inps.lin = count_dim[1]
    inps.sam = count_dim[2]

    inps.range_win = int(inps.range_win)
    inps.azimuth_win = int(inps.azimuth_win)

    ###################### Find SHPs ###############################

    rslc = np.memmap(inps.patch_dir + '/RSLC', dtype=np.complex64, mode='r', shape=(inps.n_image, inps.lin, inps.sam))

    if not os.path.isfile(inps.patch_dir + '/SHP.pkl'):

        if inps.n_image < 20:
            num_slc = inps.n_image
        else:
            num_slc = 20

        time0 = time.time()
        lin = np.ogrid[0:inps.lin]
        sam = np.ogrid[0:inps.sam]
        lin, sam = np.meshgrid(lin, sam)
        coords = list(map(lambda x, y: [int(x), int(y)],
                     lin.T.reshape(inps.lin*inps.sam, 1), sam.T.reshape(inps.lin*inps.sam, 1)))
        del lin, sam

        shp_df = pd.DataFrame({'ref_pixel': coords})
        shp_df.insert(1, 'pixeltype', 'default')
        shp_df.insert(2, 'rows', 'default')
        shp_df.insert(3, 'cols', 'default')

        for item in range(len(shp_df)):

            row_0 = shp_df.at[item,'ref_pixel'][0]
            col_0 = shp_df.at[item,'ref_pixel'][1]

            r = np.ogrid[row_0 - ((inps.azimuth_win - 1) / 2):row_0 + ((inps.azimuth_win - 1) / 2) + 1]
            refr = np.array([(inps.azimuth_win - 1) / 2])
            r = r[r >= 0]
            r = r[r < inps.lin]
            refr = refr - (inps.azimuth_win - len(r))
            c = np.ogrid[col_0 - ((inps.range_win - 1) / 2):col_0 + ((inps.range_win - 1) / 2) + 1]
            refc = np.array([(inps.range_win - 1) / 2])
            c = c[c >= 0]
            c = c[c < inps.sam]
            refc = refc - (inps.range_win - len(c))

            x, y = np.meshgrid(r.astype(int), c.astype(int), sparse=True)
            win = np.abs(rslc[0:num_slc, x, y])
            win = pysq.trwin(win)
            testvec = win.reshape(num_slc, len(r) * len(c))
            ksres = np.ones(len(r) * len(c))
            S1 = np.abs(rslc[0:num_slc, row_0, col_0])
            S1 = S1.reshape(num_slc, 1)
            for m in range(len(testvec[0])):
                S2 = testvec[:, m]
                S2 = S2.reshape(num_slc, 1)

                try:
                    test = anderson_ksamp([S1, S2])
                    if test.significance_level > 0.05:
                        ksres[m] = 1
                    else:
                        ksres[m] = 0
                except:
                    ksres[m] = 0

            ks_res = ksres.reshape(len(r), len(c))
            ks_label = label(ks_res, background=False, connectivity=2)
            reflabel = ks_label[refr.astype(int), refc.astype(int)]
            rr, cc = np.where(ks_label == reflabel)
            rr = rr + r[0]
            cc = cc + c[0]
            shp_df.at[item,'rows'] = rr
            shp_df.at[item,'cols'] = cc
            if len(rr) > 20:
                shp_df.at[item,'pixeltype'] = 'DS'
            else:
                shp_df.at[item,'pixeltype'] = 'Unknown'


        shp_df.to_pickle(inps.patch_dir + '/SHP.pkl')

        timep = time.time() - time0

        print('time spent to find SHPs {}: min'.format(timep / 60))

    else:

        shp_df = pd.read_pickle(inps.patch_dir + '/SHP.pkl')

        print('SHP Exists ...')

    ###################### Phase linking inversion ###############################


    if 'sequential' in inps.phase_linking_method:

        time0 = time.time()

        #shp_df = shp_df_0.loc[shp_df['pixeltype']=='DS']

        num_seq = np.int(np.floor(inps.n_image / 10))

        if os.path.isfile(inps.patch_dir + '/num_processed.npy'):
            num_image_processed = np.load(inps.patch_dir + '/num_processed.npy')[0]
            if num_image_processed == inps.n_image:
                doprocess = False
            else:
                doprocess = True
        else:
            num_image_processed = 0
            doprocess = True

        if doprocess:

            if os.path.isfile(inps.patch_dir + '/RSLC_ref'):
                rslc_ref = np.memmap(inps.patch_dir + '/RSLC_ref', dtype='complex64', mode='r+',
                                     shape=(inps.n_image, inps.lin, inps.sam))

            else:
                rslc_ref = np.memmap(inps.patch_dir + '/RSLC_ref', dtype='complex64', mode='w+',
                                     shape=(inps.n_image, inps.lin, inps.sam))
                rslc_ref[num_image_processed::,:,:] = rslc[num_image_processed::,:,:]

            datumshift = np.zeros([num_seq, rslc.shape[1], rslc.shape[2]])

            if os.path.isfile(inps.patch_dir + '/datum_shift.npy'):
                datumshift_old = np.load(inps.patch_dir + '/datum_shift.npy')
                step_0 = datumshift.shape[0] - 1
            else:
                datumshift_old = np.zeros([num_seq,rslc.shape[1],rslc.shape[2]])
                step_0 = 0


            if os.path.isfile(inps.patch_dir + '/quality'):
                quality = np.memmap(inps.patch_dir + '/quality', dtype='float32', mode='r+',shape=(inps.lin, inps.sam))
            else:
                quality = np.memmap(inps.patch_dir + '/quality', dtype='float32', mode='w+',shape=(inps.lin, inps.sam))
                quality[:,:] = -1

            for item in range(len(shp_df)):


                ref_row, ref_col = (shp_df.at[item,'ref_pixel'][0], shp_df.at[item,'ref_pixel'][1])
                rr = shp_df.at[item,'rows'].astype(int)
                cc = shp_df.at[item,'cols'].astype(int)

                CCG = np.matrix(1.0 * np.arange(inps.n_image * len(rr)).reshape(inps.n_image, len(rr)))
                CCG = np.exp(1j * CCG)
                CCG[:, :] = np.matrix(rslc[:, rr, cc])
                phase_ref = np.zeros([inps.n_image,1])
                phase_ref[:,:] = np.angle(rslc_ref[:,ref_row,ref_col]).reshape(inps.n_image,1)

                if shp_df.at[item,'pixeltype']=='Unknown':
                    amp_ps = np.abs(rslc_ref[:,ref_row,ref_col]).reshape(inps.n_image,1)
                    DA = np.std(amp_ps)/np.mean(amp_ps)
                    if DA < 0.25:
                        Laq = 1
                else:

                    if not step_0 == 0:
                        squeezed_pixels = np.complex64(np.zeros([step_0, len(rr)]))
                        for seq in range(0, step_0):
                            squeezed_pixels[seq, :] = pysq.squeez_im(phase_ref[seq * 10:seq * 10 + 10, 0],
                                                                CCG[seq * 10:seq * 10 + 10, :])
                    Laq = quality[ref_row,ref_col]

                    if num_seq == 0 or num_seq == 1:
                        first_line = 0
                        last_line = inps.n_image
                        num_lines = last_line - first_line
                        ccg_sample = CCG[first_line:last_line, :]
                        res, La, squeezed_pixels = pysq.phase_linking_process(ccg_sample, 0, inps.phase_linking_method)
                        phase_ref[first_line:last_line, 0:1] = res[0::].reshape(num_lines, 1)
                        Laq = np.max([La[0], Laq])

                    else:

                        for stepp in range(step_0, num_seq):

                            first_line = stepp * 10
                            if stepp == num_seq - 1:
                                last_line = inps.n_image
                            else:
                                last_line = first_line + 10
                            num_lines = last_line - first_line

                            if stepp == 0:

                                ccg_sample = CCG[first_line:last_line, :]
                                res,La, squeezed_pixels = pysq.phase_linking_process(ccg_sample, 0, inps.phase_linking_method)
                                phase_ref[first_line:last_line, 0:1] = res[stepp::].reshape(num_lines, 1)

                            else:

                                ccg_sample = np.zeros([1 + num_lines, CCG.shape[1]]) + 1j
                                ccg_sample[0:1, :] = np.complex64(squeezed_pixels[-1, :])
                                ccg_sample[1::, :] = CCG[first_line:last_line, :]
                                res,La, squeezed_p = pysq.phase_linking_process(ccg_sample, 1, inps.phase_linking_method)
                                phase_ref[first_line:last_line, 0:1] = res[1::].reshape(num_lines, 1)
                                squeezed_pixels = np.complex64(np.vstack([squeezed_pixels, squeezed_p]))
                            Laq = np.max([La[0],Laq])
                        res_d,Lad = pysq.phase_linking_process(squeezed_pixels, 0, 'EMI', squeez=False)
                        res_d = np.unwrap(res_d, np.pi, axis=0)

                        for stepp in range(step_0, len(res_d)):
                            first_line = stepp * 10
                            if stepp == num_seq - 1:
                                last_line = inps.n_image
                            else:
                                last_line = first_line + 10
                            num_lines = last_line - first_line

                            phase_ref[first_line:last_line, 0:1] = (
                                        phase_ref[first_line:last_line, 0:1] + np.matrix(res_d[int(stepp)]) - datumshift_old[
                                    int(stepp),ref_row,ref_col]).reshape(num_lines, 1)

                amp_ref = np.array(np.mean(np.abs(CCG), axis=1))
                ph_ref = np.array(phase_ref)



                rslc_ref[:,ref_row:ref_row+1,ref_col:ref_col+1] = np.complex64(np.multiply(amp_ref, np.exp(1j * ph_ref))).reshape(inps.n_image,1,1)


                quality[ref_row:ref_row+1, ref_col:ref_col+1] = Laq

                if num_seq > 1:
                    datumshift[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = res_d.reshape(num_seq, 1, 1)


            np.save(inps.patch_dir + '/num_processed.npy', [inps.n_image,inps.lin,inps.sam])
            np.save(inps.patch_dir + '/datum_shift.npy', datumshift)

            del rslc_ref, rslc, quality

            timep = time.time() - time0
            print('time spent to do sequential phase linking {}: min'.format(timep/60))


    else:
        time0 = time.time()

        #shp_df = shp_df_0.loc[shp_df['pixeltype'] == 'DS']

        rslc_ref = np.memmap(inps.patch_dir + '/RSLC_ref', dtype='complex64', mode='w+',
                             shape=(inps.n_image, inps.lin, inps.sam))
        rslc_ref[:, :, :] = rslc[:, :, :]

        quality = np.memmap(inps.patch_dir + '/quality', dtype='float32', mode='w+', shape=(inps.lin, inps.sam))
        quality[:,:] = -1

        for item in range(len(shp_df)):
            ref_row, ref_col = (shp_df.at[item, 'ref_pixel'][0], shp_df.at[item, 'ref_pixel'][1])
            rr = shp_df.at[item, 'rows'].astype(int)
            cc = shp_df.at[item, 'cols'].astype(int)
            CCG = np.matrix(1.0 * np.arange(inps.n_image * len(rr)).reshape(inps.n_image, len(rr)))
            CCG = np.exp(1j * CCG)
            CCG[:, :] = np.matrix(rslc[:, rr, cc])
            phase_ref = np.zeros([inps.n_image, 1])
            phase_ref[:, :] = np.angle(rslc_ref[:, ref_row, ref_col]).reshape(inps.n_image, 1)

            if shp_df.at[item, 'pixeltype'] == 'Unknown':
                amp_ps = np.abs(rslc_ref[:, ref_row, ref_col]).reshape(inps.n_image, 1)
                DA = np.std(amp_ps) / np.mean(amp_ps)
                if DA < 0.25:
                    Laq = 1
            else:

                res, Laq = pysq.phase_linking_process(CCG, 0, inps.phase_linking_method, squeez=False)
                phase_ref[:, 0:1] = res.reshape(inps.n_image, 1)

            quality[ref_row:ref_row+1,ref_col:ref_col+1] = Laq


            amp_ref = np.array(np.mean(np.abs(CCG), axis=1))
            ph_ref = np.array(phase_ref)


            rslc_ref[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = np.complex64(
                np.multiply(amp_ref, np.exp(1j * ph_ref))).reshape(inps.n_image, 1, 1)

        del rslc_ref, rslc, quality

        timep = time.time() - time0
        print('time spent to do phase linking {}: min'.format(timep / 60))

if __name__ == '__main__':
    '''
    Phase linking process.
    '''
    main()

#################################################
