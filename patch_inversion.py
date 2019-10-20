#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import minopy_utilities as mnp
import dask
import pandas as pd
from skimage.measure import label
from minopy.objects.arg_parser import MinoPyParser

#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='patch_inversion')
    inps = Parser.parse()

    if not inps.patch_dir:
        raise Exception('No patch specified')

    inversionObj = PhaseLink(inps)

    # Phase linking inversion:

    inversionObj.patch_phase_linking()

    print('{} is done successfuly'.format(inps.patch_dir))

    return None


class PhaseLink:
    def __init__(self, inps):
        self.work_dir = inps.work_dir
        self.patch_rows = np.load(os.path.join(self.work_dir, 'patches/rowpatch.npy'))
        self.patch_cols = np.load(os.path.join(self.work_dir, 'patches/colpatch.npy'))
        self.phase_linking_method = inps.inversion_method
        self.range_window = inps.range_window
        self.azimuth_window = inps.azimuth_window
        self.patch_dir = os.path.join(inps.work_dir,'patches', inps.patch_dir)
        self.shp_test = inps.shp_test
        if self.shp_test == 'ks':
            self.shp_function = mnp.ks2smapletest
        elif self.shp_test == 'ad':
            self.shp_function = mnp.ADtest
        elif self.shp_test == 'ttest':
            self.shp_function = mnp.ttest_indtest
        else:   # default is KS 2 sample test
            self.shp_function = mnp.ks2smapletest

        count_dim = np.load(self.patch_dir + '/count.npy')
        self.n_image = count_dim[0]
        self.length = count_dim[1]
        self.width = count_dim[2]

        if self.n_image < 20:
            self.num_slc = self.n_image
        else:
            self.num_slc = 20

        self.distance_thresh = mnp.ks_lut(self.num_slc, self.num_slc, alpha=0.05)

        lin = np.ogrid[0:self.length]
        sam = np.ogrid[0:self.width]
        lin, sam = np.meshgrid(lin, sam)
        self.coords = list(map(lambda y, x: [int(y), int(x)],
                          lin.T.reshape(self.length * self.width, 1),
                          sam.T.reshape(self.length * self.width, 1)))

        self.sample_rows = np.ogrid[-((self.azimuth_window - 1) / 2):((self.azimuth_window - 1) / 2) + 1]
        self.sample_rows = self.sample_rows.astype(int)
        self.reference_row = np.array([(self.azimuth_window - 1) / 2]).astype(int)
        self.reference_row = self.reference_row - (self.azimuth_window - len(self.sample_rows))

        self.sample_cols = np.ogrid[-((self.range_window - 1) / 2):((self.range_window - 1) / 2) + 1]
        self.sample_cols = self.sample_cols.astype(int)
        self.reference_col = np.array([(self.range_window - 1) / 2]).astype(int)
        self.reference_col = self.reference_col - (self.range_window - len(self.sample_cols))

        self.rslc = np.memmap(self.patch_dir + '/rslc', dtype=np.complex64, mode='r',
                                 shape=(self.n_image, self.length, self.width))

        shp_size = self.range_window * self.azimuth_window
        if not os.path.isfile(self.patch_dir + '/shp'):
            self.shp = np.memmap(self.patch_dir + '/shp', dtype='byte', mode='write',
                                 shape=(shp_size, self.length, self.width))
        else:
            self.shp = np.memmap(self.patch_dir + '/shp', dtype='byte', mode='r+',
                                 shape=(shp_size, self.length, self.width))

        if not os.path.exists(self.patch_dir + '/rslc_ref'):

            self.rslc_ref = np.memmap(self.patch_dir + '/rslc_ref', dtype='complex64', mode='w+',
                                         shape=(self.n_image, self.length, self.width))

            self.rslc_ref[:, :, :] = self.rslc[:, :, :]

        else:
            self.rslc_ref = np.memmap(self.patch_dir + '/rslc_ref', dtype='complex64', mode='r+',
                                         shape=(self.n_image, self.length, self.width))

        if not os.path.exists(self.patch_dir + '/quality'):

            self.quality = np.memmap(self.patch_dir + '/quality', dtype='float32', mode='w+',
                                        shape=(self.length, self.width))
            self.quality[:, :] = -1
        else:
            self.quality = np.memmap(self.patch_dir + '/quality', dtype='float32', mode='r+',
                                     shape=(self.length, self.width))

        return

    def get_shp_row_col(self, data):

        row_0, col_0 = data

        sample_rows = row_0 + self.sample_rows
        sample_rows[sample_rows < 0] = -1
        sample_rows[sample_rows >= self.length] = -1

        sample_cols = col_0 + self.sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= self.width] = -1

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        win = np.abs(self.rslc[0:self.num_slc, y, x])
        testvec = np.sort(win.reshape(self.num_slc, self.azimuth_window * self.range_window), axis=0)
        ksres = np.zeros(self.azimuth_window * self.range_window).astype(int)

        S1 = np.abs(self.rslc[0:self.num_slc, row_0, col_0]).reshape(self.num_slc, 1)
        S1 = np.sort(S1.flatten())

        x = x.flatten()
        y = y.flatten()

        for m in range(testvec.shape[1]):
            if x[m] >= 0 and y[m] >= 0:
                S2 = testvec[:, m]
                S2 = np.sort(S2.flatten())
                ksres[m] = self.shp_function(S1, S2, threshold=self.distance_thresh)

        ks_label = label(ksres.reshape(self.azimuth_window, self.range_window), background=False, connectivity=2)
        ksres = 1 * (ks_label == ks_label[self.reference_row, self.reference_col])

        self.shp[:, row_0:row_0 + 1, col_0:col_0 + 1] = ksres.reshape(self.azimuth_window * self.range_window, 1, 1)

        return ksres

    def inversion_sequential(self, CCG, phase_linking_method):

        phase_refined = mnp.sequential_phase_linking(CCG, phase_linking_method, num_stack=1)
        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        return amp_refined, phase_refined

    def inversion_all(self, CCG, phase_linking_method):

        phase_refined = mnp.phase_linking_process(CCG, 1, phase_linking_method, squeez=False)
        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        return amp_refined, phase_refined

    def patch_phase_linking(self):

        print('Inversion for {}'.format(self.patch_dir))

        time0 = time.time()

        if os.path.exists(self.patch_dir + '/inversion_flag'):
            return print('Inversion is already done for {}'.format(self.patch_dir))

        for coord in self.coords:
            if not self.shp[:, coord[0], coord[1]].any():
                shp = self.get_shp_row_col(coord)
                num_shp = len(shp[np.nonzero(shp > 0)])
                if num_shp > 0:

                    shp_rows, shp_cols = np.where(shp == 1)
                    shp_rows = np.array(shp_rows + coord[0] - (self.azimuth_window - 1) / 2).astype(int)
                    shp_cols = np.array(shp_cols + coord[1] - (self.range_window - 1) / 2).astype(int)

                    CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
                    CCG = np.exp(1j * CCG)
                    CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

                    if num_shp > 20:
                        if 'sequential' in self.phase_linking_method:
                            amp_refined, phase_refined = self.inversion_sequential(CCG, self.phase_linking_method)
                        else:
                            amp_refined, phase_refined = self.inversion_all(CCG, self.phase_linking_method)
                    else:
                        status = mnp.test_PS(CCG)
                        if status:
                            amp_refined, phase_refined = self.inversion_all(CCG, 'EMI')

                    self.rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = \
                        np.complex64(np.multiply(amp_refined, np.exp(1j * phase_refined))).reshape(self.n_image, 1, 1)

                    ph_filt = np.angle(mnp.est_corr(CCG))
                    self.quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mnp.gam_pta(ph_filt, phase_refined)

        timep = time.time() - time0
        print('time spent to do phase linking {}: min'.format(timep / 60))

        with open(self.patch_dir + '/inversion_flag', 'w') as f:
            f.write('Inversion done')

        return


if __name__ == '__main__':

    main()

#################################################
