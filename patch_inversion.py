#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import minopy_utilities as mnp
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

    inversionObj.iterate_coords()

    print('{} is done successfuly'.format(inps.patch_dir))

    return None


class PhaseLink:
    def __init__(self, inps):
        self.work_dir = inps.work_dir
        self.phase_linking_method = inps.inversion_method
        self.range_window = inps.range_window
        self.azimuth_window = inps.azimuth_window
        self.patch_dir = os.path.join(inps.work_dir, 'patches', inps.patch_dir)
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

        self.distance_thresh = mnp.ks_lut(self.n_image, self.n_image, alpha=0.01)

        success = False
        while success is False:
            try:
                patch_rows = np.load(inps.work_dir + '/patches/rowpatch.npy')
                patch_cols = np.load(inps.work_dir + '/patches/colpatch.npy')
                success = True
            except:
                success = False

        row = int(self.patch_dir.split('patch')[-1].split('_')[0])
        col = int(self.patch_dir.split('patch')[-1].split('_')[1])

        patch_rows_overlap = np.zeros(np.shape(patch_rows), dtype=int)
        patch_rows_overlap[:, :, :] = patch_rows[:, :, :]
        patch_rows_overlap[1, 0, 0] = patch_rows_overlap[1, 0, 0] - self.azimuth_window + 1
        patch_rows_overlap[0, 0, 1::] = patch_rows_overlap[0, 0, 1::] + self.azimuth_window - 1
        patch_rows_overlap[1, 0, 1::] = patch_rows_overlap[1, 0, 1::] - self.azimuth_window + 1
        patch_rows_overlap[1, 0, -1] = patch_rows_overlap[1, 0, -1] + self.azimuth_window - 1

        patch_cols_overlap = np.zeros(np.shape(patch_cols), dtype=int)
        patch_cols_overlap[:, :, :] = patch_cols[:, :, :]
        patch_cols_overlap[1, 0, 0] = patch_cols_overlap[1, 0, 0] - self.range_window + 1
        patch_cols_overlap[0, 0, 1::] = patch_cols_overlap[0, 0, 1::] + self.range_window - 1
        patch_cols_overlap[1, 0, 1::] = patch_cols_overlap[1, 0, 1::] - self.range_window + 1
        patch_cols_overlap[1, 0, -1] = patch_cols_overlap[1, 0, -1] + self.range_window - 1

        row1 = patch_rows_overlap[0, 0, row] - patch_rows[0, 0, row]
        row2 = patch_rows_overlap[1, 0, row] - patch_rows[0, 0, row]
        col1 = patch_cols_overlap[0, 0, col] - patch_cols[0, 0, col]
        col2 = patch_cols_overlap[1, 0, col] - patch_cols[0, 0, col]

        lin = np.ogrid[row1:row2]
        overlap_length = len(lin)
        sam = np.ogrid[col1:col2]
        overlap_width = len(sam)
        lin, sam = np.meshgrid(lin, sam)

        self.coords = list(map(lambda y, x: (int(y), int(x)),
                          lin.T.reshape(overlap_length * overlap_width, 1),
                          sam.T.reshape(overlap_length * overlap_width, 1)))
        self.coords = np.transpose(np.array(self.coords))

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
        mask = 1 * (x >= 0) * (y >= 0)
        indx = np.where(mask == 1)
        x = x[indx[0], indx[1]]
        y = y[indx[0], indx[1]]

        testvec = np.sort(np.abs(self.rslc[:, y, x]), axis=0)
        S1 = np.sort(np.abs(self.rslc[:, row_0, col_0])).reshape(self.n_image, 1)

        data1 = np.repeat(S1, testvec.shape[1], axis=1)
        data_all = np.concatenate((data1, testvec), axis=0)

        res = np.zeros([self.azimuth_window, self.range_window])
        res[indx[0], indx[1]] = 1 * (np.apply_along_axis(mnp.ecdf_distance, 0, data_all) <= self.distance_thresh)
        ks_label = label(res, background=0, connectivity=2)
        ksres = 1 * (ks_label == ks_label[self.reference_row, self.reference_col]) * mask

        self.shp[:, row_0:row_0 + 1, col_0:col_0 + 1] = ksres.reshape(self.azimuth_window * self.range_window, 1, 1)

        return ksres

    def phase_inversion(self, coord):

        if not self.shp[:, coord[0], coord[1]].any():
            shp = self.get_shp_row_col(coord)
        else:
            shp = self.shp[:, coord[0], coord[1]].reshape(self.azimuth_window, self.range_window)

        if self.quality[coord[0], coord[1]] == -1:

            num_shp = len(shp[shp > 0])

            shp_rows, shp_cols = np.where(shp == 1)
            shp_rows = np.array(shp_rows + coord[0] - (self.azimuth_window - 1) / 2).astype(int)
            shp_cols = np.array(shp_cols + coord[1] - (self.range_window - 1) / 2).astype(int)

            CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
            CCG = np.exp(1j * CCG)
            CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

            coh_mat = mnp.est_corr(CCG)

            if num_shp > 20:

                if 'sequential' in self.phase_linking_method:
                    vec_refined = mnp.sequential_phase_linking(CCG, self.phase_linking_method, num_stack=100)
                else:
                    vec_refined = mnp.phase_linking_process(coh_mat, 0, self.phase_linking_method, squeez=False)
            else:
                vec_refined = mnp.test_PS(coh_mat)

            self.quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mnp.gam_pta(np.angle(coh_mat), vec_refined)
            phase_refined = np.angle(np.array(vec_refined)).reshape(self.n_image, 1, 1)
            amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(self.n_image, 1, 1)

            self.rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = \
                np.multiply(amp_refined, np.exp(1j * phase_refined))

        return

    def iterate_coords(self):

        print('Inversion for {}'.format(self.patch_dir))

        time0 = time.time()

        np.apply_along_axis(self.phase_inversion, 0, self.coords)

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))

        return


if __name__ == '__main__':

    main()

#################################################
