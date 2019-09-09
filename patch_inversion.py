#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import minopy_utilities as mnp
import dask
from scipy.stats import ks_2samp, anderson_ksamp, ttest_ind
from skimage.measure import label
from minsar.objects.auto_defaults import PathFind

pathObj = PathFind()
#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    inps = mnp.cmd_line_parse(iargs, script='patch_inversion')

    if not inps.patch:
        raise Exception('No patch specified')

    inversionObj = PhaseLink(inps)

    # Find SHPs:

    inversionObj.find_shp()

    # Phase linking inversion:

    inversionObj.patch_phase_linking()

    # Quality (temporal coherence) calculation based on PTA

    inversionObj.get_pta_coherence()

    print('{} is done successfuly'.format(inps.patch))

    return None


class PhaseLink:
    def __init__(self, inps):

        self.minopydir = os.path.join(inps.work_dir, pathObj.minopydir)
        self.patch_rows = np.load(os.path.join(self.minopydir, 'rowpatch.npy'))
        self.patch_cols = np.load(os.path.join(self.minopydir, 'colpatch.npy'))
        self.phase_linking_method = inps.template['minopy.plmethod']
        self.range_window = int(inps.template['minopy.range_window'])
        self.azimuth_window = int(inps.template['minopy.azimuth_window'])
        self.patch_dir = inps.patch

        count_dim = np.load(inps.patch + '/count.npy')
        self.n_image = count_dim[0]
        self.length = count_dim[1]
        self.width = count_dim[2]

        if self.n_image < 20:
            self.num_slc = self.n_image
        else:
            self.num_slc = 20

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

        self.rslc = np.memmap(self.patch_dir + '/RSLC', dtype=np.complex64, mode='r',
                                 shape=(self.n_image, self.length, self.width))

        shp_size = self.range_window * self.azimuth_window
        if not os.path.isfile(self.patch_dir + '/SHP'):
            self.shp = np.memmap(self.patch_dir + '/SHP', dtype='byte', mode='write',
                                 shape=(shp_size, self.length, self.width))
        else:
            self.shp = np.memmap(self.patch_dir + '/SHP', dtype='byte', mode='r+',
                                 shape=(shp_size, self.length, self.width))

        self.rslc_ref = None
        self.progress = None
        self.quality = None

        return

    def find_shp(self):

        time0 = time.time()
        
        print('Find SHP for {}'.format(self.patch_dir))

        if not os.path.isfile(self.patch_dir + '/shp_flag'):
            for coord in self.coords:
                if not self.shp[:, coord[0], coord[1]].any():
                    self.get_shp_row_col(coord)
                else:
                    print('coord {} done before'.format(coord))

        with open(self.patch_dir + '/shp_flag', 'w') as f:
            f.write('shp step done')

        timep = time.time() - time0

        print('time spent to find SHPs {}: min'.format(timep / 60))

        return

    def get_shp_row_col(self, data):

        print(data)

        row_0, col_0 = data

        sample_rows = row_0 + self.sample_rows
        sample_rows[sample_rows < 0] = -1
        sample_rows[sample_rows >= self.length] = -1

        sample_cols = col_0 + self.sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= self.width] = -1

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        win = np.abs(self.rslc[0:self.num_slc, y, x])
        testvec = win.reshape(self.num_slc, self.azimuth_window * self.range_window)
        ksres = np.zeros(self.azimuth_window * self.range_window).astype(int)

        S1 = np.abs(self.rslc[0:self.num_slc, row_0, col_0])
        S1 = S1.flatten()

        x = x.flatten()
        y = y.flatten()

        for m in range(testvec.shape[1]):
            if x[m] >= 0 and y[m] >= 0:
                S2 = testvec[:, m]
                S2 = S2.flatten()

                try:
                    # test = anderson_ksamp([S1, S2])
                    test = ks_2samp(S1, S2)
                    # test = ttest_ind(S1, S2, equal_var=False)
                    if test[1] > 0.95:
                        ksres[m] = 1
                    # if test.significance_level > 0.01:
                    #    adres[m] = 1
                    # if test[1] > 0.05:
                    #    adres[m] = 1
                except:
                    ksres[m] = 0
                    #adres[m] = 0

        ks_label = label(ksres.reshape(self.azimuth_window, self.range_window), background=False, connectivity=2)
        ksres = 1 * (ks_label == ks_label[self.reference_row, self.reference_col])

        self.shp[:, row_0:row_0 + 1, col_0:col_0 + 1] = ksres.reshape(self.azimuth_window * self.range_window, 1, 1)

        return

    def inversion_sequential(self, data):

        ref_row, ref_col, shp_rows, shp_cols, phase_linking_method = data

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

        phase_refined = mnp.sequential_phase_linking(CCG, phase_linking_method, num_stack=1)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        self.rslc_ref[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = \
            np.complex64(np.multiply(amp_refined, np.exp(1j * phase_refined))).reshape(self.n_image, 1, 1)

        self.progress[ref_row:ref_row + 1, ref_col:ref_col + 1] = 1

        # print(ref_row, ref_col, 'DS')

        return None

    def inversion_all(self, data):
        ref_row, ref_col, shp_rows, shp_cols, phase_linking_method = data

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

        phase_refined = mnp.phase_linking_process(CCG, 1, phase_linking_method, squeez=False)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1))
        phase_refined = np.array(phase_refined)

        self.rslc_ref[:, ref_row:ref_row + 1, ref_col:ref_col + 1] = \
            np.complex64(np.multiply(amp_refined, np.exp(1j * phase_refined))).reshape(self.n_image, 1, 1)

        self.progress[ref_row:ref_row + 1, ref_col:ref_col + 1] = 1

        # print(ref_row, ref_col, 'DS')

        return None

    def filter_persistent_scatterer(self, data):
        ref_row, ref_col, shp_rows, shp_cols = data

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

        status = mnp.test_PS(CCG)

        if status:
            data_invert = (ref_row, ref_col, shp_rows, shp_cols, 'EMI')
            self.inversion_all(data_invert)
            # print(ref_row, ref_col, 'PS')

        return None

    def patch_phase_linking(self):

        print('Inversion for {}'.format(self.patch_dir))

        if not os.path.exists(self.patch_dir + '/RSLC_ref'):

            self.rslc_ref = np.memmap(self.patch_dir + '/RSLC_ref', dtype='complex64', mode='w+',
                                         shape=(self.n_image, self.length, self.width))

            self.progress = np.memmap(self.patch_dir + '/progress', dtype='int8', mode='w+',
                                         shape=(self.length, self.width))

            self.rslc_ref[:, :, :] = self.rslc[:, :, :]

        else:
            self.rslc_ref = np.memmap(self.patch_dir + '/RSLC_ref', dtype='complex64', mode='r+',
                                         shape=(self.n_image, self.length, self.width))

            self.progress = np.memmap(self.patch_dir + '/progress', dtype='int8', mode='r+',
                                         shape=(self.length, self.width))

        # futures = []

        time0 = time.time()

        if not os.path.isfile('inversion_flag'):

            for coord in self.coords:

                print(coord)

                num_shp = len(self.shp[self.shp[:, coord[0], coord[1]] == 1])
                if num_shp > 0:

                    ad_label = self.shp[:, coord[0], coord[1]].reshape(self.azimuth_window, self.range_window)
                    shp_rows, shp_cols = np.where(ad_label == 1)
                    shp_rows = np.array(shp_rows + coord[0] - (self.azimuth_window - 1) / 2).astype(int)
                    shp_cols = np.array(shp_cols + coord[1] - (self.range_window - 1) / 2).astype(int)

                    if num_shp < 20:
                        data = (coord[0], coord[1], shp_rows, shp_cols)
                        self.filter_persistent_scatterer(data)
                        # futures.append(dask.delayed(self.filter_persistent_scatterer)(data))
                    else:
                        data = (coord[0], coord[1], shp_rows, shp_cols, self.phase_linking_method)

                        if 'sequential' in self.phase_linking_method:
                            self.inversion_sequential(data)
                            # futures.append(dask.delayed(self.inversion_sequential)(data))
                        else:
                            self.inversion_all(data)
                            # futures.append(dask.delayed(self.inversion_all)(data))

        # results = dask.compute(*futures)

        timep = time.time() - time0
        print('time spent to do phase linking {}: min'.format(timep / 60))

        with open(self.patch_dir + '/inversion_flag', 'w') as f:
            f.write('Inversion done')

        return

    def get_pta_coherence(self):

        if not os.path.exists(self.patch_dir + '/quality'):

            self.quality = np.memmap(self.patch_dir + '/quality', dtype='float32', mode='w+',
                                        shape=(self.length, self.width))
            self.quality[:, :] = -1

            if self.rslc_ref is None:
                self.rslc_ref = np.memmap(self.patch_dir + '/RSLC_ref', dtype='complex64', mode='r',
                                          shape=(self.n_image, self.length, self.width))

            futures = []

            for coord in self.coords:

                if self.progress[coord[0], coord[1]] == 1:

                    phase_initial = np.angle(self.rslc[:, coord[0], coord[1]]).reshape(self.n_image, 1)
                    phase_refined = np.angle(self.rslc_ref[:, coord[0], coord[1]]).reshape(self.n_image, 1)
                    data = (phase_initial, phase_refined, coord[0], coord[1])

                    future = dask.delayed(mnp.gam_pta)(data)
                    futures.append(future)

            results = dask.compute(*futures)

            for result in results:
                gam, row, col = result
                self.quality[row:row + 1, col:col + 1] = gam

        else:
            print('Done with the inversion of {}'.format(self.patch_dir))

        del self.rslc_ref, self.rslc, self.quality, self.progress, self.shp

        return


if __name__ == '__main__':

    main()

#################################################
