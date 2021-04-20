#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time

import h5py
import minopy_utilities as mnp
import numpy as np
from isceobj.Util.ImageUtil import ImageLib as IML
from minopy.objects import cluster_minopy
from minopy.objects.arg_parser import MinoPyParser
from mintpy.utils import ptime
from skimage.measure import label


#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    # --cluster and --num-worker option
    inps.numWorker = str(cluster_minopy.DaskCluster.format_num_worker(inps.cluster, inps.numWorker))
    if inps.cluster != 'no' and inps.numWorker == '1':
        print('WARNING: number of workers is 1, turn OFF parallel processing and continue')
        inps.cluster = 'no'

    inversionObj = PhaseLink(inps)

    # Phase linking inversion:

    inversionObj.iterate_coords()

    return None


class PhaseLink:
    def __init__(self, inps):
        self.work_dir = inps.work_dir
        self.phase_linking_method = inps.inversion_method
        self.range_window = inps.range_window
        self.azimuth_window = inps.azimuth_window
        self.patch_size = inps.patch_size
        self.cluster = inps.cluster
        self.numWorker = inps.numWorker
        self.config = inps.config
        self.out_dir = self.work_dir + '/inverted'
        os.makedirs(self.out_dir, exist_ok='True')

        self.shp_test = inps.shp_test
        if self.shp_test == 'ks':
            self.shp_function = mnp.ks2smapletest
        elif self.shp_test == 'ad':
            self.shp_function = mnp.ADtest
        elif self.shp_test == 'ttest':
            self.shp_function = mnp.ttest_indtest
        else:  # default is KS 2 sample test
            self.shp_function = mnp.ks2smapletest

        # read input slcStack.h5
        self.slc_stack = inps.slc_stack
        self.slcObj = h5py.File(inps.slc_stack, 'r')
        self.n_image, self.length, self.width = self.slcObj['slc'].shape
        self.slcObj.close()
        print('Total number of pixels {}'.format(self.length * self.width))
        self.shp_size = self.range_window * self.azimuth_window

        self.distance_thresh = mnp.ks_lut(self.n_image, self.n_image, alpha=0.01)

        if not os.path.exists(self.out_dir + '/rslc_ref'):

            self.rslc_ref = np.memmap(self.out_dir + '/rslc_ref', dtype='complex64', mode='w+',
                                      shape=(self.n_image, self.length, self.width))

        else:
            self.rslc_ref = np.memmap(self.out_dir + '/rslc_ref', dtype='complex64', mode='r+',
                                      shape=(self.n_image, self.length, self.width))

        if not os.path.isfile(self.out_dir + '/shp'):
            self.shp = np.memmap(self.out_dir + '/shp', dtype='byte', mode='write',
                                 shape=(self.shp_size, self.length, self.width))
        else:
            self.shp = np.memmap(self.out_dir + '/shp', dtype='byte', mode='r+',
                                 shape=(self.shp_size, self.length, self.width))

        if not os.path.exists(self.out_dir + '/quality'):

            self.quality = np.memmap(self.out_dir + '/quality', dtype='float32', mode='w+',
                                     shape=(self.length, self.width))
            self.quality[:, :] = -1
        else:
            self.quality = np.memmap(self.out_dir + '/quality', dtype='float32', mode='r+',
                                     shape=(self.length, self.width))

        self.box_list, self.num_box = self.patch_slice()

        IML.renderISCEXML(self.out_dir + '/rslc_ref', bands=self.n_image, nyy=self.length, nxx=self.width,
                          datatype='complex64', scheme='BSQ')
        IML.renderISCEXML(self.out_dir + '/quality', bands=1, nyy=self.length, nxx=self.width,
                          datatype='float32', scheme='BSQ')
        IML.renderISCEXML(self.out_dir + '/shp', bands=self.shp_size, nyy=self.length, nxx=self.width,
                          datatype='byte', scheme='BSQ')
        return

    def iterate_coords(self):

        time0 = time.time()

        data_kwargs = {
            "slc_file": self.slc_stack,
            "shp_file": self.out_dir + '/shp',
            "quality_file": self.out_dir + '/quality',
            "distance_thresh": self.distance_thresh,
            "azimuth_window": self.azimuth_window,
            "range_window": self.range_window,
            "phase_linking_method": self.phase_linking_method,
        }

        # invert / write block-by-block
        for i, box in enumerate(self.box_list):
            box_width = box[2] - box[0]
            box_length = box[3] - box[1]
            if self.num_box > 1:
                print('\n------- processing patch {} out of {} --------------'.format(i + 1, self.num_box))
                print('box width:  {}'.format(box_width))
                print('box length: {}'.format(box_length))

            data_kwargs['box'] = box
            # self.cluster = 'no'
            if self.cluster == 'no':
                rslc_ref, SHP, quality, bbox = inversion(**data_kwargs)
            else:
                # parallel
                print('\n\n------- start parallel processing using Dask -------')

                rslc_ref = np.zeros((self.n_image, box_length, box_width), np.complex64)
                quality = np.zeros((box_length, box_width), np.float32)
                SHP = np.zeros((self.shp_size, box_length, box_width), np.int)

                # initiate dask cluster and client
                cluster_obj = cluster_minopy.MDaskCluster(self.cluster, self.numWorker, config_name=self.config)
                cluster_obj.open()

                # run dask
                rslc_ref, SHP, quality = cluster_obj.run(func=inversion, func_data=data_kwargs,
                                                         results=[rslc_ref, SHP, quality],
                                                         dimlimits=[self.length, self.width])

                # close dask cluster and client
                cluster_obj.close()

                print('------- finished parallel processing -------\n\n')

            row1 = box[1]
            row2 = box[3]
            col1 = box[0]
            col2 = box[2]

            # write the block to disk
            self.rslc_ref[:, row1:row2, col1:col2] = rslc_ref[:, :, :]

            self.quality[row1:row2, col1:col2] = quality[:, :]

            self.shp[:, row1:row2, col1:col2] = SHP[:, :, :]

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))

        return

    def patch_slice(self):
        patch_row_1 = np.ogrid[0:self.length - self.azimuth_window:self.patch_size]
        patch_row_2 = patch_row_1 + self.patch_size
        patch_row_2[-1] = self.length

        patch_col_1 = np.ogrid[0:self.width - self.range_window:self.patch_size]
        patch_col_2 = patch_col_1 + self.patch_size
        patch_col_2[-1] = self.width

        num_box = len(patch_col_1) * len(patch_row_1)

        box_list = []
        for i in range(len(patch_row_1)):
            for j in range(len(patch_col_1)):
                box = (patch_col_1[j], patch_row_1[i], patch_col_2[j], patch_row_2[i])
                box_list.append(box)

        return box_list, num_box

#################################################


def get_shp_row_col(data, input_slc, distance_thresh, sample_rows, sample_cols, ref_row, ref_col,
                    azimuth_window, range_window):

    row_0, col_0 = data

    n_image, length, width = input_slc.shape

    sample_rows = row_0 + sample_rows
    sample_rows[sample_rows < 0] = -1
    sample_rows[sample_rows >= length] = -1

    sample_cols = col_0 + sample_cols
    sample_cols[sample_cols < 0] = -1
    sample_cols[sample_cols >= width] = -1

    x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)
    mask = 1 * (x >= 0) * (y >= 0)
    indx = np.where(mask == 1)
    x = x[indx[0], indx[1]]
    y = y[indx[0], indx[1]]

    rslc = input_slc[:, y, x].reshape(n_image, -1)
    testvec = np.sort(np.abs(rslc), axis=0)
    S1 = np.sort(np.abs(input_slc[:, row_0, col_0])).reshape(n_image, 1)

    data1 = np.repeat(S1, testvec.shape[1], axis=1)
    data_all = np.concatenate((data1, testvec), axis=0)

    res = np.zeros([azimuth_window, range_window])
    res[indx[0], indx[1]] = 1 * (np.apply_along_axis(mnp.ecdf_distance, 0, data_all) <= distance_thresh)
    ks_label = label(res, background=0, connectivity=2)
    ksres = 1 * (ks_label == ks_label[ref_row, ref_col]) * mask

    return ksres


def inversion(slc_file=None, shp_file=None, quality_file=None, box=None, distance_thresh=None,
              azimuth_window=None, range_window=None, phase_linking_method='EMI'):

    """
    box : [x0 y0 x1 y1]
    """

    sample_rows = np.ogrid[-((azimuth_window - 1) / 2):((azimuth_window - 1) / 2) + 1]
    sample_rows = sample_rows.astype(int)
    reference_row = np.array([(azimuth_window - 1) / 2]).astype(int)
    reference_row = reference_row - (azimuth_window - len(sample_rows))

    sample_cols = np.ogrid[-((range_window - 1) / 2):((range_window - 1) / 2) + 1]
    sample_cols = sample_cols.astype(int)
    reference_col = np.array([(range_window - 1) / 2]).astype(int)
    reference_col = reference_col - (range_window - len(sample_cols))

    with h5py.File(slc_file, 'r') as slcObj:
        ds = slcObj['slc']
        numSlc, length, width = ds.shape
        big_box = [box[0] - range_window, box[1] - azimuth_window,
                   box[2] + range_window, box[3] + azimuth_window]
        if big_box[0] <= 0:
            big_box[0] = 0
        if big_box[1] <= 0:
            big_box[1] = 0
        if big_box[2] > width:
            big_box[2] = width
        if big_box[3] > length:
            big_box[3] = length

        rslc = ds[:, big_box[1]:big_box[3], big_box[0]:big_box[2]]

    rslc_ref_memmap = np.memmap(os.path.join(os.path.dirname(shp_file), 'rslc_ref'), dtype='complex64', mode='r+',
                       shape=(numSlc, length, width))
    rslc_ref = rslc_ref_memmap[:, box[1]:box[3], box[0]:box[2]]

    shp_size = azimuth_window * range_window
    shp_memmap = np.memmap(shp_file, dtype='byte', mode='r+', shape=(shp_size, length, width))
    SHP = shp_memmap[:, box[1]:box[3], box[0]:box[2]]

    quality_memmap = np.memmap(quality_file, dtype='float32', mode='r+', shape=(length, width))
    quality = quality_memmap[box[1]:box[3], box[0]:box[2]]

    # In box coordinate
    row1 = box[1] - box[1]
    row2 = box[3] - box[1]
    col1 = box[0] - box[0]
    col2 = box[2] - box[0]

    lin = np.ogrid[row1:row2]
    overlap_length = len(lin)
    sam = np.ogrid[col1:col2]
    overlap_width = len(sam)
    lin, sam = np.meshgrid(lin, sam)

    coords = list(map(lambda y, x: (int(y), int(x)),
                      lin.T.reshape(overlap_length * overlap_width, 1),
                      sam.T.reshape(overlap_length * overlap_width, 1)))

    num_pixel2inv = len(coords)
    prog_bar = ptime.progressBar(maxValue=num_pixel2inv)
    i = 0

    for coord in coords:
        data = [coord[0] + box[1] - big_box[1], coord[1] + box[0] - big_box[0]]

        if not SHP[:, coord[0], coord[1]].any():
            shp = get_shp_row_col(data, rslc, distance_thresh, sample_rows, sample_cols,
                                  reference_row, reference_col, azimuth_window, range_window)

            SHP[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = shp.reshape(azimuth_window * range_window, 1, 1)
        else:
            shp = SHP[:, coord[0], coord[1]].reshape(azimuth_window, range_window)

        if quality[coord[0], coord[1]] == -1:

            num_shp = len(shp[shp > 0])
            shp_rows, shp_cols = np.where(shp == 1)
            shp_rows = np.array(shp_rows + data[0] - (azimuth_window - 1) / 2).astype(int)
            shp_cols = np.array(shp_cols + data[1] - (range_window - 1) / 2).astype(int)

            CCG = np.array(1.0 * np.arange(numSlc * len(shp_rows)).reshape(numSlc, len(shp_rows)))
            CCG = np.exp(1j * CCG)
            CCG[:, :] = np.array(rslc[:, shp_rows, shp_cols])

            coh_mat = mnp.est_corr(CCG)

            if num_shp > 20:

                if 'sequential' in phase_linking_method:
                    vec_refined = sequential_phase_linking(CCG, phase_linking_method, num_stack=10)
                else:
                    vec_refined = mnp.phase_linking_process(CCG, 0, phase_linking_method, squeez=False)
            else:
                vec_refined = mnp.test_PS(coh_mat)

            quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mnp.gam_pta(np.angle(coh_mat), vec_refined)
            phase_refined = np.angle(np.array(vec_refined)).reshape(numSlc, 1, 1)
            amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(numSlc, 1, 1)

            rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = \
                np.multiply(amp_refined, np.exp(1j * phase_refined))

        prog_bar.update(i + 1, every=1000, suffix='{}/{} pixels'.format(i + 1, num_pixel2inv))
        i += 1

    return rslc_ref, SHP, quality, box


def sequential_phase_linking(full_stack_complex_samples, method, num_stack=10):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    n_image = full_stack_complex_samples.shape[0]
    mini_stack_size = 10
    num_mini_stacks = np.int(np.floor(n_image / mini_stack_size))
    vec_refined = np.zeros([np.shape(full_stack_complex_samples)[0], 1]) + 0j

    for sstep in range(0, num_mini_stacks):

        first_line = sstep * mini_stack_size
        if sstep == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images = mnp.phase_linking_process(mini_stack_complex_samples, sstep, method)

            vec_refined[first_line:last_line, 0:1] = res[sstep::, 0:1]
        else:

            if num_stack == 1:
                mini_stack_complex_samples = np.zeros([1 + num_lines, full_stack_complex_samples.shape[1]]) + 0j
                mini_stack_complex_samples[0, :] = np.complex64(squeezed_images[-1, :])
                mini_stack_complex_samples[1::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = mnp.phase_linking_process(mini_stack_complex_samples, 1, method)
                vec_refined[first_line:last_line, :] = res[1::, :]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            else:
                mini_stack_complex_samples = np.zeros([sstep + num_lines, full_stack_complex_samples.shape[1]]) + 0j
                mini_stack_complex_samples[0:sstep, :] = squeezed_images
                mini_stack_complex_samples[sstep::, :] = full_stack_complex_samples[first_line:last_line, :]
                res, new_squeezed_image = mnp.phase_linking_process(mini_stack_complex_samples, sstep, method)
                vec_refined[first_line:last_line, :] = res[sstep::, :]
                squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            ###

    datum_connection_samples = squeezed_images
    datum_shift = np.angle(mnp.phase_linking_process(datum_connection_samples, 0, 'PTA', squeez=False))

    for sstep in range(len(datum_shift)):
        first_line = sstep * mini_stack_size
        if sstep == num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_size

        vec_refined[first_line:last_line, 0:1] = np.multiply(vec_refined[first_line:last_line, 0:1],
                                                  np.exp(1j * datum_shift[sstep:sstep + 1, 0:1]))

    return vec_refined


if __name__ == '__main__':
    main()

#################################################
