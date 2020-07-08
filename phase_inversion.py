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
import h5py
from minopy.objects import cluster_minopy
from mintpy.utils import ptime
from isceobj.Util.ImageUtil import ImageLib as IML
import gdal
from minopy.objects.slcStack import slcStack
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
    inversionObj.close()

    return None


class PhaseLink:
    def __init__(self, inps):
        self.work_dir = inps.work_dir
        self.phase_linking_method = inps.inversion_method
        self.range_window = inps.range_window
        self.azimuth_window = inps.azimuth_window
        self.patch_size = inps.patch_size
        self.start_index = inps.index_inversion
        self.cluster = inps.cluster
        self.numWorker = inps.numWorker
        self.config = inps.config
        self.out_dir = self.work_dir + '/inverted'
        os.makedirs(self.out_dir, exist_ok='True')

        self.shp_test = inps.shp_test
        self.shp_function = self.get_shp_function()

        # read input slcStack.h5
        self.slc_stack = inps.slc_stack
        slcStackObj = slcStack(self.slc_stack)
        self.metadata = slcStackObj.get_metadata()
        self.date_list = slcStackObj.get_date_list()
        self.n_image, self.length, self.width = slcStackObj.get_size()
        print('Total number of pixels {}'.format(self.length * self.width))

        self.shp_size = self.range_window * self.azimuth_window
        self.distance_thresh = mnp.ks_lut(self.n_image, self.n_image, alpha=0.01)

        self.box_list, self.num_box = self.patch_slice()

        self.mini_stack_default_size = 10
        self.new_num_mini_stacks = (self.n_image - self.start_index) // self.mini_stack_default_size
        self.temp_mini_stack_slc_size = self.initiate_datum()
        self.initiate_stacks()

        return

    def close(self):
        with open(self.out_dir + '/inverted_date_list.txt', 'w+') as f:
            dates = [date + '\n' for date in self.date_list]
            f.writelines(dates)

        if 'sequential' in self.phase_linking_method:

            if os.path.exists(self.out_dir + '/old'):
                os.system('rm -r {}'.format(self.out_dir + '/old'))

            datum_file = os.path.join(self.out_dir, 'datum.h5')
            squeezed_image_file = os.path.join(self.out_dir, 'squeezed_images')
            datum_shift_file = os.path.join(self.out_dir, 'datum_shift')

            squeezed_images_memmap = np.memmap(squeezed_image_file, dtype='complex64', mode='r',
                                        shape=(len(self.temp_mini_stack_slc_size), self.length, self.width))
            datum_shift_memmap = np.memmap(datum_shift_file, dtype='float32', mode='r',
                                        shape=(len(self.temp_mini_stack_slc_size), self.length, self.width))


            with h5py.File(datum_file, 'a') as ds:
                if 'squeezed_images' in ds.keys():
                    del ds['squeezed_images']
                    del ds['datum_shift']
                    del ds['miniStack_size']

                squeezed_images = ds.create_dataset('squeezed_images',
                                                    shape=(len(self.temp_mini_stack_slc_size), self.length, self.width),
                                                    maxshape=(None, self.length, self.width),
                                                    dtype='complex64')
                datum_shift = ds.create_dataset('datum_shift',
                                                    shape=(len(self.temp_mini_stack_slc_size), self.length, self.width),
                                                    maxshape=(None, self.length, self.width),
                                                    dtype='float32')
                for line in range(self.length):
                    squeezed_images[:, line:line+1, :] = squeezed_images_memmap[:, line:line+1, :]
                    datum_shift[:, line:line+1, :] = datum_shift_memmap[:, line:line+1, :]

                miniStack_size = ds.create_dataset('miniStack_size',
                                                    shape=(len(self.temp_mini_stack_slc_size), 1),
                                                    maxshape=(None, 1),
                                                    dtype='float32')
                miniStack_size[:] = self.temp_mini_stack_slc_size[:]

                self.metadata['FILE_TYPE'] = 'datum'
                for key, value in self.metadata.items():
                    ds.attrs[key] = value

            os.system('rm -r {} {}'.format(squeezed_image_file, datum_shift_file))

        return

    def get_shp_function(self):
        if self.shp_test == 'ks':
            shp_function = mnp.ks2smapletest
        elif self.shp_test == 'ad':
            shp_function = mnp.ADtest
        elif self.shp_test == 'ttest':
            shp_function = mnp.ttest_indtest
        else:  # default is KS 2 sample test
            shp_function = mnp.ks2smapletest
        return shp_function

    def window_for_shp(self):
        sample_rows = np.ogrid[-((self.azimuth_window - 1) / 2):((self.azimuth_window - 1) / 2) + 1]
        sample_rows = sample_rows.astype(int)
        reference_row = np.array([(self.azimuth_window - 1) / 2]).astype(int)
        reference_row = reference_row - (self.azimuth_window - len(sample_rows))

        sample_cols = np.ogrid[-((self.range_window - 1) / 2):((self.range_window - 1) / 2) + 1]
        sample_cols = sample_cols.astype(int)
        reference_col = np.array([(self.range_window - 1) / 2]).astype(int)
        reference_col = reference_col - (self.range_window - len(sample_cols))
        return sample_rows, sample_cols, reference_row, reference_col

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

    def initiate_stacks(self):

        rslc_ref_file = self.out_dir + '/rslc_ref'
        quality_file = self.out_dir + '/quality'
        shp_file = self.out_dir + '/shp'

        if os.path.exists(rslc_ref_file):
            temp_ds = gdal.Open(rslc_ref_file + '.vrt', gdal.GA_ReadOnly)
            n_image_rslc = temp_ds.RasterCount

            if n_image_rslc == self.n_image:
                rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='r+',
                                     shape=(self.n_image, self.length, self.width))
            else:
                old_dir = self.out_dir + '/old'
                os.makedirs(old_dir, exist_ok='True')
                os.system('mv {} {}'.format(rslc_ref_file + '*', old_dir))
                os.system('mv {} {}'.format(quality_file + '*', old_dir))

                rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='w+',
                                     shape=(self.n_image, self.length, self.width))

                old_rslc_ref = np.memmap(os.path.join(old_dir, 'rslc_ref'), dtype='complex64', mode='w+',
                                         shape=(self.n_image, self.length, self.width))
                for line in range(0, self.start_index):
                    rslc_ref[0:self.start_index, :, :] = old_rslc_ref[:, :, :]

                del old_rslc_ref
        else:
            rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='w+',
                                 shape=(self.n_image, self.length, self.width))

        if not os.path.isfile(shp_file):
            shp = np.memmap(shp_file, dtype='byte', mode='write',
                            shape=(self.shp_size, self.length, self.width))

        if not os.path.exists(quality_file):

            quality = np.memmap(quality_file, dtype='float32', mode='w+',
                                shape=(self.length, self.width))
            quality[:, :] = -1

        IML.renderISCEXML(rslc_ref_file, bands=self.n_image, nyy=self.length, nxx=self.width,
                          datatype='complex64', scheme='BSQ')
        IML.renderISCEXML(quality_file, bands=1, nyy=self.length, nxx=self.width,
                          datatype='float32', scheme='BIL')
        IML.renderISCEXML(shp_file, bands=self.shp_size, nyy=self.length, nxx=self.width,
                          datatype='byte', scheme='BSQ')
        return

    def initiate_datum(self):
        datum_file = self.out_dir + '/datum.h5'
        with h5py.File(datum_file, 'a') as ds:
            if 'squeezed_images' in ds.keys():
                old_squeezed_images = ds['squeezed_images']
                old_datum_shift = ds['datum_shift']
                old_num_mini_stacks = old_squeezed_images.shape[0]
                old_mini_stack_slc_size = ds['miniStack_size'][:]
            else:
                old_num_mini_stacks = 0
                old_mini_stack_slc_size = None

        total_num_mini_stacks = self.new_num_mini_stacks + old_num_mini_stacks
        temp_mini_stack_slc_size = np.zeros([total_num_mini_stacks, 1])

        if not old_mini_stack_slc_size is None:
            temp_mini_stack_slc_size[0: old_num_mini_stacks] = old_mini_stack_slc_size[:]

        for sstep in range(old_num_mini_stacks, total_num_mini_stacks):
            first_line = sstep * self.mini_stack_default_size
            if sstep == total_num_mini_stacks - 1:
                last_line = self.n_image
            else:
                last_line = first_line + self.mini_stack_default_size
            num_lines = last_line - first_line
            temp_mini_stack_slc_size[sstep, 0] = num_lines

        temp_squeezed_images_file = self.out_dir + '/squeezed_images'
        temp_datum_shift_file = self.out_dir + '/datum_shift'

        if not os.path.exists(temp_squeezed_images_file):
            temp_squeezed_image = np.memmap(temp_squeezed_images_file, dtype='complex64', mode='write',
                                            shape=(total_num_mini_stacks, self.length, self.width))
        else:
            temp_squeezed_image = np.memmap(temp_squeezed_images_file, dtype='complex64', mode='r+',
                                            shape=(total_num_mini_stacks, self.length, self.width))

        if not os.path.exists(temp_datum_shift_file):
            temp_datum_shift = np.memmap(temp_datum_shift_file, dtype='complex64', mode='write',
                                            shape=(total_num_mini_stacks, self.length, self.width))
        else:
            temp_datum_shift = np.memmap(temp_datum_shift_file, dtype='complex64', mode='r+',
                                            shape=(total_num_mini_stacks, self.length, self.width))

        if not old_mini_stack_slc_size is None:
            with h5py.File(datum_file, 'a') as ds:
                old_squeezed_images = ds['squeezed_images']
                old_datum_shift = ds['datum_shift']
                for line in range(self.length):
                    temp_squeezed_image[0:old_num_mini_stacks, line:line + 1, :] = old_squeezed_images[:, line:line + 1, :]
                    temp_datum_shift[0:old_num_mini_stacks, line:line+1, :] = old_datum_shift[:, line:line+1, :]

        return temp_mini_stack_slc_size

    def iterate_coords(self):

        time0 = time.time()
        sample_rows, sample_cols, reference_row, reference_col = self.window_for_shp()

        data_kwargs = {
            "slc_file": self.slc_stack,
            "out_dir": self.out_dir,
            "distance_thresh": self.distance_thresh,
            "azimuth_window": self.azimuth_window,
            "range_window": self.range_window,
            "phase_linking_method": self.phase_linking_method,
            "mini_stack_slc_size": self.temp_mini_stack_slc_size,
            "samples": {"sample_rows": sample_rows, "sample_cols": sample_cols,
                        "reference_row": reference_row, "reference_col": reference_col}
        }

        if 'sequential' in self.phase_linking_method:
            data_kwargs['new_num_mini_stacks'] = self.new_num_mini_stacks

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
                inversion(**data_kwargs)
            else:
                # parallel
                print('\n\n------- start parallel processing using Dask -------')

                # initiate dask cluster and client
                cluster_obj = cluster_minopy.MDaskCluster(self.cluster, self.numWorker, config_name=self.config)
                cluster_obj.open()

                cluster_obj.run(inversion, func_data=data_kwargs)

                # close dask cluster and client
                cluster_obj.close()

                print('------- finished parallel processing -------\n\n')

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))
        return

#################################################


def sequential_phase_linking(full_stack_complex_samples, method, mini_stack_default_size, new_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    n_image = full_stack_complex_samples.shape[0]
    vec_refined = np.zeros([np.shape(full_stack_complex_samples)[0], 1]) + 0j

    squeezed_images = None
    for sstep in range(new_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == new_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images = mnp.phase_linking_process(mini_stack_complex_samples, sstep, method)

            vec_refined[first_line:last_line, 0:1] = res[sstep::, 0:1]
        else:

            mini_stack_complex_samples = np.zeros([sstep + num_lines, full_stack_complex_samples.shape[1]]) + 0j
            mini_stack_complex_samples[0:sstep, :] = squeezed_images
            mini_stack_complex_samples[sstep::, :] = full_stack_complex_samples[first_line:last_line, :]
            res, new_squeezed_image = mnp.phase_linking_process(mini_stack_complex_samples, sstep, method)
            vec_refined[first_line:last_line, :] = res[sstep::, :]
            squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            ###

    return vec_refined, squeezed_images


def datum_connect(squeezed_images, vector_refined, mini_stack_slc_size, datum_shift_old):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector
    mini_stack_slc_size: a vector that has number of images refined in each ministack
    datum_shift_old: the old datum shift values from previous inversion, to reset based on new values

    Returns
    -------

    """

    datum_connection_samples = squeezed_images
    datum_shift = np.array(np.angle(mnp.phase_linking_process(datum_connection_samples, 0, 'PTA', squeez=False)))

    new_vector_refined = vector_refined

    if datum_shift_old:
        for step in range(len(datum_shift_old)):
            if step == 0:
                first_im = 0
                last_im = mini_stack_slc_size[0]
            else:
                first_im = np.sum(mini_stack_slc_size[0:step])
                last_im = np.sum(mini_stack_slc_size[0:step + 1])
            new_vector_refined[first_im:last_im] = np.multiply(vector_refined[first_im:last_im, 0:1],
                                                           np.exp(-1j * datum_shift_old[step:step + 1, 0]))
    vector_refined = new_vector_refined

    for step in range(len(datum_shift)):
        if step == 0:
            first_im = 0
            if len(mini_stack_slc_size) == 1:
                last_im = int(mini_stack_slc_size)
            else:
                last_im = int(mini_stack_slc_size[0])
        else:
            first_im = int(np.sum(mini_stack_slc_size[0:step]))
            last_im = int(np.sum(mini_stack_slc_size[0:step + 1]))

        new_vector_refined[first_im:last_im, 0] = np.multiply(vector_refined[first_im:last_im, 0],
                                                           np.exp(1j * datum_shift[step:step + 1, 0]))

    return new_vector_refined.reshape(-1, 1, 1), datum_shift


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


def inversion(slc_file=None, out_dir=None, box=None, distance_thresh=None, azimuth_window=None, range_window=None,
              phase_linking_method='EMI', mini_stack_slc_size=None, samples=None, new_num_mini_stacks=None):

    """
    box : [x0 y0 x1 y1]
    """
    sample_rows = samples['sample_rows']
    sample_cols = samples['sample_cols']
    reference_row = samples['reference_row']
    reference_col = samples['reference_col']

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

    with h5py.File(os.path.join(out_dir, 'datum.h5'), 'r') as datumObj:
        if 'squeezed_images' in datumObj.keys():
            old_squeezed_images = datumObj['squeezed_images'][:, big_box[1]:big_box[3], big_box[0]:big_box[2]]
            old_datum_shift = datumObj['datum_shift'][:, box[1]:box[3], box[0]:box[2]]
        else:
            old_squeezed_images = None
            old_datum_shift = None

    shp_size = azimuth_window * range_window

    temp_squeezed_images = get_numpy_data_from_file(out_dir, 'squeezed_images', big_box, 'complex64',
                                                    shape=(len(mini_stack_slc_size), length, width))

    temp_datum_shift = get_numpy_data_from_file(out_dir, 'datum_shift', box, 'float32',
                                                shape=(len(mini_stack_slc_size), length, width))

    rslc_ref = get_numpy_data_from_file(out_dir, 'rslc_ref', box, 'complex64', shape=(numSlc, length, width))

    SHP = get_numpy_data_from_file(out_dir, 'shp', box, 'byte', shape=(shp_size, length, width))

    quality = get_numpy_data_from_file(out_dir, 'quality', box, 'float32', shape=(length, width))

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
                    vec_refined, squeezed_images = sequential_phase_linking(CCG, phase_linking_method, 10, new_num_mini_stacks)
                    do_datum = True

                else:
                    vec_refined, squeezed_images = mnp.phase_linking_process(CCG, 0, phase_linking_method, squeez=True)

            else:
                vec_refined = mnp.test_PS(coh_mat)
                squeezed_images = None
                do_datum = False

            quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mnp.gam_pta(np.angle(coh_mat), vec_refined)
            phase_refined = np.angle(np.array(vec_refined)).reshape(numSlc, 1, 1)
            amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(numSlc, 1, 1)

            rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = \
                np.multiply(amp_refined, np.exp(1j * phase_refined))

            if do_datum:
                if old_squeezed_images:
                    ccg_datum = np.array(1.0 * np.arange(old_squeezed_images.shape[0] * len(shp_rows)
                                                         ).reshape(old_squeezed_images.shape[0], len(shp_rows)))
                    ccg_datum = np.exp(1j * ccg_datum)
                    ccg_datum[:, :] = np.array(old_squeezed_images[:, shp_rows, shp_cols])
                    squeezed_images = np.vstack([ccg_datum, squeezed_images])
                    old_datum_shift_coord = old_datum_shift[:, coord[0] - box[1], coord[1] - box[0]]
                else:
                    old_datum_shift_coord = None

                rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1], datum_shift = datum_connect(squeezed_images,
                                  rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1].reshape(-1, 1),
                                  mini_stack_slc_size,
                                  old_datum_shift_coord)

                rowcol_set = [(r, c) for r, c in zip(shp_rows, shp_cols)]
                ref_row = int(reference_row + data[0] - (azimuth_window - 1) / 2)
                ref_col = int(reference_col + data[1] - (range_window - 1) / 2)
                target_index = rowcol_set.index((ref_row, ref_col))

                temp_squeezed_images[:, coord[0]:coord[0] + 1, coord[1]: coord[1] + 1] = \
                    np.array(squeezed_images[:, target_index]).reshape(-1, 1, 1)

                temp_datum_shift[:, coord[0]:coord[0] + 1, coord[1]: coord[1] + 1] = datum_shift.reshape(-1, 1, 1)

        prog_bar.update(i + 1, every=1000, suffix='{}/{} pixels'.format(i + 1, num_pixel2inv))
        i += 1

    return


def get_numpy_data_from_file(out_dir, file, box, dtype, shape=None):
    temp_image_memmap = np.memmap(os.path.join(out_dir, file), dtype=dtype, mode='r+', shape=shape)
    if len(shape) == 3:
        temp_image = temp_image_memmap[:, box[1]:box[3], box[0]:box[2]]
    else:
        temp_image = temp_image_memmap[box[1]:box[3], box[0]:box[2]]
    return temp_image


if __name__ == '__main__':
    main()

#################################################
