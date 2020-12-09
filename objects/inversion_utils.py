#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################
import logging
import warnings


warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import os
import numpy as np
import minopy_utilities as mut
from skimage.measure import label
import h5py
import gdal
from isceobj.Util.ImageUtil import ImageLib as IML
import time


def get_big_box(box, range_window, azimuth_window, width, length):
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
    return big_box


def sequential_phase_linking(full_stack_complex_samples, method, mini_stack_default_size, total_num_mini_stacks):
    """ phase linking of each pixel sequentially and applying a datum shift at the end """

    n_image = full_stack_complex_samples.shape[0]
    vec_refined = np.zeros([np.shape(full_stack_complex_samples)[0], 1]) + 0j

    squeezed_images = None
    for sstep in range(total_num_mini_stacks):

        first_line = sstep * mini_stack_default_size
        if sstep == total_num_mini_stacks - 1:
            last_line = n_image
        else:
            last_line = first_line + mini_stack_default_size
        num_lines = last_line - first_line

        if sstep == 0:

            mini_stack_complex_samples = full_stack_complex_samples[first_line:last_line, :]
            res, squeezed_images = mut.phase_linking_process(mini_stack_complex_samples, sstep, method)

            vec_refined[first_line:last_line, 0:1] = res[sstep::, 0:1]
        else:

            mini_stack_complex_samples = np.zeros([sstep + num_lines, full_stack_complex_samples.shape[1]]) + 0j
            mini_stack_complex_samples[0:sstep, :] = squeezed_images
            mini_stack_complex_samples[sstep::, :] = full_stack_complex_samples[first_line:last_line, :]
            res, new_squeezed_image = mut.phase_linking_process(mini_stack_complex_samples, sstep, method)
            vec_refined[first_line:last_line, :] = res[sstep::, :]
            squeezed_images = np.vstack([squeezed_images, new_squeezed_image])
            ###

    return vec_refined, squeezed_images


def datum_connect(squeezed_images, vector_refined, mini_stack_size):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector

    Returns
    -------

    """

    datum_connection_samples = squeezed_images
    datum_shift = np.array(np.angle(mut.phase_linking_process(datum_connection_samples, 0, 'PTA', squeez=False)))
    new_vector_refined = np.zeros([len(vector_refined), 1]).astype(complex)

    for step in range(len(datum_shift)):
        first_line = step * mini_stack_size
        if step == len(datum_shift) - 1:
            last_line = len(vector_refined)
        else:
            last_line = first_line + mini_stack_size
        new_vector_refined[first_line:last_line, 0] = np.multiply(vector_refined[first_line:last_line, 0],
                                                              np.exp(1j * datum_shift[step:step + 1, 0]))

    return new_vector_refined.reshape(-1, 1, 1), datum_shift


def get_shp_row_col(data, input_slc, distance_thresh, samp_rows, samp_cols, ref_row, ref_col,
                    azimuth_window, range_window):

    row_0, col_0 = data

    n_image, length, width = input_slc.shape

    sample_rows = row_0 + samp_rows
    sample_rows[sample_rows < 0] = -1
    sample_rows[sample_rows >= length] = -1

    sample_cols = col_0 + samp_cols
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
    res[indx[0], indx[1]] = 1 * (np.apply_along_axis(mut.ecdf_distance, 0, data_all) <= distance_thresh)
    ks_label = label(res, background=0, connectivity=2)
    ksres = 1 * (ks_label == ks_label[ref_row, ref_col]) * mask

    return ksres


def get_numpy_data_from_file(file, box, dtype, shape=None):
    temp_image_memmap = np.memmap(file, dtype=dtype, mode='r+', shape=shape)
    if box:
        if len(shape) == 3:
            temp_image = temp_image_memmap[:, box[1]:box[3], box[0]:box[2]]
        else:
            temp_image = temp_image_memmap[box[1]:box[3], box[0]:box[2]]
        return temp_image
    else:
        return temp_image_memmap


def initiate_stacks(slc_stack, patch_dir, box, mini_stack_default_size, phase_linking_method, shp_size, range_window,
                    azimuth_window):

    shp_file = patch_dir + '/shp'
    rslc_file = patch_dir + '/rslc'
    datum_file = patch_dir + '/datum_shift'
    squeezed_file = patch_dir + '/squeezed_images'
    rslc_ref_file = patch_dir + '/rslc_ref'
    quality_file = patch_dir + '/quality'

    # create PATCH_* folder if not exist
    os.makedirs(patch_dir, exist_ok=True)

    length = box[3] - box[1]
    width = box[2] - box[0]

    # read current slc_stack to be inverted
    RSLC = h5py.File(slc_stack, 'a')
    rslc_stack = RSLC['slc']
    numSlc, rslc_length, rslc_width = rslc_stack.shape

    # big box is box + shp window
    big_box = get_big_box(box, range_window, azimuth_window, rslc_width, rslc_length)
    big_length = big_box[3] - big_box[1]
    big_width = big_box[2] - big_box[0]

    if 'sequential' in phase_linking_method:
        total_num_mini_stacks = numSlc // mini_stack_default_size

        # subset squeezed images stack for the patch with the big box size
        if not os.path.exists(squeezed_file):
            squeezed_image = np.memmap(squeezed_file, dtype='complex64', mode='w+', shape=(total_num_mini_stacks,
                                                                                           length, width))
            IML.renderISCEXML(squeezed_file, bands=total_num_mini_stacks, nyy=length, nxx=width,
                              datatype='complex64', scheme='BSQ')

        # subset datum shift stack for the patch with the big box size
        if not os.path.exists(datum_file):
            datum_shift = np.memmap(datum_file, dtype='float32', mode='w+',
                                    shape=(total_num_mini_stacks, length, width))
            IML.renderISCEXML(datum_file, bands=total_num_mini_stacks, nyy=length, nxx=width,
                              datatype='float32', scheme='BSQ')
    else:
        total_num_mini_stacks = 1

    # subset slc stack for the patch with the big box size
    if not os.path.exists(rslc_file):
        rslc = np.memmap(rslc_file, dtype='complex64', mode='w+', shape=(numSlc, big_length, big_width))
        rslc[:, :, :] = rslc_stack[:, big_box[1]: big_box[3], big_box[0]:big_box[2]]
        RSLC.close()
        IML.renderISCEXML(rslc_file, bands=numSlc, nyy=length, nxx=width, datatype='complex64', scheme='BSQ')

    # initiate refined slc file for the patch
    if not os.path.exists(rslc_ref_file):
        rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='w+', shape=(numSlc, length, width))
        IML.renderISCEXML(rslc_ref_file, bands=numSlc, nyy=length, nxx=width, datatype='complex64', scheme='BSQ')

    # initiate shp file for the patch
    if not os.path.isfile(shp_file):
        shp = np.memmap(shp_file, dtype='byte', mode='w+', shape=(shp_size, length, width))
        IML.renderISCEXML(shp_file, bands=shp_size, nyy=length, nxx=width, datatype='byte', scheme='BSQ')

    # initiate quality file for the patch
    if not os.path.exists(quality_file):
        quality = np.memmap(quality_file, dtype='float32', mode='w+', shape=(length, width))
        quality[:, :] = -1
        IML.renderISCEXML(quality_file, bands=1, nyy=length, nxx=width, datatype='float32', scheme='BIL')

    return big_box, total_num_mini_stacks


def parallel_invertion(distance_thresh=None, azimuth_window=None, range_window=None, default_mini_stack_size=10,
              phase_linking_method='EMI', samples=None, patch_length=None, patch_width=None,
              patch_dir=None, box=None, big_box=None, patch_row_0=0, patch_col_0=0, total_num_mini_stacks=None):

    """
    box : [x0 y0 x1 y1]
    """
    sample_rows = samples['sample_rows']
    sample_cols = samples['sample_cols']
    reference_row = samples['reference_row']
    reference_col = samples['reference_col']
    shp_size = azimuth_window * range_window

    rslc_file = patch_dir + '/rslc'
    shp_file = patch_dir + '/shp'
    datum_file = patch_dir + '/datum_shift'
    squeezed_file = patch_dir + '/squeezed_images'
    rslc_ref_file = patch_dir + '/rslc_ref'
    quality_file = patch_dir + '/quality'

    slc_stack = gdal.Open(rslc_ref_file + '.vrt', gdal.GA_ReadOnly)
    numSlc = slc_stack.RasterCount

    big_length = big_box[3] - big_box[1]
    big_width = big_box[2] - big_box[0]

    slc_images = get_numpy_data_from_file(rslc_file, None, 'complex64', shape=(numSlc, big_length, big_width))

    rslc_ref = get_numpy_data_from_file(rslc_ref_file, None, 'complex64', shape=(numSlc, patch_length, patch_width))

    SHP = get_numpy_data_from_file(shp_file, None, 'byte', shape=(shp_size, patch_length, patch_width))

    quality = get_numpy_data_from_file(quality_file, None, 'float32', shape=(patch_length, patch_width))

    if not -1 in quality:
        return

    if 'sequential' in phase_linking_method:
        temp_squeezed_images = get_numpy_data_from_file(squeezed_file, None, 'complex64', shape=(total_num_mini_stacks,
                                                                                            patch_length, patch_width))

        temp_datum_shift = get_numpy_data_from_file(datum_file, None, 'float32', shape=(total_num_mini_stacks,
                                                                                   patch_length, patch_width))
        do_datum = True
    else:
        temp_squeezed_images = None
        temp_datum_shift = None
        do_datum = False

    # In box coordinate
    row1 = box[1] - patch_row_0
    row2 = box[3] - patch_row_0
    col1 = box[0] - patch_col_0
    col2 = box[2] - patch_col_0

    lin = np.ogrid[row1:row2]
    overlap_length = len(lin)
    sam = np.ogrid[col1:col2]
    overlap_width = len(sam)
    lin, sam = np.meshgrid(lin, sam)

    coords = list(map(lambda y, x: (int(y), int(x)),
                      lin.T.reshape(overlap_length * overlap_width, 1),
                      sam.T.reshape(overlap_length * overlap_width, 1)))

    def invert_coord(coord):
        # big box coordinate:
        data = [coord[0] + patch_row_0 - big_box[1], coord[1] + patch_col_0 - big_box[0]]

        if quality[coord[0], coord[1]] == -1:

            shp = get_shp_row_col(data, slc_images, distance_thresh, sample_rows, sample_cols,
                                  reference_row, reference_col, azimuth_window, range_window)

            SHP[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = shp.reshape(azimuth_window * range_window, 1, 1)

            num_shp = len(shp[shp > 0])
            shp_rows, shp_cols = np.where(shp == 1)
            rowcol_set = [(r, c) for r, c in zip(shp_rows, shp_cols)]
            target_index = rowcol_set.index((int(reference_row), int(reference_col)))
            shp_rows = np.array(shp_rows + data[0] - (azimuth_window - 1) / 2).astype(int)
            shp_cols = np.array(shp_cols + data[1] - (range_window - 1) / 2).astype(int)
            CCG = np.array(1.0 * np.arange(numSlc * len(shp_rows)).reshape(numSlc, len(shp_rows))).astype(complex)
            CCG[:, :] = np.array(slc_images[:, shp_rows, shp_cols])

            coh_mat = mut.est_corr(CCG)

            if num_shp > 20:

                if 'sequential' in phase_linking_method:
                    vec_refined, squeezed_images = sequential_phase_linking(CCG, phase_linking_method, 10,
                                                                            total_num_mini_stacks)

                else:
                    vec_refined = mut.phase_linking_process(CCG, 0, phase_linking_method, squeez=False)

            else:
                vec_refined = mut.test_PS(coh_mat)
                squeezed_images = None

            phase_refined = np.angle(np.array(vec_refined)).reshape(numSlc, 1, 1)
            amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(numSlc, 1, 1)

            vec_refined = np.multiply(amp_refined, np.exp(1j * phase_refined))

            if do_datum and num_shp > 20:

                vec_refined, datum_shift = \
                    datum_connect(squeezed_images, vec_refined.reshape(-1, 1), default_mini_stack_size)

                temp_squeezed_images[:, coord[0]:coord[0] + 1, coord[1]: coord[1] + 1] = \
                    np.array(squeezed_images[:, target_index]).reshape(-1, 1, 1)

                temp_datum_shift[:, coord[0]:coord[0] + 1, coord[1]: coord[1] + 1] = datum_shift.reshape(-1, 1, 1)

            rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = vec_refined
            quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mut.gam_pta(np.angle(coh_mat),
                                                                                vec_refined.reshape(-1, 1))

        return coord

    t1 = time.perf_counter()

    results = map(invert_coord, coords)
    for result in results:
        continue

    '''
    processes = list()
    for coord in coords:
        prc = Process(target=invert_coord, args=(coord,))
        processes.append(prc)
        prc.start()

    for index, proc in enumerate(processes):
        proc.join()
    '''

    t2 = time.perf_counter()

    print(f'finish time : {t2 - t1} seconds')

    np.save(patch_dir + '/flag.npy', '{} is done inverting'.format(os.path.basename(patch_dir)))

    return

