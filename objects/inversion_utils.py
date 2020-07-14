#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import numpy as np
import minopy_utilities as mut
from skimage.measure import label
import h5py
from mintpy.utils import ptime
from minopy.objects.arg_parser import MinoPyParser


def inversion(slc_stack_file=None, distance_thresh=None, azimuth_window=None, range_window=None, phase_linking_method='EMI',
              total_mini_stack_slc_size=None, samples=None, slc_start_index=0, total_num_images=None,
              num_inverted_images=None, new_num_mini_stacks=None, patch_length=None, patch_width=None,
              patch_dir=None, box=None, patch_row_0=0, patch_col_0=0):

    """
    box : [x0 y0 x1 y1]
    """
    sample_rows = samples['sample_rows']
    sample_cols = samples['sample_cols']
    reference_row = samples['reference_row']
    reference_col = samples['reference_col']

    slcObj = h5py.File(slc_stack_file, 'r')
    slc_stack = slcObj['slc']
    numSlc, length, width = slc_stack.shape
    numSlc -= slc_start_index
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

    slc_images = slc_stack[slc_start_index::, big_box[1]:big_box[3], big_box[0]:big_box[2]]

    out_dir = os.path.dirname(patch_dir)

    shp_size = azimuth_window * range_window
    total_num_mini_stacks = len(total_mini_stack_slc_size)
    if new_num_mini_stacks:
        old_num_mini_stacks = total_num_mini_stacks - new_num_mini_stacks
    else:
        old_num_mini_stacks = None

    temp_squeezed_images = get_numpy_data_from_file(out_dir, 'squeezed_images', big_box, 'complex64',
                                                    shape=(total_num_mini_stacks, length, width))

    temp_datum_shift = get_numpy_data_from_file(out_dir, 'datum_shift', box, 'float32',
                                                shape=(total_num_mini_stacks, length, width))

    rslc_ref = get_numpy_data_from_file(patch_dir, 'rslc_ref', None, 'complex64', shape=(total_num_images, patch_length,
                                                                                         patch_width))

    SHP = get_numpy_data_from_file(patch_dir, 'shp', None, 'byte', shape=(shp_size, patch_length, patch_width))

    quality = get_numpy_data_from_file(patch_dir, 'quality', None, 'float32', shape=(patch_length, patch_width))

    # In box coordinate
    row1 = box[1] - patch_row_0 #box[1]
    row2 = box[3] - patch_row_0 #box[1]
    col1 = box[0] - patch_col_0 #box[0]
    col2 = box[2] - patch_col_0 #box[0]

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
        data = [coord[0] + patch_row_0 - big_box[1], coord[1] + patch_col_0 - big_box[0]]

        if not SHP[:, coord[0], coord[1]].any():
            shp = get_shp_row_col(data, slc_images, distance_thresh, sample_rows, sample_cols,
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
            CCG[:, :] = np.array(slc_images[:, shp_rows, shp_cols])

            coh_mat = mut.est_corr(CCG)

            do_datum = False
            if num_shp > 20:

                if 'sequential' in phase_linking_method:
                    vec_refined, squeezed_images = sequential_phase_linking(CCG, phase_linking_method, 10, new_num_mini_stacks)
                    do_datum = True

                else:
                    vec_refined, squeezed_images = mut.phase_linking_process(CCG, 0, phase_linking_method, squeez=True)

            else:
                vec_refined = mut.test_PS(coh_mat)
                squeezed_images = None

            quality[coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = mut.gam_pta(np.angle(coh_mat), vec_refined)
            phase_refined = np.angle(np.array(vec_refined)).reshape(numSlc, 1, 1)
            amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(numSlc, 1, 1)

            rslc_ref[num_inverted_images::, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1] = \
                np.multiply(amp_refined, np.exp(1j * phase_refined))

            if do_datum:
                if not new_num_mini_stacks == total_num_mini_stacks and old_num_mini_stacks:
                    ccg_datum = temp_squeezed_images[0: old_num_mini_stacks, shp_rows, shp_cols].reshape(
                        old_num_mini_stacks, -1)
                    squeezed_images = np.vstack([ccg_datum, squeezed_images])
                    old_datum_shift = temp_datum_shift[0:old_num_mini_stacks, coord[0], coord[1]]
                else:
                    old_datum_shift = None

                rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1], datum_shift = datum_connect(squeezed_images,
                                  rslc_ref[:, coord[0]:coord[0] + 1, coord[1]:coord[1] + 1].reshape(-1, 1),
                                  total_mini_stack_slc_size,
                                  old_datum_shift)

                rowcol_set = [(r, c) for r, c in zip(shp_rows, shp_cols)]
                ref_row = int(reference_row + data[0] - (azimuth_window - 1) / 2)
                ref_col = int(reference_col + data[1] - (range_window - 1) / 2)
                target_index = rowcol_set.index((ref_row, ref_col))

                temp_squeezed_images[:, data[0]:data[0] + 1, data[1]: data[1] + 1] = \
                    np.array(squeezed_images[:, target_index]).reshape(-1, 1, 1)

                temp_datum_shift[:, coord[0]:coord[0] + 1, coord[1]: coord[1] + 1] = datum_shift.reshape(-1, 1, 1)

        prog_bar.update(i + 1, every=100, suffix='{}/{} pixels'.format(i + 1, num_pixel2inv))
        i += 1

    return

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


def datum_connect(squeezed_images, vector_refined, mini_stack_slc_size, old_datum_shift):
    """

    Parameters
    ----------
    squeezed_images: a 2D matrix in format of squeezed_images * num_of_samples
    vector_refined: n*1 refined complex vector
    mini_stack_slc_size: a vector that has number of images refined in each ministack
    old_datum_shift: the old datum shift values from previous inversion, to reset based on new values

    Returns
    -------

    """

    datum_connection_samples = squeezed_images
    datum_shift = np.array(np.angle(mut.phase_linking_process(datum_connection_samples, 0, 'PTA', squeez=False)))

    new_vector_refined = vector_refined

    if old_datum_shift:
        for step in range(len(old_datum_shift)):
            if step == 0:
                first_im = 0
                last_im = mini_stack_slc_size[0]
            else:
                first_im = np.sum(mini_stack_slc_size[0:step])
                last_im = np.sum(mini_stack_slc_size[0:step + 1])
            new_vector_refined[first_im:last_im] = np.multiply(vector_refined[first_im:last_im, 0:1],
                                                           np.exp(-1j * old_datum_shift[step:step + 1, 0]))
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
    res[indx[0], indx[1]] = 1 * (np.apply_along_axis(mut.ecdf_distance, 0, data_all) <= distance_thresh)
    ks_label = label(res, background=0, connectivity=2)
    ksres = 1 * (ks_label == ks_label[ref_row, ref_col]) * mask

    return ksres


def get_numpy_data_from_file(out_dir, file, box, dtype, shape=None):
    temp_image_memmap = np.memmap(os.path.join(out_dir, file), dtype=dtype, mode='r+', shape=shape)
    if box:
        if len(shape) == 3:
            temp_image = temp_image_memmap[:, box[1]:box[3], box[0]:box[2]]
        else:
            temp_image = temp_image_memmap[box[1]:box[3], box[0]:box[2]]
        return  temp_image
    else:
        return temp_image_memmap


def initiate_stacks(patch_dir, box, n_inverted, n_total, shp_size):
    from isceobj.Util.ImageUtil import ImageLib as IML

    out_dir = os.path.dirname(patch_dir)
    RSLCfile = out_dir + '/rslc_ref.h5'
    RSLC = h5py.File(RSLCfile, 'a')
    if 'slc' in RSLC.keys():
        old_rslc_ref = RSLC['slc']
        old_shp = RSLC['shp']
    else:
        old_rslc_ref = None
        old_shp = None

    os.makedirs(patch_dir, exist_ok=True)

    rslc_ref_file = patch_dir + '/rslc_ref'
    quality_file = patch_dir + '/quality'
    shp_file = patch_dir + '/shp'

    length = box[3] - box[1]
    width = box[2] - box[0]

    if os.path.exists(rslc_ref_file):
        rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='r+',
                                 shape=(n_total, length, width))
    else:

        rslc_ref = np.memmap(rslc_ref_file, dtype='complex64', mode='w+',
                                 shape=(n_total, length, width))
        if old_rslc_ref:
            rslc_ref[0:n_inverted, :, :] = old_rslc_ref[:, box[1]: box[3], box[0]:box[2]]

    if os.path.isfile(shp_file):
        shp = np.memmap(shp_file, dtype='byte', mode='r+', shape=(shp_size, length, width))
    else:
        shp = np.memmap(shp_file, dtype='byte', mode='w+', shape=(shp_size, length, width))
        if old_shp:
            shp[:, :, :] = old_shp[:, box[1]: box[3], box[0]:box[2]]

    if os.path.exists(quality_file):

        quality = np.memmap(quality_file, dtype='float32', mode='r+', shape=(length, width))
    else:
        quality = np.memmap(quality_file, dtype='float32', mode='w+', shape=(length, width))
        quality[:, :] = -1

    IML.renderISCEXML(rslc_ref_file, bands=n_total, nyy=length, nxx=width,
                      datatype='complex64', scheme='BSQ')
    IML.renderISCEXML(quality_file, bands=1, nyy=length, nxx=width,
                      datatype='float32', scheme='BIL')
    IML.renderISCEXML(shp_file, bands=shp_size, nyy=length, nxx=width,
                      datatype='byte', scheme='BSQ')
    RSLC.close()
    return

