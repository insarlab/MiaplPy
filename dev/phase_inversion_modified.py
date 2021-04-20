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
import time
import numpy as np
import h5py

import minopy.minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
from mintpy.utils import ptime
from minopy.objects.slcStack import slcStack
import minopy.objects.inversion_utils as iut
from skimage.measure import label
#from minopy.lib.phase_inversion_c import PhaseLink as PhaseLink_c
from isceobj.Util.ImageUtil import ImageLib as IML
import glob
import shutil
#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    inversionObj = PhaseLink(inps)
    inversionObj.loop_patches()
    inversionObj.close()

    return None


def write_hdf5_block(fname, data, datasetName, block=None, mode='a', print_msg=True):
    """Write data to existing HDF5 dataset in disk block by block.
    Parameters: data        - np.ndarray 1/2/3D matrix
                datasetName - str, dataset name
                block       - list of 2/4/6 int, for
                              [zStart, zEnd,
                               yStart, yEnd,
                               xStart, xEnd]
                mode        - str, open mode
    Returns:    fname
    """

    # default block value
    if block is None:

        # data shape
        if isinstance(data, list):
            shape=(len(data),)
        else:
            shape = data.shape

        # set default block as the entire data
        if len(shape) ==1:
            block = [0, shape[0]]
        elif len(shape) == 2:
            block = [0, shape[0],
                     0, shape[1]]
        elif len(shape) == 3:
            block = [0, shape[0],
                     0, shape[1],
                     0, shape[2]]

    # write
    if print_msg:
        print('-'*50)
        print('open  HDF5 file {} in {} mode'.format(fname, mode))
        print("writing dataset /{:<25} block: {}".format(datasetName, block))
    with h5py.File(fname, mode) as f:
        if len(block) == 6:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3],
                           block[4]:block[5]] = data

        elif len(block) == 4:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3]] = data

        elif len(block) == 2:
            f[datasetName][block[0]:block[1]] = data

    if print_msg:
        print('close HDF5 file {}.'.format(fname))
    return fname

def get_shp_row_col_f(data, input_slc, def_sample_rows, def_sample_cols, azimuth_window,
                      range_window, reference_row, reference_col, distance_threshold):

    row_0 = data[0]
    col_0 = data[1]
    n_image = input_slc.shape[0]
    length = input_slc.shape[1]
    width = input_slc.shape[2]

    sample_rows = row_0 + def_sample_rows
    sample_rows[sample_rows < 0] = -1
    sample_rows[sample_rows >= length] = -1

    sample_cols = col_0 + def_sample_cols
    sample_cols[sample_cols < 0] = -1
    sample_cols[sample_cols >= width] = -1

    sample_cols = sample_cols[np.flatnonzero(sample_cols >= 0)]
    sample_rows = sample_rows[np.flatnonzero(sample_rows >= 0)]

    ref_row = reference_row - azimuth_window + len(sample_rows)
    ref_col = reference_col - range_window + len(sample_cols)

    x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

    rslc = input_slc[:, y, x].reshape(n_image, -1)
    testvec = np.sort(np.abs(rslc), axis=0)
    S1 = np.sort(np.abs(input_slc[:, row_0, col_0])).reshape(n_image, 1)

    data1 = np.repeat(S1, testvec.shape[1], axis=1)
    data_all = np.concatenate((data1, testvec), axis=0)

    res = 1 * (np.apply_along_axis(mut.ecdf_distance, 0, data_all) <= distance_thresh)
    res = res.reshape(len(sample_rows), len(sample_cols))
    ks_label = label(res, background=0, connectivity=2)
    ksres = 1 * (ks_label == ks_label[ref_row, ref_col])

    return ksres, sample_rows[0], sample_cols[0]


def process_patch_f(box=None, RSLCfile=None, range_window=None, azimuth_window=None, width=None, length=None,
                    n_image=None, slcStackObj=None):

    with h5py.File(RSLCfile, 'r') as f:
        quality = f['quality'][box[1]:box[3], box[0]:box[2]]

    if not np.any(quality < 0):
        return

    box_width = box[2] - box[0]
    box_length = box[3] - box[1]

    big_box = iut.get_big_box(box, range_window, azimuth_window, width, length)

    # In box coordinate
    row1 = box[1] - big_box[1]
    row2 = box[3] - big_box[1]
    col1 = box[0] - big_box[0]
    col2 = box[2] - big_box[0]

    lin = np.arange(row1, row2)
    overlap_length = len(lin)
    sam = np.arange(col1, col2)
    overlap_width = len(sam)
    lin, sam = np.meshgrid(lin, sam)
    coords = set(map(lambda y, x: (int(y), int(x)),
                     lin.T.reshape(overlap_length * overlap_width, 1),
                     sam.T.reshape(overlap_length * overlap_width, 1)))

    patch_slc_images = slcStackObj.read(datasetName='slc', box=big_box)
    rslc_ref = np.zeros([n_image, box_length, box_width], dtype='complex')

    def invert_coord_f(data):
        CCG = None
        result = {}

        # big box coordinate:
        shp, row0, col0 = get_shp_row_col_f(data, patch_slc_images)

        num_shp = len(shp[shp > 0])
        shp_rows, shp_cols = np.nonzero(shp)
        shp_rows = np.array(shp_rows + row0).astype(int)
        shp_cols = np.array(shp_cols + col0).astype(int)

        CCG = np.array(self.patch_slc_images[:, shp_rows, shp_cols])
        coh_mat = mut.est_corr(CCG)

        squeezed_images = None

        if num_shp > 20:

            if self.sequential:
                vec_refined, squeezed_images = iut.sequential_phase_linking(CCG, self.phase_linking_method, 10, self.total_num_mini_stacks)

            else:
                vec_refined = mut.phase_linking_process(CCG, 0, self.phase_linking_method, squeez=False)

        else:
            vec_refined = mut.test_PS(coh_mat)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(-1, 1)
        vec_refined /= np.abs(vec_refined)
        vec_refined *= amp_refined

        if self.sequential and num_shp > 20:
            vec_refined = iut.datum_connect(squeezed_images, vec_refined, self.default_mini_stack_size)

        result['x'] = data[1]
        result['y'] = data[0]
        result['rvector'] = vec_refined
        result['quality'] = mut.gam_pta(np.angle(coh_mat), vec_refined)

        return result

    results = map(self.invert_coord, coords)
    num_points = len(coords)

    block = [0, self.n_image, box[1], box[3], box[0], box[2]]

    prog_bar = ptime.progressBar(maxValue=num_points)
    t = 0
    time0 = time.time()
    for result in results:
        rf = result['y'] - row1
        cf = result['x'] - col1
        rslc_ref[:, rf:rf+1, cf:cf+1] = result['rvector'].reshape(-1, 1, 1)
        quality[rf:rf+1, cf:cf+1] = result['quality']
        prog_bar.update(t + 1, every=20, suffix='{}/{} pixels, box: {}/{}'.format(t + 1, num_points, i + 1, self.num_box))
        t += 1
    print('Total time for Box {}: {} s'.format(i, time.time() - time0))

    return rslc_ref, quality, box


class PhaseLink:
    def __init__(self, inps):

        if inps.cluster == 'local':
            from dask.distributed import LocalCluster
            # initiate cluster object
            self.cluster = LocalCluster()
            self.parallel = True
        else:
            self.parallel = False

        self.inps = inps
        self.work_dir = inps.work_dir
        self.phase_linking_method = inps.inversion_method
        self.range_window = int(inps.range_window)
        self.azimuth_window = int(inps.azimuth_window)
        self.patch_size = int(inps.patch_size)
        self.numWorker = int(inps.numWorker)
        self.config = inps.config
        self.out_dir = self.work_dir + '/inverted'
        os.makedirs(self.out_dir, exist_ok='True')

        self.shp_test = inps.shp_test
        self.shp_function = self.get_shp_function()

        # read input slcStack.h5
        self.slc_stack = inps.slc_stack  # slcStack.h5 file
        self.slcStackObj = slcStack(self.slc_stack)
        self.metadata = self.slcStackObj.get_metadata()
        self.all_date_list = self.slcStackObj.get_date_list()
        self.n_image, self.length, self.width = self.slcStackObj.get_size()

        # total number of neighbouring pixels
        self.shp_size = self.range_window * self.azimuth_window

        # threshold for shp test based on number of images to test
        self.distance_thresh = mut.ks_lut(self.n_image, self.n_image, alpha=0.01)

        # split the area in to patches of size 'self.patch_size'
        self.box_list, self.num_box = self.patch_slice

        # default number of images in each ministack
        self.mini_stack_default_size = 10
        if 'sequential' in self.phase_linking_method:
            self.total_num_mini_stacks = self.n_image // self.mini_stack_default_size
        else:
            self.total_num_mini_stacks = 1

        self.sample_rows, self.sample_cols, self.reference_row, self.reference_col = self.window_for_shp()
        self.total_num_mini_stacks = None
        self.temp_prefix = os.path.basename(os.path.dirname(self.work_dir))
        self.RSLCfile = '/tmp/{}_rslc_ref.h5'.format(self.temp_prefix)
        self.patch_slc_images = None

        if 'sequential' in self.phase_linking_method:
            self.sequential = True
        else:
            self.sequential = False

        return

    def get_shp_function(self):
        """
        Reads the shp testing function based on template file
        Returns: shp_function
        -------

        """
        if self.shp_test == 'ks':
            shp_function = mut.ks2smapletest
        elif self.shp_test == 'ad':
            shp_function = mut.ADtest
        elif self.shp_test == 'ttest':
            shp_function = mut.ttest_indtest
        else:  # default is KS 2 sample test
            shp_function = mut.ks2smapletest
        return shp_function

    def window_for_shp(self):
        """
        Shp window to be placed on each pixel
        Returns rows, cols, reference pixel row index, reference pixel col index
        -------

        """
        sample_rows = np.arange(-((self.azimuth_window - 1) / 2), ((self.azimuth_window - 1) / 2) + 1, dtype=int)
        reference_row = np.array([(self.azimuth_window - 1) / 2], dtype=int)

        sample_cols = np.arange(-((self.range_window - 1) / 2), ((self.range_window - 1) / 2) + 1, dtype=int)
        reference_col = np.array([(self.range_window - 1) / 2], dtype=int)

        return sample_rows, sample_cols, reference_row, reference_col

    @property
    def patch_slice(self):
        """
        Slice the image into patches of size patch_size
        box = (x0 y0 x1 y1) = (col0, row0, col1, row1) for each patch with respect to the whole image
        Returns box list, number of boxes
        -------

        """
        patch_row_1 = np.arange(0, self.length - self.azimuth_window, self.patch_size, dtype=int)
        patch_row_2 = patch_row_1 + self.patch_size
        patch_row_2[-1] = self.length

        patch_col_1 = np.arange(0, self.width - self.range_window, self.patch_size, dtype=int)
        patch_col_2 = patch_col_1 + self.patch_size
        patch_col_2[-1] = self.width
        num_box = len(patch_col_1) * len(patch_row_1)

        box_list = []
        for i in range(len(patch_row_1)):
            for j in range(len(patch_col_1)):
                box = (patch_col_1[j], patch_row_1[i], patch_col_2[j], patch_row_2[i])
                box_list.append(box)

        return box_list, num_box

    def initiate_output(self):

        RSLC = h5py.File(self.RSLCfile, 'a')
        if 'slc' in RSLC.keys():
            RSLC['slc'].resize(self.n_image, 0)
        else:
            self.metadata['FILE_TYPE'] = 'slc'
            for key, value in self.metadata.items():
                RSLC.attrs[key] = value

            RSLC.create_dataset('slc',
                                shape=(self.n_image, self.length, self.width),
                                maxshape=(None, self.length, self.width),
                                chunks=True,
                                dtype='complex64')

            RSLC.create_dataset('quality',
                                shape=(self.length, self.width),
                                maxshape=(self.length, self.width),
                                chunks=True,
                                dtype='float')

            RSLC['quality'][:, :] = -1

            # 1D dataset containing dates of all images
            dsName = 'dates'
            dsDataType = np.string_
            data = np.array(self.all_date_list, dtype=dsDataType)
            RSLC.create_dataset(dsName, data=data)

        RSLC.close()

        return

    def loop_patches(self):

        self.initiate_output()

        from mpi4py.futures import MPIPoolExecutor
        with MPIPoolExecutor() as executor:
            executor.map(self.process_patch, self.box_list)
        '''
        for i, box in enumerate(self.box_list):
            inputs = {'box': box,
                      'index': i}
            if self.parallel:
                self.run_dask(inputs)
            else:
                self.process_patch(inputs)
        '''
        return

    def process_patch(self, box): #inputs):

        big_box = None
        row1 = None
        row2 = None
        col1 = None
        col2 = None
        lin = None
        sam = None
        coord = None
        overlap_length = None
        overlap_width = None
        box_width = None
        box_length = None
        quality = None
        rslc_ref = None
        self.patch_slc_images = None

        #box = inputs['box']

        with h5py.File(self.RSLCfile, 'r') as f:
            quality = f['quality'][box[1]:box[3], box[0]:box[2]]

        if not np.any(quality < 0):
            return

        box_width = box[2] - box[0]
        box_length = box[3] - box[1]

        big_box = iut.get_big_box(box, self.range_window, self.azimuth_window, self.width, self.length)

        # In box coordinate
        row1 = box[1] - big_box[1]
        row2 = box[3] - big_box[1]
        col1 = box[0] - big_box[0]
        col2 = box[2] - big_box[0]

        lin = np.arange(row1, row2)
        overlap_length = len(lin)
        sam = np.arange(col1, col2)
        overlap_width = len(sam)
        lin, sam = np.meshgrid(lin, sam)
        coords = set(map(lambda y, x: (int(y), int(x)),
                         lin.T.reshape(overlap_length * overlap_width, 1),
                         sam.T.reshape(overlap_length * overlap_width, 1)))

        self.patch_slc_images = self.slcStackObj.read(datasetName='slc', box=big_box)
        rslc_ref = np.zeros([self.n_image, box_length, box_width], dtype='complex')

        results = map(self.invert_coord, coords)
        num_points = len(coords)

        block = [0, self.n_image, box[1], box[3], box[0], box[2]]

        #prog_bar = ptime.progressBar(maxValue=num_points)
        t = 0
        time0 = time.time()
        for result in results:
            rf = result['y'] - row1
            cf = result['x'] - col1
            rslc_ref[:, rf:rf+1, cf:cf+1] = result['rvector'].reshape(-1, 1, 1)
            quality[rf:rf+1, cf:cf+1] = result['quality']
            #Sprog_bar.update(t + 1, every=20, suffix='{}/{} pixels, box: {}/{}'.format(t + 1, num_points, inputs['index'] + 1, self.num_box))
            t += 1
        print('Total time for Box {}: {} s'.format(i, time.time() - time0))

        write_hdf5_block(fname=self.RSLCfile, data=rslc_ref, datasetName='slc', block=block)
        write_hdf5_block(fname=self.RSLCfile, data=quality, datasetName='quality', block=[box[1], box[3], box[0], box[2]])

        return

    def invert_coord(self, data):
        CCG = None
        result = {}

        # big box coordinate:
        shp, row0, col0 = self.get_shp_row_col(data, self.patch_slc_images)

        num_shp = len(shp[shp > 0])
        shp_rows, shp_cols = np.nonzero(shp)
        shp_rows = np.array(shp_rows + row0).astype(int)
        shp_cols = np.array(shp_cols + col0).astype(int)

        CCG = np.array(self.patch_slc_images[:, shp_rows, shp_cols])
        coh_mat = mut.est_corr(CCG)

        squeezed_images = None

        if num_shp > 20:

            if self.sequential:
                vec_refined, squeezed_images = iut.sequential_phase_linking(CCG, self.phase_linking_method, 10, self.total_num_mini_stacks)

            else:
                vec_refined = mut.phase_linking_process(CCG, 0, self.phase_linking_method, squeez=False)

        else:
            vec_refined = mut.test_PS(coh_mat)

        amp_refined = np.array(np.mean(np.abs(CCG), axis=1)).reshape(-1, 1)
        vec_refined /= np.abs(vec_refined)
        vec_refined *= amp_refined

        if self.sequential and num_shp > 20:
            vec_refined = iut.datum_connect(squeezed_images, vec_refined, self.default_mini_stack_size)

        result['x'] = data[1]
        result['y'] = data[0]
        result['rvector'] = vec_refined
        result['quality'] = mut.gam_pta(np.angle(coh_mat), vec_refined)

        return result

    def get_shp_row_col(self, data, input_slc):

        row_0 = data[0]
        col_0 = data[1]
        length = input_slc.shape[1]
        width = input_slc.shape[2]

        sample_rows = row_0 + self.sample_rows
        sample_rows[sample_rows < 0] = -1
        sample_rows[sample_rows >= length] = -1

        sample_cols = col_0 + self.sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= width] = -1

        sample_cols = sample_cols[np.flatnonzero(sample_cols >= 0)]
        sample_rows = sample_rows[np.flatnonzero(sample_rows >= 0)]

        ref_row = self.reference_row - self.azimuth_window + len(sample_rows)
        ref_col = self.reference_col - self.range_window + len(sample_cols)

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        rslc = input_slc[:, y, x].reshape(self.n_image, -1)
        testvec = np.sort(np.abs(rslc), axis=0)
        S1 = np.sort(np.abs(input_slc[:, row_0, col_0])).reshape(self.n_image, 1)

        data1 = np.repeat(S1, testvec.shape[1], axis=1)
        data_all = np.concatenate((data1, testvec), axis=0)

        res = 1 * (np.apply_along_axis(mut.ecdf_distance, 0, data_all) <= self.distance_thresh)
        res = res.reshape(len(sample_rows), len(sample_cols))
        ks_label = label(res, background=0, connectivity=2)
        ksres = 1 * (ks_label == ks_label[ref_row, ref_col])

        return ksres, sample_rows[0], sample_cols[0]

    def run_dask(self, inputs):
        from dask.distributed import Client, as_completed

        client = Client(self.cluster)
        box = inputs['box']
        sub_boxes = self.split_box2sub_boxes(box, num_split=self.numWorker)

        submission_time = time.time()
        futures = []

        for i, sub_box in enumerate(sub_boxes):
            print('submit a job to the worker for sub box {}: {}'.format(i, sub_box))
            inputs['box'] = sub_box

            future = client.submit(self.process_patch, inputs, retries=3)
            futures.append(future)

        num_future = 0
        for future in as_completed(futures, with_results=True):
            num_future += 1
            sub_t = time.time() - submission_time
            print("FUTURE #{} complete. Time used: {:.0f} seconds".format(num_future, sub_t))

        self.cluster.close()
        client.close()
        return

    def move_dask_stdout_stderr_files(self):
        """Move *o and *e files produced by dask into stdout and sderr directory"""

        stdout_files = glob.glob('*.o')
        stderr_files = glob.glob('*.e')
        job_files = glob.glob('dask_command_run_from_python.txt*')

        if len(stdout_files + stderr_files + job_files) == 0:
            return

        stdout_folder = 'stdout_dask'
        stderr_folder = 'stderr_dask'
        for std_dir in [stdout_folder, stderr_folder]:
            if os.path.isdir(std_dir):
                shutil.rmtree(std_dir)
            os.mkdir(std_dir)

        for item in stdout_files + job_files:
            shutil.move(item, stdout_folder)

        for item in stderr_files:
            shutil.move(item, stderr_folder)

    def split_box2sub_boxes(self, box, num_split):
        """Divide the input box into `num_split` different sub_boxes.
        :param box: [x0, y0, x1, y1]: list[int] of size 4
        :param num_split: int, the number of sub_boxes to split a box into
        """

        x0, y0, x1, y1 = box
        length, width = y1 - y0, x1 - x0
        num_split_x = int(np.sqrt(num_split))
        num_split_y = int(np.sqrt(num_split))

        sub_boxes = []

        if (length // num_split_y) < 10:
            num_split_y = length // 10
        if (width // num_split_x) < 10:
            num_split_x = width // (2 * self.range_window)

        for i in range(num_split_y):
            start_y = (i * length) // num_split_y + y0
            end_y = ((i + 1) * length) // num_split_y + y0
            if i == num_split_y - 1:
                end_y = y1
            for j in range(num_split_x):
                start_x = (j * width) // num_split_x + x0
                end_x = ((j + 1) * width) // num_split_x + x0
                if j == num_split_x - 1:
                    end_x = x1
                sub_boxes.append([start_x, start_y, end_x, end_y])

        return sub_boxes


    def close(self):
        if os.path.exists(self.RSLCfile):

            with h5py.File(self.RSLCfile, 'a') as f:
                for d, date in enumerate(self.all_date_list):
                    wrap_date = os.path.join(self.out_dir, 'wrapped_phase', date)
                    os.makedirs(wrap_date, exist_ok=True)
                    out_name = os.path.join(wrap_date, date + '.slc')
                    if not os.path.exists(out_name):
                        out_rslc = np.memmap(out_name, dtype='complex64', mode='w+', shape=(self.length, self.width))
                        out_rslc[:, :] = f['slc'][d, :, :]
                        IML.renderISCEXML(out_name, bands=1, nyy=self.length, nxx=self.width, datatype='complex64', scheme='BSQ')
                    else:
                        IML.renderISCEXML(out_name, bands=1, nyy=self.length, nxx=self.width, datatype='complex64', scheme='BSQ')

                quality_file = self.out_dir + '/quality'
                if not os.path.exists(quality_file):
                    quality_memmap = np.memmap(quality_file, mode='write', dtype='float32', shape=(self.length, self.width))
                    IML.renderISCEXML(quality_file, bands=1, nyy=self.length, nxx=self.width, datatype='float32',
                                      scheme='BIL')
                else:
                    quality_memmap = np.memmap(quality_file, mode='r+', dtype='float32', shape=(self.length, self.width))

                quality_memmap[:, :] = f['quality']
                quality_memmap = None

            command = 'mv {} {}'.format(self.RSLCfile, self.out_dir + '/rslc_ref.h5')
            os.system(command)

        return


#################################################


if __name__ == '__main__':
    main()
