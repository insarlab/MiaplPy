#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import minopy_utilities as mut
from skimage.measure import label
from minopy.objects.arg_parser import MinoPyParser
import h5py
from minopy.objects import cluster_minopy
from mintpy.utils import ptime
from isceobj.Util.ImageUtil import ImageLib as IML
import gdal
from minopy.objects.slcStack import slcStack
import minopy.objects.inversion_utils as iut
#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    # --cluster and --num-worker option
    inps.numWorker = str(cluster_minopy.cluster.DaskCluster.format_num_worker(inps.cluster, inps.numWorker))
    if inps.cluster != 'no' and inps.numWorker == '1':
        print('WARNING: number of workers is 1, turn OFF parallel processing and continue')
        inps.cluster = 'no'

    inversionObj = PhaseLink(inps)

    # Phase linking inversion:
    if not inversionObj.start_index is None:
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
        self.cluster = inps.cluster
        self.numWorker = inps.numWorker
        self.config = inps.config
        self.out_dir = self.work_dir + '/inverted'
        os.makedirs(self.out_dir, exist_ok='True')

        self.shp_test = inps.shp_test
        self.shp_function = self.get_shp_function()

        # read input slcStack.h5
        self.slc_stack = inps.slc_stack              # slcStack.h5 file
        self.inverted_date_list_file = self.out_dir + '/inverted_date_list.txt'
        self.start_index, self.all_date_list, self.metadata, self.n_image, self.length, self.width = \
            self.update_inversion_test()

        if self.start_index:
            self.n_image -= self.start_index         # number of images to invert

        self.shp_size = self.range_window * self.azimuth_window
        self.distance_thresh = mut.ks_lut(self.n_image, self.n_image, alpha=0.01)
        self.box_list, self.num_box = self.patch_slice()

        self.mini_stack_default_size = 10
        self.new_num_mini_stacks = self.n_image // self.mini_stack_default_size

        self.total_mini_stack_slc_size = self.initiate_datum()

        return

    def update_inversion_test(self):

        slcStackObj = slcStack(self.slc_stack)
        metadata = slcStackObj.get_metadata()
        date_list = slcStackObj.get_date_list()
        numSlc, length, width = slcStackObj.get_size()
        print('Total number of pixels {}'.format(length * width))

        if not os.path.exists(self.inverted_date_list_file):
            start_index = 0
            all_date_list = date_list
        else:
            start_index, all_date_list = mut.update_or_skip_inversion(self.inverted_date_list_file, date_list)

        return start_index, all_date_list, metadata, numSlc, length, width

    def get_shp_function(self):
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

    def initiate_datum(self):
        datum_file = self.out_dir + '/datum.h5'
        ds = h5py.File(datum_file, 'a')
        if 'squeezed_images' in ds.keys():
            old_squeezed_images = ds['squeezed_images']
            old_datum_shift = ds['datum_shift']
            old_num_mini_stacks = old_squeezed_images.shape[0]
            old_mini_stack_slc_size = ds['miniStack_size'][:]
        else:
            old_num_mini_stacks = 0
            old_mini_stack_slc_size = None

        total_num_mini_stacks = self.new_num_mini_stacks + old_num_mini_stacks
        total_mini_stack_slc_size = np.zeros([total_num_mini_stacks, 1])

        if not old_mini_stack_slc_size is None:
            total_mini_stack_slc_size[0: old_num_mini_stacks] = old_mini_stack_slc_size[:]

        for sstep in range(old_num_mini_stacks, total_num_mini_stacks):
            first_line = sstep * self.mini_stack_default_size
            if sstep == total_num_mini_stacks - 1:
                last_line = len(self.all_date_list)
            else:
                last_line = first_line + self.mini_stack_default_size
            num_lines = last_line - first_line
            total_mini_stack_slc_size[sstep, 0] = num_lines

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
            temp_squeezed_image[0:old_num_mini_stacks, :, :] = old_squeezed_images[:, :, :]
            temp_datum_shift[0:old_num_mini_stacks, :, :] = old_datum_shift[:, :, :]

        ds.close()
        del temp_squeezed_image, temp_datum_shift
        return total_mini_stack_slc_size

    def iterate_coords(self):

        time0 = time.time()
        sample_rows, sample_cols, reference_row, reference_col = self.window_for_shp()
        n_inverted = len(self.all_date_list) - self.n_image  # number of images that are already inverted

        data_kwargs = {
            "slc_stack_file": self.slc_stack,
            "distance_thresh": self.distance_thresh,
            "azimuth_window": self.azimuth_window,
            "range_window": self.range_window,
            "phase_linking_method": self.phase_linking_method,
            "total_mini_stack_slc_size": self.total_mini_stack_slc_size,
            "samples": {"sample_rows": sample_rows, "sample_cols": sample_cols,
                        "reference_row": reference_row, "reference_col": reference_col},
            "slc_start_index": self.start_index,
            "total_num_images": len(self.all_date_list),
            "num_inverted_images": n_inverted
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

            patch_dir = os.path.join(self.out_dir, 'PATCH_{}'.format(i))
            iut.initiate_stacks(patch_dir, box, n_inverted, len(self.all_date_list), self.shp_size)
            data_kwargs['patch_length'] = box_length
            data_kwargs['patch_width'] = box_width
            data_kwargs['patch_dir'] = patch_dir
            data_kwargs['box'] = box

            #self.cluster = 'no'
            if self.cluster == 'no':
                iut.inversion(**data_kwargs)
            else:
                # parallel
                print('\n\n------- start parallel processing using Dask -------')

                # initiate dask cluster and client
                cluster_obj = cluster_minopy.MDaskCluster(self.cluster, self.numWorker, config_name=self.config)
                cluster_obj.open()

                cluster_obj.run(iut.inversion, func_data=data_kwargs)

                # close dask cluster and client
                cluster_obj.close()

                print('------- finished parallel processing -------\n\n')

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))
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


#################################################



if __name__ == '__main__':
    main()

