#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import h5py
import shutil
import pickle

import minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
#from minopy.objects import cluster_minopy
from mintpy.utils import ptime
from minopy.objects.slcStack import slcStack
import minopy.objects.inversion_utils as iut
from isceobj.Util.ImageUtil import ImageLib as IML

from minsar.job_submission import JOB_SUBMIT
import minsar.utils.process_utilities as putils
#################################


def main(iargs=None):
    '''
        Phase linking process.
    '''

    Parser = MinoPyParser(iargs, script='phase_inversion')
    inps = Parser.parse()

    # --cluster and --num-worker option
    #inps.numWorker = str(cluster_minopy.cluster.DaskCluster.format_num_worker(inps.cluster, inps.numWorker))
    #if inps.cluster != 'no' and inps.numWorker == '1':
    #    print('WARNING: number of workers is 1, turn OFF parallel processing and continue')
    #    inps.cluster = 'no'

    inversionObj = PhaseLink(inps)
    inversionObj.iterate_coords()
    inversionObj.close()

    # Phase linking inversion:
    #if not inversionObj.start_index is None:
    #    inversionObj.iterate_coords()
    #    inversionObj.close()

    return None


class PhaseLink:
    def __init__(self, inps):
        self.inps = inps
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
        #reference_row = reference_row - (self.azimuth_window - len(sample_rows))

        sample_cols = np.ogrid[-((self.range_window - 1) / 2):((self.range_window - 1) / 2) + 1]
        sample_cols = sample_cols.astype(int)
        reference_col = np.array([(self.range_window - 1) / 2]).astype(int)
        #reference_col = reference_col - (self.range_window - len(sample_cols))
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

        total_num_mini_stacks = len(self.all_date_list) // self.mini_stack_default_size
        #self.new_num_mini_stacks + old_num_mini_stacks
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
            IML.renderISCEXML(temp_squeezed_images_file, bands=total_num_mini_stacks, nyy=self.length,
                              nxx=self.width, datatype='complex64', scheme='BSQ')


        if not os.path.exists(temp_datum_shift_file):
            temp_datum_shift = np.memmap(temp_datum_shift_file, dtype='float32', mode='write',
                                            shape=(total_num_mini_stacks, self.length, self.width))
        else:
            temp_datum_shift = np.memmap(temp_datum_shift_file, dtype='float32', mode='r+',
                                            shape=(total_num_mini_stacks, self.length, self.width))

            IML.renderISCEXML(temp_squeezed_images_file, bands=total_num_mini_stacks, nyy=self.length,
                              nxx=self.width, datatype='float32', scheme='BSQ')

        if not old_mini_stack_slc_size is None:
            for img in range(old_num_mini_stacks):
                temp_squeezed_image[img, :, :] = old_squeezed_images[img, :, :]
                temp_datum_shift[img, :, :] = old_datum_shift[img, :, :]

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
            "total_mini_stack_slc_size": self.total_mini_stack_slc_size.flatten(),
            "samples": {"sample_rows": sample_rows, "sample_cols": sample_cols,
                        "reference_row": reference_row, "reference_col": reference_col},
            "slc_start_index": self.start_index,
            "total_num_images": len(self.all_date_list),
            "num_inverted_images": n_inverted
        }

        if 'sequential' in self.phase_linking_method:
            data_kwargs['new_num_mini_stacks'] = self.new_num_mini_stacks

        run_commands = []

        # invert / write block-by-block
        for i, box in enumerate(self.box_list):
            box_width = box[2] - box[0]
            box_length = box[3] - box[1]
            if self.num_box > 1:
                print('\n------- processing patch {} out of {} --------------'.format(i + 1, self.num_box))
                print('box width:  {}'.format(box_width))
                print('box length: {}'.format(box_length))

            patch_dir = os.path.join(self.out_dir, 'PATCH_{}'.format(i))

            if os.path.exists(patch_dir + '/flag.npy'):
                continue

            iut.initiate_stacks(patch_dir, box, n_inverted, len(self.all_date_list), self.shp_size)
            data_kwargs['patch_length'] = box_length
            data_kwargs['patch_width'] = box_width
            data_kwargs['patch_dir'] = patch_dir
            data_kwargs['box'] = box
            data_kwargs['patch_row_0'] = box[1]
            data_kwargs['patch_col_0'] = box[0]

            args_file = patch_dir + '/data_kwargs.pkl'
            with open(args_file, 'wb') as handle:
                pickle.dump(data_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

            cmd = 'patch_invert.py --dataArg {a1} --cluster {a2} --num-worker {a3} --config-name {a4}\n'.format(
                a1=args_file, a2=self.cluster, a3=self.numWorker, a4=self.config)
            run_commands.append(cmd)

        if len(run_commands) > 0:
            run_dir = self.work_dir + '/run_file'
            os.makedirs(run_dir, exist_ok=True)
            run_file_inversion = os.path.join(run_dir, 'run_minopy_inversion')
            with open(run_file_inversion, 'w+') as f:
                f.writelines(run_commands)

            inps_args = self.inps
            inps_args.work_dir = run_dir
            inps_args.out_dir = run_dir
            job_obj = JOB_SUBMIT(inps_args)

            putils.remove_last_job_running_products(run_file=run_file_inversion)
            job_status = job_obj.submit_batch_jobs(batch_file=run_file_inversion)
            if job_status:
                putils.remove_zero_size_or_length_error_files(run_file=run_file_inversion)
                putils.rerun_job_if_exit_code_140(run_file=run_file_inversion, inps_dict=inps_args)
                putils.raise_exception_if_job_exited(run_file=run_file_inversion)
                putils.concatenate_error_files(run_file=run_file_inversion, work_dir=inps_args.work_dir)
                putils.move_out_job_files_to_stdout(run_file=run_file_inversion)

            '''
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
            '''

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))
        return

    def close(self):

        with open(self.out_dir + '/inverted_date_list.txt', 'w+') as f:
            dates = [date + '\n' for date in self.all_date_list]
            f.writelines(dates)

        if 'sequential' in self.phase_linking_method:

            datum_file = os.path.join(self.out_dir, 'datum.h5')
            squeezed_image_file = os.path.join(self.out_dir, 'squeezed_images')
            datum_shift_file = os.path.join(self.out_dir, 'datum_shift')

            squeezed_images_memmap = np.memmap(squeezed_image_file, dtype='complex64', mode='r',
                                        shape=(len(self.total_mini_stack_slc_size), self.length, self.width))
            datum_shift_memmap = np.memmap(datum_shift_file, dtype='float32', mode='r',
                                        shape=(len(self.total_mini_stack_slc_size), self.length, self.width))

            with h5py.File(datum_file, 'a') as ds:
                if 'squeezed_images' in ds.keys():
                    ds['squeezed_images'].resize(len(self.total_mini_stack_slc_size), 0)
                    ds['datum_shift'].resize(len(self.total_mini_stack_slc_size), 0)
                    ds['miniStack_size'].resize(len(self.total_mini_stack_slc_size), 0)
                else:
                    ds.create_dataset('squeezed_images',
                                      shape=(len(self.total_mini_stack_slc_size), self.length, self.width),
                                      maxshape=(None, self.length, self.width),
                                      chunks=True,
                                      dtype='complex64')
                    ds.create_dataset('datum_shift',
                                      shape=(len(self.total_mini_stack_slc_size), self.length, self.width),
                                      maxshape=(None, self.length, self.width),
                                      chunks=True,
                                      dtype='float32')
                    ds.create_dataset('miniStack_size',
                                      shape=(len(self.total_mini_stack_slc_size), 1),
                                      maxshape=(None, 1),
                                      chunks=True,
                                      dtype='float32')

                ds['squeezed_images'][:, :, :] = squeezed_images_memmap[:, :, :]
                ds['datum_shift'][:, :, :] = datum_shift_memmap[:, :, :]
                ds['miniStack_size'][:] = self.total_mini_stack_slc_size[:]

                self.metadata['FILE_TYPE'] = 'datum'
                for key, value in self.metadata.items():
                    ds.attrs[key] = value

        quality_file = self.out_dir + '/quality'
        if not os.path.exists(quality_file):
            quality_memmap = np.memmap(quality_file, mode='write', dtype='float32', shape=(self.length, self.width))
            IML.renderISCEXML(quality_file, bands=1, nyy=self.length, nxx=self.width, datatype='float32', scheme='BIL')
        else:
            quality_memmap = np.memmap(quality_file, mode='r+', dtype='float32', shape=(self.length, self.width))

        RSLCfile = self.out_dir + '/rslc_ref.h5'
        RSLC = h5py.File(RSLCfile, 'a')
        if 'slc' in RSLC.keys():
            RSLC['slc'].resize(len(self.all_date_list), 0)
            shp_update = False
        else:
            self.metadata['FILE_TYPE'] = 'slc'
            for key, value in self.metadata.items():
                RSLC.attrs[key] = value
            RSLC.create_dataset('slc',
                              shape=(len(self.all_date_list), self.length, self.width),
                              maxshape=(None, self.length, self.width),
                              chunks=True,
                              dtype='complex64')

            RSLC.create_dataset('shp',
                              shape=(self.shp_size, self.length, self.width),
                              maxshape=(None, self.length, self.width),
                              chunks=True,
                              dtype='complex64')
            shp_update = True
        IML.renderISCEXML(RSLCfile, bands=len(self.all_date_list), nyy=self.length, nxx=self.width,
                          datatype='complex64', scheme='BSQ')
        '''
        patch_list = []
        prog_bar = ptime.progressBar(maxValue=len(self.box_list))
        i = 0
        for i, box in enumerate(self.box_list):
            patch_dir = self.out_dir + '/PATCH_{}'.format(i)
            patch_list.append(patch_dir)
            box_length = box[3] - box[1]
            box_width = box[2] - box[0]
            rslc_ref = np.memmap(patch_dir + '/rslc_ref', mode='r+', dtype='complex64',
                                 shape=(len(self.all_date_list), box_length, box_width))
            quality = np.memmap(patch_dir + '/quality', mode='r+', dtype='float32', shape=(box_length, box_width))

            RSLC['slc'][:, box[1]:box[3], box[0]:box[2]] = rslc_ref[:, :, :]
            quality_memmap[box[1]:box[3], box[0]:box[2]] = quality[:, :]

            if shp_update:
                shp = np.memmap(patch_dir + '/shp', mode='r+', dtype='float32',
                                shape=(self.shp_size, box_length, box_width))
                RSLC['shp'][:, box[1]:box[3], box[0]:box[2]] = shp[:, :, :]

            prog_bar.update(i + 1, every=10, suffix='{}/{} pixels'.format(i + 1, len(self.box_list)))
            i += 1
        '''

        RSLC.close()

        #remove_directories = [squeezed_image_file, datum_shift_file] + patch_list
        #for item in remove_directories:
        #    if os.path.exists(item):
        #        shutil.rmtree(item)
        return


#################################################

if __name__ == '__main__':
    main()

