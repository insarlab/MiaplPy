#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import time
import numpy as np
import h5py
import pickle

import minopy_utilities as mut
from minopy.objects.arg_parser import MinoPyParser
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

    inversionObj = PhaseLink(inps)
    inversionObj.iterate_coords()
    inversionObj.close()

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
        slcStackObj = slcStack(self.slc_stack)
        self.metadata = slcStackObj.get_metadata()
        self.all_date_list = slcStackObj.get_date_list()
        self.n_image, self.length, self.width = slcStackObj.get_size()

        # total number of neighbouring pixels
        self.shp_size = self.range_window * self.azimuth_window

        # threshold for shp test based on number of images to test
        self.distance_thresh = mut.ks_lut(self.n_image, self.n_image, alpha=0.01)

        # split the area in to patches of size 'self.patch_size'
        self.box_list, self.num_box = self.patch_slice()
        
        # default number of images in each ministack
        self.mini_stack_default_size = 10

        self.total_num_mini_stacks = None

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
        sample_rows = np.ogrid[-((self.azimuth_window - 1) / 2):((self.azimuth_window - 1) / 2) + 1]
        sample_rows = sample_rows.astype(int)
        reference_row = np.array([(self.azimuth_window - 1) / 2]).astype(int)

        sample_cols = np.ogrid[-((self.range_window - 1) / 2):((self.range_window - 1) / 2) + 1]
        sample_cols = sample_cols.astype(int)
        reference_col = np.array([(self.range_window - 1) / 2]).astype(int)

        return sample_rows, sample_cols, reference_row, reference_col

    def patch_slice(self):
        """
        Slice the image into patches of size patch_size
        box = (x0 y0 x1 y1) = (col0, row0, col1, row1) for each patch with respect to the whole image
        Returns box list, number of boxes
        -------

        """
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

    def iterate_coords(self):
        """
        Calculates inversion parameters, writes a run file containing inversion tasks and each task is a patch
        then submits the run file as a batch job
        results will be saved in each patch folder separately
        Returns
        -------

        """

        time0 = time.time()
        # get the shp moving window
        sample_rows, sample_cols, reference_row, reference_col = self.window_for_shp()

        # Inversion input arguments
        data_kwargs = {
            "distance_thresh": self.distance_thresh,
            "azimuth_window": self.azimuth_window,
            "range_window": self.range_window,
            "phase_linking_method": self.phase_linking_method,
            "default_mini_stack_size": self.mini_stack_default_size,
            "samples": {"sample_rows": sample_rows, "sample_cols": sample_cols,
                        "reference_row": reference_row, "reference_col": reference_col},
        }

        # write each box as a separate task in a run file
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

            big_box, self.total_num_mini_stacks = iut.initiate_stacks(self.slc_stack, patch_dir, box,
                                                                 self.mini_stack_default_size,
                                                                 self.phase_linking_method, self.shp_size,
                                                                 self.range_window, self.azimuth_window)
            data_kwargs['patch_length'] = box_length
            data_kwargs['patch_width'] = box_width
            data_kwargs['patch_dir'] = patch_dir
            data_kwargs['box'] = box
            data_kwargs['patch_row_0'] = box[1]
            data_kwargs['patch_col_0'] = box[0]
            data_kwargs['big_box'] = big_box
            data_kwargs['total_num_mini_stacks'] = self.total_num_mini_stacks

            args_file = patch_dir + '/data_kwargs.pkl'
            with open(args_file, 'wb') as handle:
                pickle.dump(data_kwargs, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # The inversion command for each box
            cmd = 'patch_invert.py --dataArg {a1} --cluster {a2} --num-worker {a3} --config-name {a4}\n'.format(
                a1=args_file, a2=self.cluster, a3=self.numWorker, a4=self.config)

            run_commands.append(cmd)

        if len(run_commands) > 0:
            run_dir = self.work_dir + '/run_files'
            os.makedirs(run_dir, exist_ok=True)
            run_file_inversion = os.path.join(run_dir, 'run_minopy_inversion')
            with open(run_file_inversion, 'w+') as f:
                f.writelines(run_commands)

            if not os.getenv('JOB_SUBMISSION_SCHEME'):
                os.environ['JOB_SUBMISSION_SCHEME'] = 'launcher_multiTask_multiNode'
            inps_args = self.inps
            inps_args.work_dir = run_dir
            inps_args.out_dir = run_dir
            inps_args.num_bursts = (self.patch_size**2)/40000
            job_obj = JOB_SUBMIT(inps_args)

            putils.remove_last_job_running_products(run_file=run_file_inversion)
            job_obj.write_batch_jobs(batch_file=run_file_inversion)
            job_status = job_obj.submit_batch_jobs(batch_file=run_file_inversion)
            if job_status:
                putils.remove_zero_size_or_length_error_files(run_file=run_file_inversion)
                putils.rerun_job_if_exit_code_140(run_file=run_file_inversion, inps_dict=inps_args)
                putils.raise_exception_if_job_exited(run_file=run_file_inversion)
                putils.concatenate_error_files(run_file=run_file_inversion, work_dir=inps_args.work_dir)
                putils.move_out_job_files_to_stdout(run_file=run_file_inversion)

        timep = time.time() - time0
        print('time spent to do phase inversion {}: min'.format(timep / 60))
        return

    def close(self):

        print('All patches are done, stitching ...')

        with open(self.out_dir + '/inverted_date_list.txt', 'w+') as f:
            dates = [date + '\n' for date in self.all_date_list]
            f.writelines(dates)

        print('Stitching wrapped phase time series...')
        merge_patches(width=self.width, length=self.length, inverted_dir=self.out_dir,
                          box_list=self.box_list, date_list=self.all_date_list)

        print('Stitching temporal coherence to quality...')

        # Save wrapped phase time series to .h5 and quality
        quality_file = self.out_dir + '/quality'
        if not os.path.exists(quality_file):
            quality_memmap = np.memmap(quality_file, mode='write', dtype='float32', shape=(self.length, self.width))
            IML.renderISCEXML(quality_file, bands=1, nyy=self.length, nxx=self.width, datatype='float32', scheme='BIL')
        else:
            quality_memmap = np.memmap(quality_file, mode='r+', dtype='float32', shape=(self.length, self.width))

        print('Export wrapped phase time series to rslc_ref.h5 ...')
        RSLCfile = self.out_dir + '/rslc_ref.h5'
        RSLC = h5py.File(RSLCfile, 'a')
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

            RSLC.create_dataset('shp',
                                shape=(self.shp_size, self.length, self.width),
                                maxshape=(None, self.length, self.width),
                                chunks=True,
                                dtype='byte')

        print('unpatch wrapped phase time series and quality')
        prog_bar = ptime.progressBar(maxValue=len(self.box_list))
        t = 0
        for i, box in enumerate(self.box_list):
            patch_dir = os.path.join(self.out_dir, 'PATCH_{}'.format(i))
            length = box[3] - box[1]
            width = box[2] - box[0]

            rslc_memmap_patch = np.memmap(patch_dir + '/rslc_ref', dtype='complex64', mode='r',
                                    shape=(self.n_image, length, width))
            RSLC['slc'][:, box[1]:box[3], box[0]:box[2]] = rslc_memmap_patch[:, :, :]

            shp_memmap_patch = np.memmap(patch_dir + '/shp', dtype='byte', mode='r',
                                    shape=(self.shp_size, length, width))
            RSLC['shp'][:, box[1]:box[3], box[0]:box[2]] = shp_memmap_patch[:, :, :]

            quality_memmap_patch = np.memmap(patch_dir + '/quality', dtype='float32', mode='r', shape=(length, width))
            quality_memmap[box[1]:box[3], box[0]:box[2]] = quality_memmap_patch[:, :]

            prog_bar.update(t + 1, every=10, suffix='{}/{} pixels'.format(t + 1, len(self.box_list)))
            t += 1
        RSLC.close()

        # Save squeezed images and datum shift to .h5
        if 'sequential' in self.phase_linking_method:

            print('Export Squeezed images and Datum shifts to datum.h5 ...')
            datum_file = os.path.join(self.out_dir, 'datum.h5')
            if self.total_num_mini_stacks is None:
                self.total_num_mini_stacks = self.n_image//self.mini_stack_default_size

            with h5py.File(datum_file, 'a') as ds:
                if not 'squeezed_images' in ds.keys():

                    ds.create_dataset('squeezed_images',
                                      shape=(self.total_num_mini_stacks, self.length, self.width),
                                      maxshape=(None, self.length, self.width),
                                      chunks=True,
                                      dtype='complex64')
                    ds.create_dataset('datum_shift',
                                      shape=(self.total_num_mini_stacks, self.length, self.width),
                                      maxshape=(None, self.length, self.width),
                                      chunks=True,
                                      dtype='float32')
                    self.metadata['FILE_TYPE'] = 'datum'
                    for key, value in self.metadata.items():
                        ds.attrs[key] = value

                print('unpatch squeezed images and datum shift')
                prog_bar = ptime.progressBar(maxValue=len(self.box_list))
                t = 0
                for i, box in enumerate(self.box_list):
                    patch_dir = os.path.join(self.out_dir, 'PATCH_{}'.format(i))
                    length = box[3] - box[1]
                    width = box[2] - box[0]
                    squeezed_images_memmap = np.memmap(patch_dir + '/squeezed_images', dtype='complex64', mode='r',
                                                shape=(self.total_num_mini_stacks, length, width))
                    ds['squeezed_images'][:, box[1]:box[3], box[0]:box[2]] = squeezed_images_memmap[:, :, :]

                    datum_shift_memmap = np.memmap(patch_dir + '/datum_shift', dtype='float32', mode='r',
                                                   shape=(self.total_num_mini_stacks, length, width))
                    ds['datum_shift'][:, box[1]:box[3], box[0]:box[2]] = datum_shift_memmap[:, :, :]

                    prog_bar.update(t + 1, every=10, suffix='{}/{} pixels'.format(t + 1, len(self.box_list)))
                    t += 1

        return


def merge_patches(width, length, inverted_dir, box_list, date_list):
    out_dir = os.path.join(inverted_dir, 'wrapped_phase')
    os.makedirs(out_dir, exist_ok=True)

    patches = {'PATCH_0': None}

    for t, box in enumerate(box_list):
        patch_name = 'PATCH_{}'.format(t)
        patch_dir = os.path.join(inverted_dir, patch_name)
        patch_file = os.path.join(patch_dir, 'rslc_ref')
        patch_width = box[2] - box[0]
        patch_length = box[3] - box[1]
        patch_rslc = np.memmap(patch_file, dtype='complex64', mode='r', shape=(len(date_list),
                                                                               patch_length, patch_width))
        patches[patch_name] = patch_rslc

    for d, date in enumerate(date_list):
        wrap_date = os.path.join(out_dir, date)
        os.makedirs(wrap_date, exist_ok=True)
        out_name = os.path.join(wrap_date, date + '.slc')
        if not os.path.exists(out_name):
            out_rslc = np.memmap(out_name, dtype='complex64', mode='w+', shape=(length, width))
            for t, box in enumerate(box_list):
                patch_name = 'PATCH_{}'.format(t)
                out_rslc[box[1]:box[3], box[0]:box[2]] = patches[patch_name][d, :, :]
            IML.renderISCEXML(out_name, bands=1, nyy=length, nxx=width, datatype='complex64', scheme='BSQ')

    return

#################################################

if __name__ == '__main__':
    main()

