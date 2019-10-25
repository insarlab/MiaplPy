#! /usr/bin/env python3
###############################################################################
# Project: Utilities for minopy
# Author: Sara Mirzaee
###############################################################################
import os
import shutil
import glob
from natsort import natsorted
from mintpy.objects.coord import coordinate
from minopy.prep_slc_isce import read_attribute
from minopy.minopy_utilities import read_image
import h5py
from mintpy.utils.readfile import standardize_metadata
from minopy.objects.slcStack import slcStack
from mintpy.objects import (
    datasetUnitDict,
    geometry,
    geometryDatasetNames,
    giantIfgramStack,
    giantTimeseries,
    ifgramDatasetNames,
    ifgramStack,
    timeseriesDatasetNames,
    timeseries,
    HDFEOS
)



class OutControl:
    def __init__(self, batch_file, work_dir):
        self.run_file = os.path.abspath(batch_file)
        self.work_dir = work_dir

    def clean(self):
        self.remove_zero_size_or_length_error_files()
        self.raise_exception_if_job_exited()
        self.concatenate_error_files()
        self.move_out_job_files_to_stdout()

    def remove_last_job_running_products(self):
        error_files = glob.glob(self.run_file + '*.e')
        job_files = glob.glob(self.run_file + '*.job')
        out_file = glob.glob(self.run_file + '*.o')
        list_files = error_files + out_file + job_files
        if not len(list_files) == 0:
            for item in list_files:
                os.remove(item)
        return

    def remove_zero_size_or_length_error_files(self):
        """Removes files with zero size or zero length (*.e files in run_files)."""

        error_files = glob.glob(self.run_file + '*.e')
        error_files = natsorted(error_files)
        for item in error_files:
            if os.path.getsize(item) == 0:  # remove zero-size files
                os.remove(item)
            elif file_len(item) == 0:
                os.remove(item)  # remove zero-line files
        return

    def raise_exception_if_job_exited(self):
        """Removes files with zero size or zero length (*.e files in run_files)."""

        files = glob.glob(self.run_file + '*.o')

        # need to add for PBS. search_string='Terminated'
        search_string = 'Exited with exit code'

        files = natsorted(files)
        for file in files:
            with open(file) as fr:
                lines = fr.readlines()
                for line in lines:
                    if search_string in line:
                        raise Exception("ERROR: {0} exited; contains: {1}".format(file, line))
        return

    def concatenate_error_files(self):
        """
        Concatenate error files to one file (*.e files in run_files).
        :param directory: str
        :param out_name: str
        :return: None
        """

        out_file = os.path.abspath(self.work_dir) + '/out_' + self.run_file.split('/')[-1] + '.e'
        if os.path.isfile(out_file):
            os.remove(out_file)

        out_name = os.path.dirname(self.run_file) + '/out_' + self.run_file.split('/')[-1] + '.e'
        error_files = glob.glob(self.run_file + '*.e')
        if not len(error_files) == 0:
            with open(out_name, 'w') as outfile:
                for fname in error_files:
                    outfile.write('#########################\n')
                    outfile.write('#### ' + fname + ' \n')
                    outfile.write('#########################\n')
                    with open(fname) as infile:
                        outfile.write(infile.read())
                    os.remove(fname)

            shutil.move(os.path.abspath(out_name), os.path.abspath(self.work_dir))

        return None

    def move_out_job_files_to_stdout(self):
        """move the error file into stdout_files directory"""

        job_files = glob.glob(self.run_file + '*.job')
        stdout_files = glob.glob(self.run_file + '*.o')
        dir_name = os.path.dirname(stdout_files[0])
        out_folder = dir_name + '/stdout_' + os.path.basename(self.run_file)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        else:
            shutil.rmtree(out_folder)
            os.mkdir(out_folder)

        for item in stdout_files:
            shutil.move(item, out_folder)
        for item in job_files:
            shutil.move(item, out_folder)

        return None

###############################################################################


class coord_rev(coordinate):
    def __init__(self, metadata, lookup_file=None):
        super().__init__(metadata, lookup_file)

    def open(self):
        try:
            self.earth_radius = float(self.src_metadata['EARTH_RADIUS'])
        except:
            self.earth_radius = 6371.0e3

        if 'Y_FIRST' in self.src_metadata.keys():
            self.geocoded = True
            self.lat0 = float(self.src_metadata['Y_FIRST'])
            self.lon0 = float(self.src_metadata['X_FIRST'])
            self.lat_step = float(self.src_metadata['Y_STEP'])
            self.lon_step = float(self.src_metadata['X_STEP'])
        else:
            self.geocoded = False
            if self.lookup_file:
                self.lut_metadata = read_attribute(self.lookup_file[0], metafile_ext='.xml')

    def read_lookup_table(self, print_msg=True):
        if 'Y_FIRST' in self.lut_metadata.keys():
            self.lut_y = readfile.read(self.lookup_file[0],
                                       datasetName='azimuthCoord',
                                       print_msg=print_msg)[0]
            self.lut_x = readfile.read(self.lookup_file[1],
                                       datasetName='rangeCoord',
                                       print_msg=print_msg)[0]
        else:
            self.lut_y = read_image(self.lookup_file[0])
            # readfile.read(self.lookup_file[0], datasetName='latitude', print_msg=print_msg)[0]
            self.lut_x = read_image(self.lookup_file[1])
            # readfile.read(self.lookup_file[1], datasetName='longitude', print_msg=print_msg)[0]
        return self.lut_y, self.lut_x

###############################################################################



###############################################################################


