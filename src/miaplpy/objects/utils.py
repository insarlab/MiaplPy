#! /usr/bin/env python3
###############################################################################
# Project: Utilities for MiaplPy
# Author: Sara Mirzaee
###############################################################################
import os
import shutil
import glob
from mintpy.objects.coord import coordinate
import h5py
from osgeo import gdal
import datetime
import re
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
from mintpy.utils import readfile, ptime, utils as ut
from mintpy.objects import (
    DSET_UNIT_DICT,
    geometry,
    GEOMETRY_DSET_NAMES,
    giantIfgramStack,
    giantTimeseries,
    IFGRAM_DSET_NAMES,
    ifgramStack,
    TIMESERIES_DSET_NAMES,
    timeseries,
    HDFEOS
)


datasetName2templateKey = {'slc': 'miaplpy.load.slcFile',
                               'unwrapPhase': 'miaplpy.load.unwFile',
                               'coherence': 'miaplpy.load.corFile',
                               'connectComponent': 'miaplpy.load.connCompFile',
                               'wrapPhase': 'miaplpy.load.intFile',
                               'iono': 'miaplpy.load.ionoFile',
                               'height': 'miaplpy.load.demFile',
                               'latitude': 'miaplpy.load.lookupYFile',
                               'longitude': 'miaplpy.load.lookupXFile',
                               'azimuthCoord': 'miaplpy.load.lookupYFile',
                               'rangeCoord': 'miaplpy.load.lookupXFile',
                               'incidenceAngle': 'miaplpy.load.incAngleFile',
                               'azimuthAngle': 'miaplpy.load.azAngleFile',
                               'shadowMask': 'miaplpy.load.shadowMaskFile',
                               'waterMask': 'miaplpy.load.waterMaskFile',
                               'bperp': 'miaplpy.load.bperpFile'
                               }



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
        error_files.sort() # = natsorted(error_files)
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

        files.sort() # = natsorted(files)
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
            print('Loading  .... ', self.lookup_file[0])
            self.lut_y = read_image(self.lookup_file[0])
            # readfile.read(self.lookup_file[0], datasetName='latitude', print_msg=print_msg)[0]
            print('Loading .... ', self.lookup_file[1])
            self.lut_x = read_image(self.lookup_file[1])
            # readfile.read(self.lookup_file[1], datasetName='longitude', print_msg=print_msg)[0]
        return self.lut_y, self.lut_x

###############################################################################


def file_len(fname):
    """Calculate the number of lines in a file."""
    with open(fname, 'r') as file:
        return len(file.readlines())

###############################################################################


def read_attribute(fname, datasetName=None, standardize=True, metafile_ext=None):
    """Read attributes of input file into a dictionary
        Parameters: fname : str, path/name of data file
                    datasetName : str, name of dataset of interest, for file with multiple datasets
                        e.g. slc         in slcStack.h5
                             date        in slcStack.h5
                             height      in geometryRadar.h5
                             latitude    in geometryRadar.h5
                             ...
                    standardize : bool, grab standardized metadata key name
        Returns:    atr : dict, attributes dictionary
        """

    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()
    if metafile_ext is None:
        test_file = fname
    else:
        test_file = fname + metafile_ext
    if not os.path.isfile(test_file):
        msg = 'input file not existed: {}\n'.format(fname)
        msg += 'current directory: '+os.getcwd()
        raise Exception(msg)

    # HDF5 files
    if fext in ['.h5', '.he5']:
        f = h5py.File(fname, 'r')
        g1_list = [i for i in f.keys() if isinstance(f[i], h5py.Group)]
        d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset) and f[i].ndim >= 2]

        # FILE_TYPE - k
        py2_mintpy_stack_files = ['interferograms', 'coherence', 'wrapped']  # obsolete mintpy format
        if any(i in d1_list for i in ['unwrapPhase', 'azimuthOffset']):
            k = 'ifgramStack'
        elif any(i in d1_list for i in ['height', 'latitude', 'azimuthCoord']):
            k = 'geometry'
        elif any(i in g1_list + d1_list for i in ['timeseries', 'displacement']):
            k = 'timeseries'
        elif any(i in g1_list + d1_list for i in ['slc']):
            k = 'slc'
        elif any(i in d1_list for i in ['velocity']):
            k = 'velocity'
        elif 'HDFEOS' in g1_list:
            k = 'HDFEOS'
        elif 'recons' in d1_list:
            k = 'giantTimeseries'
        elif any(i in d1_list for i in ['igram', 'figram']):
            k = 'giantIfgramStack'
        elif any(i in g1_list for i in py2_mintpy_stack_files):
            k = list(set(g1_list) & set(py2_mintpy_stack_files))[0]
        elif len(d1_list) > 0:
            k = d1_list[0]
        elif len(g1_list) > 0:
            k = g1_list[0]
        else:
            raise ValueError('unrecognized file type: ' + fname)

        # metadata dict
        if k == 'giantTimeseries':
            atr = giantTimeseries(fname).get_metadata()
        elif k == 'giantIfgramStack':
            atr = giantIfgramStack(fname).get_metadata()
        else:
            if len(f.attrs) > 0 and 'WIDTH' in f.attrs.keys():
                atr = dict(f.attrs)
            else:
                # grab the list of attrs in HDF5 file
                global atr_list

                def get_hdf5_attrs(name, obj):
                    global atr_list
                    if len(obj.attrs) > 0 and 'WIDTH' in obj.attrs.keys():
                        atr_list.append(dict(obj.attrs))

                atr_list = []
                f.visititems(get_hdf5_attrs)
                # use the attrs with most items
                if atr_list:
                    num_list = [len(i) for i in atr_list]
                    atr = atr_list[np.argmax(num_list)]
                else:
                    raise ValueError('No attribute WIDTH found in file:', fname)

        # decode string format
        for key, value in atr.items():
            try:
                atr[key] = value.decode('utf8')
            except:
                atr[key] = value

        # attribute identified by MintPy
        # 1. FILE_TYPE
        atr['FILE_TYPE'] = str(k)

        # 2. DATA_TYPE
        ds = None
        if datasetName and datasetName in f.keys():
            ds = f[datasetName]
        else:
            # get the 1st dataset
            global ds_list

            def get_hdf5_dataset(name, obj):
                global ds_list
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    ds_list.append(obj)

            ds_list = []
            f.visititems(get_hdf5_dataset)
            if ds_list:
                ds = ds_list[0]
        if ds is not None:
            atr['DATA_TYPE'] = str(ds.dtype)
        f.close()

        # 3. PROCESSOR
        if 'INSAR_PROCESSOR' in atr.keys():
            atr['PROCESSOR'] = atr['INSAR_PROCESSOR']
        if 'PROCESSOR' not in atr.keys():
            atr['PROCESSOR'] = 'mintpy'

    else:
        # grab all existed potential metadata file given the data file in prefered order/priority
        # .aux.xml file does not have geo-coordinates info
        # .vrt file (e.g. incLocal.rdr.vrt from isce) does not have band interleavee info
        metafiles = [
            fname + '.rsc',
            fname + '.xml',
            fname + '.par',
            os.path.splitext(fname)[0] + '.hdr',
            fname + '.vrt',
            fname + '.aux.xml',
        ]
        metafiles = [i for i in metafiles if os.path.isfile(i)]
        if len(metafiles) == 0:
            raise FileNotFoundError('No metadata file found for data file: {}'.format(fname))

        atr = {}
        # PROCESSOR
        if fname.endswith('.img') and any(i.endswith('.hdr') for i in metafiles):
            atr['PROCESSOR'] = 'snap'

        elif any(i.endswith(('.xml', '.hdr', '.vrt')) for i in metafiles):
            atr['PROCESSOR'] = 'isce'
            xml_files = [i for i in metafiles if i.endswith('.xml')]
            if len(xml_files) > 0:
                atr.update(readfile.read_isce_xml(xml_files[0]))

        elif any(i.endswith('.par') for i in metafiles):
            atr['PROCESSOR'] = 'gamma'

        elif any(i.endswith('.rsc') for i in metafiles):
            if 'PROCESSOR' not in atr.keys():
                atr['PROCESSOR'] = 'roipac'

        if 'PROCESSOR' not in atr.keys():
            atr['PROCESSOR'] = 'mintpy'

        # Read metadata file and FILE_TYPE
        metafile = metafiles[0]
        while fext in ['.geo', '.rdr', '.full']:
            fbase, fext = os.path.splitext(fbase)
        if not fext:
            fext = fbase

        if metafile.endswith('.rsc'):
            atr.update(readfile.read_roipac_rsc(metafile))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext

        elif metafile.endswith('.xml'):
            atr.update(readfile.read_isce_xml(metafile))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext

        elif metafile.endswith('.par'):
            atr.update(readfile.read_gamma_par(metafile))
            atr['FILE_TYPE'] = fext

        elif metafile.endswith('.hdr'):
            atr.update(readfile.read_envi_hdr(metafile))

            # both snap and isce produce .hdr file
            # grab file type based on their different naming conventions
            if atr['PROCESSOR'] == 'snap':
                fbase = os.path.basename(fname).lower()
                if fbase.startswith('unw'):
                    atr['FILE_TYPE'] = '.unw'
                elif fbase.startswith(('coh', 'cor')):
                    atr['FILE_TYPE'] = '.cor'
                elif fbase.startswith('phase_ifg'):
                    atr['FILE_TYPE'] = '.int'
                elif 'dem' in fbase:
                    atr['FILE_TYPE'] = 'dem'
                else:
                    atr['FILE_TYPE'] = atr['file type']
            else:
                atr['FILE_TYPE'] = fext

        elif metafile.endswith('.vrt'):
            atr.update(readfile.read_gdal_vrt(metafile))
            atr['FILE_TYPE'] = fext

        # DATA_TYPE for ISCE products
        dataTypeDict = {
            'byte': 'int8',
            'float': 'float32',
            'double': 'float64',
            'cfloat': 'complex64',
        }
        data_type = atr.get('DATA_TYPE', 'none').lower()
        if data_type != 'none' and data_type in dataTypeDict.keys():
            atr['DATA_TYPE'] = dataTypeDict[data_type]

    # UNIT
    k = atr['FILE_TYPE'].replace('.', '')
    if k == 'ifgramStack':
        if datasetName and datasetName in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[datasetName]
        else:
            atr['UNIT'] = 'radian'

    elif datasetName and datasetName in DSET_UNIT_DICT.keys():
        atr['UNIT'] = DSET_UNIT_DICT[datasetName]

    elif 'UNIT' not in atr.keys():
        if k in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[k]
        else:
            atr['UNIT'] = '1'

    # UNIT
    k = atr['FILE_TYPE'].replace('.', '')
    if k == 'slc':
        atr['UNIT'] = 'i'
    elif 'UNIT' not in atr.keys():
        if datasetName and datasetName in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[datasetName]
        elif k in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[k]
        else:
            atr['UNIT'] = '1'

    # FILE_PATH
    atr['FILE_PATH'] = os.path.abspath(fname)

    if standardize:
        atr = readfile.standardize_metadata(atr)

    return atr


def check_template_auto_value(templateDict, mintpyTemplateDict=None, auto_file='../defaults/miaplpyApp_auto.cfg',
                              templateFile=None):
    """Replace auto value based on the input auto config file."""
    # Read default template value and turn yes/no to True/False
    templateAutoFile = os.path.join(os.path.dirname(__file__), auto_file)
    templateAutoDict = readfile.read_template(templateAutoFile)

    if not mintpyTemplateDict is None:
        for key, value in mintpyTemplateDict.items():
            templateDict[key] = value

    # Update auto value of input template dict
    for key, value in templateDict.items():
        if value == 'auto' and key in templateAutoDict.keys():
            templateDict[key] = templateAutoDict[key]

    common_keys = ['load.autoPath', 'load.compression']

    if not mintpyTemplateDict is None:
        status = 'skip'

        if templateDict['miaplpy.subset.lalo'] == 'no' and templateDict['miaplpy.subset.yx'] == 'no':
            if not mintpyTemplateDict['mintpy.subset.lalo'] == 'no':
                templateDict['miaplpy.subset.lalo'] = mintpyTemplateDict['mintpy.subset.lalo']
                status = 'run'
            if not mintpyTemplateDict['mintpy.subset.yx'] == 'no':
                templateDict['miaplpy.subset.yx'] = mintpyTemplateDict['mintpy.subset.yx']
                status = 'run'
        for key in common_keys:
            if templateDict['miaplpy.' + key] == 'no':
                if not mintpyTemplateDict['mintpy.' + key] in ['no', False]:
                    templateDict['miaplpy.' + key] = mintpyTemplateDict['mintpy.' + key]
                    status = 'run'

        if not templateDict['miaplpy.load.processor'] == mintpyTemplateDict['mintpy.load.processor']:
            if templateDict['miaplpy.load.processor'] == 'isce':
                templateDict['miaplpy.load.processor'] = mintpyTemplateDict['mintpy.load.processor']
            status = 'run'

        common_keys = ['miaplpy.' + key for key in common_keys] + ['miaplpy.subset.lalo', 'miaplpy.subset.yx']
        if status == 'run':
            tmp_file = templateFile + '.tmp'
            f_tmp = open(tmp_file, 'w')
            for line in open(templateFile, 'r'):
                c = [i.strip() for i in line.strip().split('=', 1)]
                if not line.startswith(('%', '#')) and len(c) > 1:
                    key = c[0]
                    if key in common_keys and templateDict[key]:
                        s_value = templateDict[key]
                        if s_value == True:
                            s_value = 'yes'
                        elif s_value == False:
                            s_value = 'no'
                        new_value_str = '= ' + s_value
                        value = str.replace(c[1], '\n', '').split("#")[0].strip()
                        value = value.replace('*', '\*')  # interpret * as character
                        old_value_str = re.findall('=' + '[\s]*' + value, line)[0]
                        line = line.replace(old_value_str, new_value_str, 1)
                        print('    {}: {} --> {}'.format(key, value, templateDict[key]))
                f_tmp.write(line)
            f_tmp.close()

            # Overwrite exsting original template file
            shutil.move(tmp_file, templateFile)
    # Change yes --> True and no --> False
    specialValues = {'yes': True,
                     'True': True,
                     'no': False,
                     'False': False,
                     'none': None,
                     }
    for key, value in templateDict.items():
        if value in specialValues.keys():
            templateDict[key] = specialValues[value]

    return templateDict


########################################################################################
def read(fname, box=None, datasetName=None, print_msg=True):
    """Read one dataset and its attributes from input file.
    Parameters: fname : str, path of file to read
                datasetName : str or list of str, slice names
                box : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2/3-D matrix in numpy.array format, return None if failed
                atr : dictionary, attributes of data, return None if failed
    Examples:
        from mintpy.utils import readfile
        data, atr = readfile.read('velocity.h5')
        data, atr = readfile.read('timeseries.h5')
        data, atr = readfile.read('timeseries.h5', datasetName='timeseries-20161020')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase-20161020_20161026')
        data, atr = readfile.read('ifgramStack.h5', datasetName='coherence', box=(100,1100, 500, 2500))
        data, atr = readfile.read('geometryRadar.h5', datasetName='height')
        data, atr = readfile.read('geometryRadar.h5', datasetName='bperp')
        data, atr = readfile.read('100120-110214.unw', box=(100,1100, 500, 2500))
    """
    # metadata
    dsname4atr = None   #used to determine UNIT
    if isinstance(datasetName, list):
        dsname4atr = datasetName[0].split('-')[0]
    elif isinstance(datasetName, str):
        dsname4atr = datasetName.split('-')[0]

    files = [fname + i for i in ['.rsc', '.xml']]
    fext0 = ['.' + i.split('.')[-1] for i in files if os.path.exists(i)][0]

    atr = read_attribute(fname, datasetName=dsname4atr, metafile_ext=fext0)

    # box
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    if not box:
        box = (0, 0, width, length)

    # Read Data
    fext = os.path.splitext(os.path.basename(fname))[1].lower()
    if fext in ['.h5', '.he5']:
        data = read_hdf5_file(fname, datasetName=datasetName, box=box)
    else:
        data, atr = read_binary_file(fname, datasetName=datasetName, box=box)
        #data, atr = readfile.read_binary_file(fname, datasetName=datasetName, box=box, xstep=1, ystep=1)
    return data, atr


#########################################################################
def read_hdf5_file(fname, datasetName=None, box=None):
    """
    Parameters: fname : str, name of HDF5 file to read
                datasetName : str or list of str, dataset name in root level with/without date info
                    'timeseries'
                    'timeseries-20150215'
                    'unwrapPhase'
                    'unwrapPhase-20150215_20150227'
                    'HDFEOS/GRIDS/timeseries/observation/displacement'
                    'recons'
                    'recons-20150215'
                    ['recons-20150215', 'recons-20150227', ...]
                    '20150215'
                    'cmask'
                    'igram-20150215_20150227'
                    ...
                box : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2D/3D array
                atr : dict, metadata
    """
    # File Info: list of slice / dataset / dataset2d / dataset3d
    slice_list = get_slice_list(fname)
    ds_list = []
    for i in [i.split('-')[0] for i in slice_list]:
        if i not in ds_list:
            ds_list.append(i)
    ds_2d_list = [i for i in slice_list if '-' not in i]
    ds_3d_list = [i for i in ds_list if i not in ds_2d_list]

    # Input Argument: convert input datasetName into list of slice
    if not datasetName:
        datasetName = [ds_list[0]]
    elif isinstance(datasetName, str):
        datasetName = [datasetName]
    if all(i.isdigit() for i in datasetName):
        datasetName = ['{}-{}'.format(ds_3d_list[0], i) for i in datasetName]
    # Input Argument: decompose slice list into dsFamily and inputDateList
    dsFamily = datasetName[0].split('-')[0]
    inputDateList = [i.replace(dsFamily,'').replace('-','') for i in datasetName]

    # read hdf5
    with h5py.File(fname, 'r') as f:
        # get dataset object
        dsNames = [i for i in [datasetName[0], dsFamily] if i in f.keys()]
        dsNamesOld = [i for i in slice_list if '/{}'.format(datasetName[0]) in i] # support for old mintpy files
        if len(dsNames) > 0:
            ds = f[dsNames[0]]
        elif len(dsNamesOld) > 0:
            ds = f[dsNamesOld[0]]
        else:
            raise ValueError('input dataset {} not found in file {}'.format(datasetName, fname))

        # 2D dataset
        if ds.ndim == 2:
            data = ds[box[1]:box[3], box[0]:box[2]]

        # 3D dataset
        elif ds.ndim == 3:
            # define flag matrix for index in time domain
            slice_flag = np.zeros((ds.shape[0]), dtype=np.bool_)
            if not inputDateList or inputDateList == ['']:
                slice_flag[:] = True
            else:
                date_list = [i.split('-')[1] for i in
                             [j for j in slice_list if j.startswith(dsFamily)]]
                for d in inputDateList:
                    slice_flag[date_list.index(d)] = True

            # read data
            data = ds[slice_flag, box[1]:box[3], box[0]:box[2]]
            data = np.squeeze(data)
    return data


#########################################################################
def read_binary_file(fname, datasetName=None, box=None, attributes_only=False):

    """Read data from binary file, such as .unw, .cor, etc.
    Parameters: fname : str, path/name of binary file
                datasetName : str, dataset name for file with multiple bands of data
                    e.g.: incidenceAngle, azimuthAngle, rangeCoord, azimuthCoord, ...
                box  : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
    Returns:    data : 2D array in size of (length, width) in BYTE / int16 / float32 / complex64 / float64 etc.
                atr  : dict, metadata of binary file
    """
    # Basic Info
    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()

    # metadata
    atr = read_attribute(fname, metafile_ext='.xml')
    processor = atr['PROCESSOR']
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])
    if not box:
        box = (0, 0, width, length)

    # default data structure
    data_type = atr.get('DATA_TYPE', 'float32').lower()
    byte_order = atr.get('BYTE_ORDER', 'little-endian').lower()
    num_band = int(atr.get('number_bands', '1'))
    band_interleave = atr.get('scheme', 'BIL').upper()

    # default data to read
    band = 1
    cpx_band = 'phase'

    # ISCE
    if processor in ['isce']:
        # convert default short name for data type from ISCE
        dataTypeDict = {
            'byte': 'int8',
            'float': 'float32',
            'double': 'float64',
            'cfloat': 'complex64',
        }
        if data_type in dataTypeDict.keys():
            data_type = dataTypeDict[data_type]

        k = atr['FILE_TYPE'].lower().replace('.', '')
        if k in ['unw', 'cor']:
            band = min(2, num_band)
            if datasetName and datasetName in ['band1', 'intensity', 'magnitude']:
                band = 1

        elif k in ['slc']:
            cpx_band = 'magnitude'

        elif k in ['los'] and datasetName and datasetName.startswith(('band2', 'az', 'head')):
            band = min(2, num_band)

        elif k in ['incLocal']:
            band = min(2, num_band)
            if datasetName and 'local' not in datasetName.lower():
                band = 1

        elif datasetName:
            if datasetName.lower() == 'band2':
                band = 2
            elif datasetName.lower() == 'band3':
                band = 3
            elif datasetName.startswith(('mag', 'amp')):
                cpx_band = 'magnitude'
            elif datasetName in ['phase', 'angle']:
                cpx_band = 'phase'
            elif datasetName.lower() == 'real':
                cpx_band = 'real'
            elif datasetName.lower().startswith('imag'):
                cpx_band = 'imag'
            elif datasetName.startswith(('cpx', 'complex')):
                cpx_band = 'complex'

        band = min(band, num_band)

    # ROI_PAC
    elif processor in ['roipac']:
        # data structure - auto
        band_interleave = 'BIL'
        byte_order = 'little-endian'

        # data structure - file specific based on file extension
        data_type = 'float32'
        num_band = 1

        if fext in ['.unw', '.cor', '.hgt', '.msk']:
            num_band = 2
            band = 2

        elif fext in ['.int']:
            data_type = 'complex64'

        elif fext in ['.amp']:
            data_type = 'complex64'
            cpx_band = 'magnitude'

        elif fext in ['.dem', '.wgs84']:
            data_type = 'int16'

        elif fext in ['.flg', '.byt']:
            data_type = 'bool_'

        elif fext in ['.trans']:
            num_band = 2
            if datasetName and datasetName.startswith(('az', 'azimuth')):
                band = 2

    # Gamma
    elif processor == 'gamma':
        # data structure - auto
        band_interleave = 'BIL'
        byte_order = atr.get('BYTE_ORDER', 'big-endian')

        data_type = 'float32'
        if fext in ['.unw', '.cor', '.hgt_sim', '.dem', '.amp', '.ramp']:
            pass

        elif fext in ['.int']:
            data_type = 'complex64'

        elif fext in ['.utm_to_rdc']:
            data_type = 'complex64'
            if datasetName and datasetName.startswith(('az', 'azimuth')):
                cpx_band = 'imag'
            else:
                cpx_band = 'real'

        elif fext == '.slc':
            data_type = 'complex32'
            cpx_band = 'magnitude'

        elif fext in ['.mli']:
            byte_order = 'little-endian'

    # SNAP
    # BEAM-DIMAP data format
    # https://www.brockmann-consult.de/beam/doc/help/general/BeamDimapFormat.html
    elif processor == 'snap':
        # data structure - auto
        band_interleave = atr.get('scheme', 'BSQ').upper()

        # byte order
        byte_order = atr.get('BYTE_ORDER', 'big-endian')
        if 'byte order' in atr.keys() and atr['byte order'] == '0':
            byte_order = 'little-endian'

    else:
        print('Unknown InSAR processor.')
    
    if 'DATA_TYPE' not in atr:
        atr['DATA_TYPE'] = data_type
    
    if attributes_only:
        return atr
    else:
        # reading
        data = read_image(fname, box=box, band=band)
        return data, atr




def get_slice_list(fname):
    """Get list of 2D slice existed in file (for display)"""
    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()
    atr = read_attribute(fname)
    k = atr['FILE_TYPE']

    global slice_list
    # HDF5 Files
    if fext in ['.h5', '.he5']:
        with h5py.File(fname, 'r') as f:
            d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset)]
        if k == 'timeseries' and k in d1_list:
            obj = timeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k == 'slc':
            obj = slcStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['geometry'] and k not in d1_list:
            obj = geometry(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['ifgramStack']:
            obj = ifgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['HDFEOS']:
            obj = HDFEOS(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantTimeseries']:
            obj = giantTimeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif k in ['giantIfgramStack']:
            obj = giantIfgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        else:
            ## Find slice by walking through the file structure
            length, width = int(atr['LENGTH']), int(atr['WIDTH'])

            def get_hdf5_2d_dataset(name, obj):
                global slice_list
                if isinstance(obj, h5py.Dataset) and obj.shape[-2:] == (length, width):
                    if obj.ndim == 2:
                        slice_list.append(name)
                    else:
                        warnings.warn('file has un-defined {}D dataset: {}'.format(obj.ndim, name))

            slice_list = []
            with h5py.File(fname, 'r') as f:
                f.visititems(get_hdf5_2d_dataset)

    # Binary Files
    else:
        if fext.lower() in ['.trans', '.utm_to_rdc']:
            slice_list = ['rangeCoord', 'azimuthCoord']
        elif fbase.startswith('los'):
            slice_list = ['incidenceAngle', 'azimuthAngle']
        elif atr.get('number_bands', '1') == '2' and 'unw' not in k:
            slice_list = ['band1', 'band2']
        else:
            slice_list = ['']
    return slice_list

#############################################################################


def most_common(L, k=1):
    """Return the k most common item in the list L.
    Examples:
        5, 8 = most_common([4,5,5,5,5,8,8,8,9], k=2)
        'duck' = most_common(['goose','duck','duck','dog'])
        'goose' = most_common(['goose','duck','duck','goose'])
    """
    from collections import Counter
    cnt = Counter(L)
    item_mm = [i[0] for i in cnt.most_common(k)]
    if k == 1:
        item_mm = item_mm[0]
    return item_mm


def print_write_setting(iDict):
    updateMode = iDict['updateMode']
    comp = iDict['compression']
    print('-'*50)
    print('updateMode : {}'.format(updateMode))
    print('compression: {}'.format(comp))

    # box
    box = iDict['box']
    # box for geometry file in geo-coordinates
    if not iDict.get('geocoded', False):
        boxGeo = iDict['box4geo_lut']
    else:
        boxGeo = box

    # step
    xyStep = (iDict['xstep'], iDict['ystep'])
    if not iDict.get('geocoded', False):
        xyStepGeo = (1, 1)
    else:
        xyStepGeo = xyStep
    print('x/ystep: {}/{}'.format(xyStep[0], xyStep[1]))

    return updateMode, comp, box, boxGeo, xyStep, xyStepGeo

############################################################


def log_message(logdir, msg):
    f = open(os.path.join(logdir, 'log'), 'a+')
    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')
    string = dateStr + " * " + msg
    print(string)
    f.write(string + "\n")
    f.close()
    return

############################################################


def read_image(image_file, box=None, band=1):
    """ Reads images from isce. """

    ds = gdal.Open(image_file + '.vrt', gdal.GA_ReadOnly)
    if not box is None:
        imds = ds.GetRasterBand(band)
        image = imds.ReadAsArray()[box[1]:box[3], box[0]:box[2]]
    else:
        image = ds.GetRasterBand(band).ReadAsArray()

    del ds

    return image


def custom_cmap(vmin=0, vmax=1):
    """ create a custom colormap based on visible portion of electromagnetive wave."""

    from miaplpy.spectrumRGB import rgb
    rgb = rgb()
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(rgb)
    norm = mpl.colors.Normalize(vmin, vmax)

    return cmap, norm



def email_miaplpy(work_dir):
    """ email mintpy results """

    import subprocess
    import sys

    email_address = os.getenv('NOTIFICATIONEMAIL')

    textStr = 'email mintpy results'

    cwd = os.getcwd()

    pic_dir = os.path.join(work_dir, 'pic')
    flist = ['avgPhaseVelocity.png', 'avgSpatialCoh.png', 'geo_maskTempCoh.png', 'geo_temporalCoherence.png',
             'geo_velocity.png', 'maskConnComp.png', 'Network.pdf', 'BperpHistory.pdf', 'CoherenceMatrix.pdf',
             'rms_timeseriesResidual_ramp.pdf', 'geo_velocity.kmz']

    file_list = [os.path.join(pic_dir, i) for i in flist]
    print(file_list)

    attachmentStr = ''
    i = 0
    for fileList in file_list:
        i = i + 1
        attachmentStr = attachmentStr + ' -a ' + fileList

    mailCmd = 'echo \"' + textStr + '\" | mail -s ' + cwd + ' ' + attachmentStr + ' ' + email_address
    command = 'ssh pegasus.ccs.miami.edu \"cd ' + cwd + '; ' + mailCmd + '\"'
    print(command)
    status = subprocess.Popen(command, shell=True).wait()
    if status is not 0:
        sys.exit('Error in email_miaplpy')

    return



def get_latest_template_miaplpy(work_dir):
    from miaplpy.objects.read_template import Template

    """Get the latest version of default template file.
    If an obsolete file exists in the working directory, the existing option values are kept.
    """
    lfile = os.path.join(os.path.dirname(__file__), '../defaults/miaplpyApp.cfg')  # latest version
    cfile = os.path.join(work_dir, 'miaplpyApp.cfg')  # current version

    if not os.path.isfile(cfile):
        print('copy default template file {} to work directory'.format(lfile))
        shutil.copy2(lfile, work_dir)
    else:
        # read custom template from file
        cdict = Template(cfile).options
        ldict = Template(lfile).options

        if any([key not in cdict.keys() for key in ldict.keys()]):
            print('obsolete default template detected, update to the latest version.')
            shutil.copy2(lfile, work_dir)
            orig_dict = Template(cfile).options
            for key, value in orig_dict.items():
                if key in cdict.keys() and cdict[key] != value:
                    update = True
                else:
                    update = False
            if not update:
                print('No new option value found, skip updating ' + cfile)
                return cfile

            # Update template_file with new value from extra_dict
            tmp_file = cfile + '.tmp'
            f_tmp = open(tmp_file, 'w')
            for line in open(cfile, 'r'):
                c = [i.strip() for i in line.strip().split('=', 1)]
                if not line.startswith(('%', '#')) and len(c) > 1:
                    key = c[0]
                    value = str.replace(c[1], '\n', '').split("#")[0].strip()
                    if key in cdict.keys() and cdict[key] != value:
                        line = line.replace(value, cdict[key], 1)
                        print('    {}: {} --> {}'.format(key, value, cdict[key]))
                f_tmp.write(line)
            f_tmp.close()

            # Overwrite exsting original template file
            shutil.move(tmp_file, cfile)
            #mvCmd = 'mv {} {}'.format(tmp_file, cfile)
            #os.system(mvCmd)
    return cfile

def read_subset_template2box(template_file):
    """Read miaplpy.subset.lalo/yx option from template file into box type
    Return None if not specified.
    (Modified from mintpy.subsets)
    """
    tmpl = readfile.read_template(template_file)
    keys = ['miaplpy.subset.lalo', 'mintpy.subset.lalo']
    key_lalo = [key for key in keys if key in tmpl][0]
    keys = ['miaplpy.subset.yx', 'mintpy.subset.yx']
    key_yx = [key for key in keys if key in tmpl][0]
    # subset.lalo -> geo_box
    try:
        opts = [i.strip().replace('[','').replace(']','') for i in tmpl[key_lalo].split(',')]
        lat0, lat1 = sorted([float(i.strip()) for i in opts[0].split(':')])
        lon0, lon1 = sorted([float(i.strip()) for i in opts[1].split(':')])
        geo_box = (lon0, lat1, lon1, lat0)
    except:
        geo_box = None

    # subset.yx -> pix_box
    try:
        opts = [i.strip().replace('[','').replace(']','') for i in tmpl[key_yx].split(',')]
        y0, y1 = sorted([int(i.strip()) for i in opts[0].split(':')])
        x0, x1 = sorted([int(i.strip()) for i in opts[1].split(':')])
        pix_box = (x0, y0, x1, y1)
    except:
        pix_box = None

    return pix_box, geo_box


def read_subset_box(inpsDict):
    import mintpy.load_data as mld
    from mintpy import subset

    # Read subset info from template
    inpsDict['box'] = None
    inpsDict['box4geo_lut'] = None

    pix_box, geo_box = read_subset_template2box(inpsDict['template_file'][0])
    #pix_box, geo_box = subset.read_subset_template2box(inpsDict['template_file'][0])

    # Grab required info to read input geo_box into pix_box

    try:
        lookupFile = [glob.glob(str(inpsDict['miaplpy.load.lookupYFile'] + '.xml'))[0],
                      glob.glob(str(inpsDict['miaplpy.load.lookupXFile'] + '.xml'))[0]]
        lookupFile = [x.split('.xml')[0] for x in lookupFile]
    except:
        lookupFile = None

    try:

        pathKey = [i for i in datasetName2templateKey.values()
                   if i in inpsDict.keys()][0]

        file = glob.glob(str(inpsDict[pathKey] + '.xml'))[0]
        atr = read_attribute(file.split('.xml')[0], metafile_ext='.rsc')
    except:
        atr = dict()
    geocoded = None
    if 'Y_FIRST' in atr.keys():
        geocoded = True
    else:
        geocoded = False

    # Check conflict
    if geo_box and not geocoded and lookupFile is None:
        geo_box = None
        print(('WARNING: mintpy.subset.lalo is not supported'
               ' if 1) no lookup file AND'
               '    2) radar/unkonwn coded dataset'))
        print('\tignore it and continue.')

    if not geo_box and not pix_box:
        # adjust for the size inconsistency problem in SNAP geocoded products
        # ONLY IF there is no input subset
        # Use the min bbox if files size are different
        if inpsDict['processor'] == 'snap':
            fnames = ut.get_file_list(inpsDict['miaplpy.load.slcFile'])
            pix_box = mld.update_box4files_with_inconsistent_size(fnames)

        #if not pix_box:
        #    return inpsDict

    # geo_box --> pix_box
    coord = coord_rev(atr, lookup_file=lookupFile)
    if geo_box is not None:
        pix_box = coord.bbox_geo2radar(geo_box)
        pix_box = coord.check_box_within_data_coverage(pix_box)
        print('input bounding box of interest in lalo: {}'.format(geo_box))
    print('box to read for datasets in y/x: {}'.format(pix_box))

    # Get box for geocoded lookup table (for gamma/roipac)
    box4geo_lut = None
    if lookupFile is not None:
        atrLut = read_attribute(lookupFile[0], metafile_ext='.xml')
        if not geocoded and 'Y_FIRST' in atrLut.keys():
            geo_box = coord.bbox_radar2geo(pix_box)
            box4geo_lut = ut.coordinate(atrLut).bbox_geo2radar(geo_box)
            print('box to read for geocoded lookup file in y/x: {}'.format(box4geo_lut))

    if pix_box in [None, 'None'] and 'WIDTH' in atr:
        pix_box = (0, 0, int(atr['WIDTH']), int(atr['LENGTH']))

    for key in atr:
        if not key in inpsDict or inpsDict[key] in [None, 'NONE']:
            inpsDict[key] = atr[key]

    inpsDict['box'] = pix_box
    inpsDict['box4geo_lut'] = box4geo_lut
    return inpsDict


def update_or_skip_inversion(inverted_date_list, slc_dates):

    with open(inverted_date_list, 'r') as f:
        inverted_dates = f.readlines()

    inverted_dates = [date.split('\n')[0] for date in inverted_dates]
    new_slc_dates = list(set(slc_dates) - set(inverted_dates))
    all_date_list = new_slc_dates + inverted_dates

    updated_index = None
    if inverted_dates == slc_dates:
        print(('All date exists in file {} with same size as required,'
               ' no need to update inversion.'.format(os.path.basename(inverted_date_list))))
    elif len(slc_dates) < 10 + len(inverted_dates):
        print('Number of new images is less than 10 --> wait until at least 10 images are acquired')

    else:
        updated_index = len(inverted_dates)

    return updated_index, all_date_list


def read_initial_info(work_dir, templateFile):
    from miaplpy.objects.slcStack import slcStack
    #import miaplpy.workflow

    slc_file = os.path.join(work_dir, 'inputs/slcStack.h5')

    if os.path.exists(slc_file):
        slcObj = slcStack(slc_file)
        slcObj.open(print_msg=False)
        date_list = slcObj.get_date_list()
        metadata = slcObj.get_metadata()
        num_pixels = int(metadata['LENGTH']) * int(metadata['WIDTH'])
    else:
        scp_args = '--template {}'.format(templateFile)
        scp_args += ' --project_dir {} --work_dir {}'.format(os.path.dirname(work_dir), work_dir)

        Parser_LoadSlc = MiaplPyParser(scp_args.split(), script='load_slc')
        inps_loadSlc = Parser_LoadSlc.parse()
        iDict = read_inps2dict(inps_loadSlc)
        prepare_metadata(iDict)
        metadata = read_subset_box(iDict)
        box = metadata['box']
        num_pixels = (box[2] - box[0]) * (box[3] - box[1])
        stackObj = read_inps_dict2slc_stack_dict_object(iDict)
        date_list = stackObj.get_date_list()

    return date_list, num_pixels, metadata


def read_inps2dict(inps):
    """Read input Namespace object info into inpsDict"""

    from miaplpy.defaults import auto_path
    from mintpy.objects import sensor

    # Read input info into inpsDict
    inpsDict = vars(inps)
    inpsDict['PLATFORM'] = None
    auto_template = os.path.join(os.path.dirname(__file__), '../defaults/miaplpyApp_auto.cfg')

    # Read template file
    template = {}
    for fname in inps.template_file:
        temp = readfile.read_template(fname)
        temp = check_template_auto_value(temp, auto_file=auto_template)
        template.update(temp)
    for key, value in template.items():
        inpsDict[key] = value
    if 'processor' in template.keys():
        template['miaplpy.load.processor'] = template['processor']

    prefix = 'miaplpy.load.'
    key_list = [i.split(prefix)[1] for i in template.keys() if i.startswith(prefix)]
    for key in key_list:
        value = template[prefix + key]
        if key in ['processor', 'updateMode', 'compression', 'autoPath']:
            inpsDict[key] = template[prefix + key]
        elif key in ['xstep', 'ystep']:
            inpsDict[key] = int(template[prefix + key])
        elif value:
            inpsDict[prefix + key] = template[prefix + key]

    if not 'compression' in inpsDict or inpsDict['compression'] == False:
        inpsDict['compression'] = None

    inpsDict['xstep'] = inpsDict.get('xstep', 1)
    inpsDict['ystep'] = inpsDict.get('ystep', 1)

    # PROJECT_NAME --> PLATFORM
    if not 'PROJECT_NAME' in inpsDict:
        cfile = [i for i in list(inps.template_file) if os.path.basename(i) != 'miaplpyApp.cfg']
        inpsDict['PROJECT_NAME'] = sensor.project_name2sensor_name(cfile)[1]

    inpsDict['PLATFORM'] = str(sensor.project_name2sensor_name(str(inpsDict['PROJECT_NAME']))[0])
    if inpsDict['PLATFORM']:
        print('SAR platform/sensor : {}'.format(inpsDict['PLATFORM']))
    print('processor: {}'.format(inpsDict['processor']))

    # Here to insert code to check default file path for miami user
    #work_dir = os.path.dirname(os.path.dirname(inpsDict['outfile']))
    if inpsDict.get('autoPath', False):
        print(('check auto path setting for Univ of Miami users'
               ' for processor: {}'.format(inpsDict['processor'])))
        inpsDict = auto_path.get_auto_path(processor=inpsDict['processor'],
                                           work_dir=inps.work_dir,
                                           template=inpsDict)

    reference_dir = os.path.dirname(inpsDict['miaplpy.load.metaFile'])
    out_reference = inps.work_dir + '/inputs/reference'
    if not os.path.exists(out_reference):
        shutil.copytree(reference_dir, out_reference)

    baseline_dir = os.path.abspath(inpsDict['miaplpy.load.baselineDir'])
    out_baseline = inps.work_dir + '/inputs/baselines'
    if not os.path.exists(out_baseline):
        shutil.copytree(baseline_dir, out_baseline)

    return inpsDict


def prepare_metadata(inpsDict):
    import warnings
    processor = inpsDict['processor']
    script_name = 'prep_slc_{}.py'.format(processor)
    print('-' * 50)
    print('prepare metadata files for {} products'.format(processor))
    if processor in ['gamma', 'roipac', 'snap']:
        for key in [i for i in inpsDict.keys() if (i.startswith('miaplpy.load.') and i.endswith('File'))]:
            if len(glob.glob(str(inpsDict[key]))) > 0:
                cmd = '{} {}'.format(script_name, inpsDict[key])
                print(cmd)
                os.system(cmd)

    elif processor == 'isce':

        slc_dir = os.path.dirname(os.path.dirname(inpsDict['miaplpy.load.slcFile']))
        slc_file = os.path.basename(inpsDict['miaplpy.load.slcFile'])
        meta_files = sorted(glob.glob(inpsDict['miaplpy.load.metaFile']))
        if len(meta_files) < 1:
            warnings.warn('No input metadata file found: {}'.format(inpsDict['miaplpy.load.metaFile']))
        try:
            meta_file = meta_files[0]
            baseline_dir = inpsDict['miaplpy.load.baselineDir']
            geom_dir = os.path.dirname(inpsDict['miaplpy.load.demFile'])
            cmd = '{s} -s {i} -f {f} -m {m} -b {b} -g {g} --force'.format(s=script_name,
                                                                  i=slc_dir,
                                                                  f=slc_file,
                                                                  m=meta_file,
                                                                  b=baseline_dir,
                                                                  g=geom_dir)
            print(cmd)
            os.system(cmd)
        except:
            pass
    return


def multilook(infile, outfile, rlks, alks, multilook_tool='gdal'):
    from mroipac.looks.Looks import Looks
    import isceobj

    if multilook_tool == "gdal":

        print(infile)
        ds = gdal.Open(infile + ".vrt", gdal.GA_ReadOnly)

        xSize = ds.RasterXSize
        ySize = ds.RasterYSize

        outXSize = xSize / int(rlks)
        outYSize = ySize / int(alks)

        gdalTranslateOpts = gdal.TranslateOptions(format="ENVI", width=outXSize, height=outYSize)

        gdal.Translate(outfile, ds, options=gdalTranslateOpts)
        ds = None

        ds = gdal.Open(outfile, gdal.GA_ReadOnly)
        gdal.Translate(outfile + ".vrt", ds, options=gdal.TranslateOptions(format="VRT"))
        ds = None

    else:

        print('Multilooking {0} ...'.format(infile))

        inimg = isceobj.createImage()
        inimg.load(infile + '.xml')

        lkObj = Looks()
        lkObj.setDownLooks(alks)
        lkObj.setAcrossLooks(rlks)
        lkObj.setInputImage(inimg)
        lkObj.setOutputFilename(outfile)
        lkObj.looks()

    return outfilie

def ks_lut(N1, N2, alpha=0.05):
    N = (N1 * N2) / float(N1 + N2)
    distances = np.arange(0.01, 1, 1/1000)
    lamda = distances*(np.sqrt(N) + 0.12 + 0.11/np.sqrt(N))
    alpha_c = np.zeros([len(distances)])
    for value in lamda:
        n = np.ogrid[1:101]
        pvalue = 2*np.sum(((-1)**(n-1))*np.exp(-2*(value**2)*(n**2)))
        pvalue = np.amin(np.amax(pvalue, initial=0), initial=1)
        alpha_c[lamda == value] = pvalue
    critical_distance = distances[alpha_c <= (alpha)]
    return np.min(critical_distance)



def est_corr(CCGsam):
    """ Estimate Correlation matrix from an ensemble."""

    CCGS = np.matrix(CCGsam)

    cov_mat = np.matmul(CCGS, CCGS.getH()) / CCGS.shape[1]

    corr_matrix = cov2corr(cov_mat)

    #corr_matrix = np.multiply(cov2corr(np.abs(cov_mat)), np.exp(1j * np.angle(cov_mat)))

    return corr_matrix


def cov2corr(cov_matrix):
    """ Converts covariance matrix to correlation/coherence matrix. """

    D = LA.pinv(np.diagflat(np.sqrt(np.diag(cov_matrix))))
    y = np.matmul(D, cov_matrix)
    corr_matrix = np.matmul(y, np.transpose(D))

    return corr_matrix


def read_inps_dict2slc_stack_dict_object(inpsDict):
    """Read input arguments into dict of slcStackDict object"""
    # inpsDict --> dsPathDict
    print('-' * 50)
    print('searching slcs info')
    print('input data files:')

    from miaplpy.objects.slcStack import (slcDatasetNames,
                                         slcStackDict,
                                         slcDict)

    maxDigit = max([len(i) for i in list(datasetName2templateKey.keys())])
    dsPathDict = {}
    for dsName in [i for i in slcDatasetNames
                   if i in datasetName2templateKey.keys()]:
        key = datasetName2templateKey[dsName]
        if key in inpsDict.keys():
            files = sorted(glob.glob(str(inpsDict[key] + '.xml')))
            if len(files) > 0:
                dsPathDict[dsName] = files
                print('{:<{width}}: {path}'.format(dsName,
                                                   width=maxDigit,
                                                   path=inpsDict[key]))

    # Check 1: required dataset
    dsName0 = 'slc'
    if dsName0 not in dsPathDict.keys():
        print('WARNING: No reqired {} data files found!'.format(dsName0))
        return None

    # Check 2: data dimension for slc files
    dsPathDict = skip_files_with_inconsistent_size(dsPathDict,
                                                   pix_box=inpsDict['box'],
                                                   dsName=dsName0)

    # Check 3: number of files for all dataset types
    # dsPathDict --> dsNumDict
    dsNumDict = {}
    for key in dsPathDict.keys():
        num_file = len(dsPathDict[key])
        dsNumDict[key] = num_file
        print('number of {:<{width}}: {num}'.format(key, width=maxDigit, num=num_file))

    dsNumList = list(dsNumDict.values())
    if any(i != dsNumList[0] for i in dsNumList):
        msg = 'WARNING: NOT all types of dataset have the same number of files.'
        msg += ' -> skip interferograms with missing files and continue.'
        print(msg)
        # raise Exception(msg)

    if not inpsDict['miaplpy.load.startDate'] in [None, 'None']:
        start_date = datetime.datetime.strptime(inpsDict['miaplpy.load.startDate'], '%Y%m%d')
    else:
        start_date = None
    if not inpsDict['miaplpy.load.endDate'] in [None, 'None']:
        end_date = datetime.datetime.strptime(inpsDict['miaplpy.load.endDate'], '%Y%m%d')
    else:
        end_date = None

    # dsPathDict --> pairsDict --> stackObj
    dsNameList = list(dsPathDict.keys())
    pairsDict = {}

    for dsPath in dsPathDict[dsName0]:
        dates = ptime.yyyymmdd(read_attribute(dsPath.split('.xml')[0], metafile_ext='.rsc')['DATE'])
        date_val = datetime.datetime.strptime(dates, '%Y%m%d')
        include_date = True
        if not start_date is None and start_date > date_val:
            include_date = False
        if not end_date is None and end_date < date_val:
            include_date = False

        if include_date:
            #####################################
            # A dictionary of data files for a given pair.
            # One pair may have several types of dataset.
            # example slcPathDict = {'slc': /pathToFile/*.slc.full}
            # All path of data file must contain the reference and secondary date, either in file name or folder name.
            slcPathDict = {}
            for i in range(len(dsNameList)):
                dsName = dsNameList[i]
                dsPath1 = dsPathDict[dsName][0]
                if dates in dsPath1:
                    slcPathDict[dsName] = dsPath1
                else:
                    dsPath2 = [i for i in dsPathDict[dsName] if dates in i]
                    if len(dsPath2) > 0:
                        slcPathDict[dsName] = dsPath2[0]
                    else:
                        print('WARNING: {} file missing for image {}'.format(dsName, dates))

            slcObj = slcDict(dates=dates, datasetDict=slcPathDict)
            pairsDict[dates] = slcObj

    if len(pairsDict) > 0:
        stackObj = slcStackDict(pairsDict=pairsDict)
    else:
        stackObj = None
    return stackObj



def skip_files_with_inconsistent_size(dsPathDict, pix_box=None, dsName='slc'):
    """Skip files by removing the file path from the input dsPathDict."""
    atr_list = [read_attribute(fname.split('.xml')[0], metafile_ext='.xml') for fname in dsPathDict[dsName]]
    length_list = [int(atr['LENGTH']) for atr in atr_list]
    width_list = [int(atr['WIDTH']) for atr in atr_list]

    # Check size requirements
    drop_inconsistent_files = False
    if any(len(set(size_list)) > 1 for size_list in [length_list, width_list]):
        if pix_box is None:
            drop_inconsistent_files = True
        else:
            # if input subset is within the min file sizes: do NOT drop
            max_box_width, max_box_length = pix_box[2:4]
            if max_box_length > min(length_list) or max_box_width > min(width_list):
                drop_inconsistent_files = True

    # update dsPathDict
    if drop_inconsistent_files:
        common_length = ut.most_common(length_list)
        common_width = ut.most_common(width_list)

        # print out warning message
        msg = '\n' + '*' * 80
        msg += '\nWARNING: NOT all input unwrapped interferograms have the same row/column number!'
        msg += '\nThe most common size is: ({}, {})'.format(common_length, common_width)
        msg += '\n' + '-' * 30
        msg += '\nThe following dates have different size:'

        dsNames = list(dsPathDict.keys())
        date_list = [atr['DATE'] for atr in atr_list]
        for i in range(len(date_list)):
            if length_list[i] != common_length or width_list[i] != common_width:
                date = date_list[i]
                dates = ptime.yyyymmdd(date)
                # update file list for all datasets
                for dsName in dsNames:
                    fnames = [i for i in dsPathDict[dsName]
                              if all(d[2:8] in i for d in dates)]
                    if len(fnames) > 0:
                        dsPathDict[dsName].remove(fnames[0])
                msg += '\n\t{}\t({}, {})'.format(date, length_list[i], width_list[i])

        msg += '\n' + '-' * 30
        msg += '\nSkip loading the interferograms above.'
        msg += '\nContinue to load the rest interferograms.'
        msg += '\n' + '*' * 80 + '\n'
        print(msg)
    return dsPathDict

def write_layout_hdf5(fname, ds_name_dict=None, metadata=None, ds_unit_dict=None, ref_file=None, compression=None, print_msg=True):
    """Create HDF5 file with defined metadata and (empty) dataset structure

    Parameters: fname        - str, HDF5 file path
                ds_name_dict - dict, dataset structure definition
                               {dname : [dtype, dshape],
                                dname : [dtype, dshape, None],
                                dname : [dtype, dshape, 1/2/3/4D np.ndarray], #for aux data
                                ...
                               }
                metadata     - dict, metadata
                ds_unit_dict - dict, dataset unit definition
                               {dname : dunit,
                                dname : dunit,
                                ...
                               }
                ref_file     - str, reference file for the data structure
                compression  - str, HDF5 compression type
    Returns:    fname        - str, HDF5 file path

    Example:    layout_hdf5('timeseries_ERA5.h5', ref_file='timeseries.h5')
                layout_hdf5('timeseries_ERA5.5h', ds_name_dict, metadata)

    # structure for ifgramStack
    ds_name_dict = {
        "date"             : [np.dtype('S8'), (num_ifgram, 2)],
        "dropIfgram"       : [np.bool_,       (num_ifgram,)],
        "bperp"            : [np.float32,     (num_ifgram,)],
        "unwrapPhase"      : [np.float32,     (num_ifgram, length, width)],
        "coherence"        : [np.float32,     (num_ifgram, length, width)],
        "connectComponent" : [np.int16,       (num_ifgram, length, width)],
    }

    # structure for geometry
    ds_name_dict = {
        "height"             : [np.float32, (length, width), None],
        "incidenceAngle"     : [np.float32, (length, width), None],
        "slantRangeDistance" : [np.float32, (length, width), None],
    }

    # structure for timeseries
    dates = np.array(date_list, np.string_)
    ds_name_dict = {
        "date"       : [np.dtype("S8"), (num_date,), dates],
        "bperp"      : [np.float32,     (num_date,), pbase],
        "timeseries" : [np.float32,     (num_date, length, width)],
    }
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    vprint('-'*50)

    # get meta from metadata and ref_file
    if metadata:
        meta = {key: value for key, value in metadata.items()}
    elif ref_file:
        with h5py.File(ref_file, 'r') as fr:
            meta = {key: value for key, value in fr.attrs.items()}
        vprint('grab metadata from ref_file: {}'.format(ref_file))
    else:
        raise ValueError('No metadata or ref_file found.')

    # check ds_name_dict
    if ds_name_dict is None:
        if not ref_file or not os.path.isfile(ref_file):
            raise FileNotFoundError('No ds_name_dict or ref_file found!')
        else:
            vprint('grab dataset structure from ref_file: {}'.format(ref_file))

        ds_name_dict = {}
        fext = os.path.splitext(ref_file)[1]
        shape2d = (int(meta['LENGTH']), int(meta['WIDTH']))

        if fext in ['.h5', '.he5']:
            # copy dset structure from HDF5 file
            with h5py.File(ref_file, 'r') as fr:
                # in case output mat size is different from the input ref file mat size
                shape2d_orig = (int(fr.attrs['LENGTH']), int(fr.attrs['WIDTH']))

                for key in fr.keys():
                    ds = fr[key]
                    if isinstance(ds, h5py.Dataset):

                        # auxliary dataset
                        if ds.shape[-2:] != shape2d_orig:
                            ds_name_dict[key] = [ds.dtype, ds.shape, ds[:]]

                        # dataset
                        else:
                            ds_shape = list(ds.shape)
                            ds_shape[-2:] = shape2d
                            ds_name_dict[key] = [ds.dtype, tuple(ds_shape), None]

        else:
            # construct dset structure from binary file
            ds_names = readfile.get_slice_list(ref_file)
            ds_dtype = meta['DATA_TYPE']
            for ds_name in ds_names:
                ds_name_dict[ds_name] = [ds_dtype, tuple(shape2d), None]

    # directory
    fdir = os.path.dirname(os.path.abspath(fname))
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
        vprint('crerate directory: {}'.format(fdir))

    # create file
    with h5py.File(fname, "a") as f:
        vprint('create HDF5 file: {} with w mode'.format(fname))

        # initiate dataset
        max_digit = max([len(i) for i in ds_name_dict.keys()])
        for key in ds_name_dict.keys():
            data_type  = ds_name_dict[key][0]
            data_shape = ds_name_dict[key][1]

            # turn ON compression for conn comp
            ds_comp = compression
            if key in ['connectComponent']:
                ds_comp = 'lzf'

            # changable dataset shape
            if len(data_shape) == 3:
                max_shape = (None, data_shape[1], data_shape[2])
            else:
                max_shape = data_shape

            # create empty dataset
            vprint(("create dataset  : {d:<{w}} of {t:<25} in size of {s:<20} with "
                    "compression = {c}").format(d=key,
                                                w=max_digit,
                                                t=str(data_type),
                                                s=str(data_shape),
                                                c=ds_comp))
            if not key in f.keys():
                ds = f.create_dataset(key,
                                      shape=data_shape,
                                      maxshape=max_shape,
                                      dtype=data_type,
                                      chunks=True,
                                      compression=ds_comp)
            else:
                ds = f[key]

            # write auxliary data
            if len(ds_name_dict[key]) > 2 and ds_name_dict[key][2] is not None:
                ds[:] = np.array(ds_name_dict[key][2])

        # write attributes in root level
        for key, value in meta.items():
            f.attrs[key] = str(value)

        # write attributes in dataset level
        if ds_unit_dict is not None:
            for key, value in ds_unit_dict.items():
                if value is not None:
                    f[key].attrs['UNIT'] = value
                    vprint(f'add /{key:<{max_digit}} attribute: UNIT = {value}')

    vprint('close  HDF5 file: {}'.format(fname))

    return fname
