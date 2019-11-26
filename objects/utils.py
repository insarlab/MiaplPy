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
from minopy.minopy_utilities import read_image
import h5py
from mintpy.utils import readfile
from minopy.minopy_utilities import read_image
from mintpy.utils.readfile import standardize_metadata
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
        if any(i in d1_list for i in ['unwrapPhase']):
            k = 'ifgramStack'
        elif any(i in d1_list for i in ['height', 'latitude', 'azimuthCoord']):
            k = 'geometry'
        elif any(i in g1_list + d1_list for i in ['timeseries', 'displacement']):
            k = 'timeseries'
        elif any(i in g1_list + d1_list for i in ['slc']):
            k = 'slc'
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

        metafile_base = fname

        if fname.endswith('.img') and os.path.isfile(os.path.splitext(fname)[0]+'.hdr'):
            metafile_base = os.path.splitext(fname)[0]

        # get existing metadata file extensions
        metafile_exts = ['.rsc', '.xml', '.aux.xml', '.par', '.hdr']
        if metafile_ext:
            metafile_exts = [i for i in metafile_exts if i.endswith(metafile_ext)]
        metafile_exts = [i for i in metafile_exts if os.path.isfile(metafile_base+i)]
        if len(metafile_exts) == 0:
            raise FileNotFoundError('No metadata file found for data file: {}'.format(fname))

        atr = {}
        # PROCESSOR
        if any(i.endswith('.hdr') for i in metafile_exts) and fname.endswith('.img'):
            atr['PROCESSOR'] = 'snap'
        elif any(i.endswith('.xml') for i in metafile_exts):
            atr['PROCESSOR'] = 'isce'
            xml_exts = [i for i in metafile_exts if i.endswith('.xml')]
            if len(xml_exts) > 0:
                atr.update(readfile.read_isce_xml(metafile_base+xml_exts[0]))
        elif any(i.endswith('.par') for i in metafile_exts):
            atr['PROCESSOR'] = 'gamma'
        elif any(i.endswith('.rsc') for i in metafile_exts):
            if 'PROCESSOR' not in atr.keys():
                atr['PROCESSOR'] = 'roipac'
        if 'PROCESSOR' not in atr.keys():
            atr['PROCESSOR'] = 'mintpy'

        # Read metadata file and FILE_TYPE
        metafile0 = metafile_base + metafile_exts[0]
        while fext in ['.geo', '.rdr', '.full']:
            fbase, fext = os.path.splitext(fbase)
        if not fext:
            fext = fbase

        if metafile0.endswith('.rsc'):
            atr.update(readfile.read_roipac_rsc(metafile0))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext

        elif metafile0.endswith('.xml'):
            atr.update(readfile.read_isce_xml(metafile0))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext  #atr.get('image_type', fext)

        elif metafile0.endswith('.par'):
            atr.update(readfile.read_gamma_par(metafile0))
            atr['FILE_TYPE'] = fext

        elif metafile0.endswith('.hdr'):
            atr.update(readfile.read_envi_hdr(metafile0))
            fbase = os.path.basename(fname).lower()
            if fbase.startswith('unw'):
                atr['FILE_TYPE'] = '.unw'
            elif fbase.startswith(('coh','cor')):
                atr['FILE_TYPE'] = '.cor'
            elif fbase.startswith('phase_ifg'):
                atr['FILE_TYPE'] = '.int'
            elif 'dem' in fbase:
                atr['FILE_TYPE'] = 'dem'
            else:
                atr['FILE_TYPE'] = atr['file type']

    # UNIT
    k = atr['FILE_TYPE'].replace('.', '')
    if k == 'slc':
        atr['UNIT'] = 'i'
    elif 'UNIT' not in atr.keys():
        if datasetName and datasetName in datasetUnitDict.keys():
            atr['UNIT'] = datasetUnitDict[datasetName]
        elif k in datasetUnitDict.keys():
            atr['UNIT'] = datasetUnitDict[k]
        else:
            atr['UNIT'] = '1'

    # FILE_PATH
    atr['FILE_PATH'] = os.path.abspath(fname)

    if standardize:
        atr = readfile.standardize_metadata(atr)
    return atr


def check_template_auto_value(templateDict, auto_file='../defaults/minopy_template_defaults.cfg'):
    """Replace auto value based on the input auto config file."""
    # Read default template value and turn yes/no to True/False
    templateAutoFile = os.path.join(os.path.dirname(__file__), auto_file)
    templateAutoDict = readfile.read_template(templateAutoFile)

    # Update auto value of input template dict
    for key, value in templateDict.items():
        if value == 'auto' and key in templateAutoDict.keys():
            templateDict[key] = templateAutoDict[key]

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
def read_binary_file(fname, datasetName=None, box=None):
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
        if k in ['unw']:
            band = 2

        elif k in ['slc']:
            cpx_band = 'complex'

        elif k in ['los'] and datasetName and datasetName.startswith(('az', 'head')):
            band = 2

        elif k in ['incLocal']:
            band = 2
            if datasetName and 'local' not in datasetName.lower():
                band = 1

        elif datasetName:
            if datasetName.lower() == 'band2':
                band = 2
            elif datasetName.lower() == 'band3':
                band = 3

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
            data_type = 'complex64'
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

    # reading
    data = read_image(fname, box=box)

    if 'DATA_TYPE' not in atr:
        atr['DATA_TYPE'] = data_type
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
