############################################################
# Program is part of MintPy                                #
# Author: Sara Mirzaee, Heresh Fattahi, Zhang Yunjun       #
############################################################


import os
import time
import warnings
import h5py
import numpy as np
#from minopy.prep_slc_isce import read_attribute
from minopy.objects.utils import read_attribute, read as read_geo
try:
    from skimage.transform import resize
except ImportError:
    raise ImportError('Could not import skimage!')

from mintpy.objects import (dataTypeDict,
                            geometryDatasetNames,
                            ifgramDatasetNames)
from mintpy.utils import readfile, ptime, utils as ut, attribute as attr
from mintpy.objects.stackDict import geometryDict as GDict, read_isce_bperp_file

BOOL_ZERO = np.bool_(0)
INT_ZERO = np.int16(0)
FLOAT_ZERO = np.float32(0.0)
CPX_ZERO = np.complex64(0.0)

dataType = np.float32


class geometryDict(GDict):

    def __init__(self, name='geometry', processor=None, datasetDict={}, extraMetadata=None):
        self.name = name
        self.processor = processor
        self.datasetDict = datasetDict
        self.extraMetadata = extraMetadata

        # get extra metadata from geometry file if possible
        self.dsNames = list(self.datasetDict.keys())
        if not self.extraMetadata:
            dsFile = self.datasetDict[self.dsNames[0]] 
            metadata = read_attribute(dsFile.split('.xml')[0], metafile_ext='.rsc')
            #metadata = read_attribute(dsFile, metafile_ext='.rsc')
            if all(i in metadata.keys() for i in ['STARTING_RANGE', 'RANGE_PIXEL_SIZE']):
                self.extraMetadata = metadata

    def read(self, family, box=None): #, xstep=1, ystep=1):
        #super().read(family, box, xstep=xstep, ystep=ystep)
        if self.file.endswith('.h5'):
            dsName = None
        else:
            dsName = family
        self.file = self.datasetDict[family].split('.xml')[0]
        data, metadata = read_geo(self.file,
                                  datasetName=dsName,
                                 box=box)
        return data, metadata
    
    def get_size(self, family=None, box=None, xstep=1, ystep=1):
        if not family:
            family = [i for i in self.datasetDict.keys() if i != 'bperp'][0]
        self.file = self.datasetDict[family]
        metadata = read_attribute(self.file.split('.xml')[0], metafile_ext='.rsc')
        #metadata = read_attribute(self.file, metafile_ext='.rsc')
        # update due to subset
        if box:
            length = box[3] - box[1]
            width = box[2] - box[0]
        else:
            length = int(metadata['LENGTH'])
            width = int(metadata['WIDTH'])

        # update due to multilook
        length = length // ystep
        width = width // xstep
        return length, width

    def get_dataset_list(self):
        self.datasetList = list(self.datasetDict.keys())
        return self.datasetList

    def get_metadata(self, family=None):
        if not family:
            family = [i for i in self.datasetDict.keys() if i != 'bperp'][0]
        self.file = self.datasetDict[family]
        self.metadata = read_attribute(self.file.split('.xml')[0], metafile_ext='.rsc')
        #self.metadata = read_attribute(self.file, metafile_ext='.rsc')
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')

        # if self.processor is None:
        #    ext = self.file.split('.')[-1]
        #    if 'PROCESSOR' in self.metadata.keys():
        #        self.processor = self.metadata['PROCESSOR']
        #    elif os.path.exists(self.file+'.xml'):
        #        self.processor = 'isce'
        #    elif os.path.exists(self.file+'.rsc'):
        #        self.processor = 'roipac'
        #    elif os.path.exists(self.file+'.par'):
        #        self.processor = 'gamma'
        #    elif ext == 'grd':
        #        self.processor = 'gmtsar'
        #    #what for DORIS/SNAP
        #    else:
        #        self.processor = 'isce'
        #self.metadata['PROCESSOR'] = self.processor
        return self.metadata

    def write2hdf5(self, outputFile='geometryRadar.h5', access_mode='w', box=None,
                   xstep=1, ystep=1, compression='lzf', extra_metadata=None):
        '''
        /                        Root level
        Attributes               Dictionary for metadata. 'X/Y_FIRST/STEP' attribute for geocoded.
        /height                  2D array of float32 in size of (l, w   ) in meter.
        /latitude (azimuthCoord) 2D array of float32 in size of (l, w   ) in degree.
        /longitude (rangeCoord)  2D array of float32 in size of (l, w   ) in degree.
        /incidenceAngle          2D array of float32 in size of (l, w   ) in degree.
        /slantRangeDistance      2D array of float32 in size of (l, w   ) in meter.
        /azimuthAngle            2D array of float32 in size of (l, w   ) in degree. (optional)
        /shadowMask              2D array of bool    in size of (l, w   ).           (optional)
        /waterMask               2D array of bool    in size of (l, w   ).           (optional)
        /bperp                   3D array of float32 in size of (n, l, w) in meter   (optional)
        /date                    1D array of string  in size of (n,     ) in YYYYMMDD(optional)
        ...
        '''
        if len(self.datasetDict) == 0:
            print('No dataset file path in the object, skip HDF5 file writing.')
            return None

        self.outputFile = outputFile
        f = h5py.File(self.outputFile, access_mode)
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))

        #groupName = self.name
        #group = f.create_group(groupName)
        #print('create group   /{}'.format(groupName))

        maxDigit = max([len(i) for i in geometryDatasetNames])
        length, width = self.get_size(box=box, xstep=xstep, ystep=ystep)
        #self.length, self.width = self.get_size()

        ###############################
        for dsName in self.dsNames:
            # 3D datasets containing bperp
            if dsName == 'bperp':
                self.dateList = list(self.datasetDict[dsName].keys())
                dsDataType = dataType
                self.numDate = len(self.dateList)
                dsShape = (self.numDate, length, width)
                ds = f.create_dataset(dsName,
                                      shape=dsShape,
                                      maxshape=(None, dsShape[1], dsShape[2]),
                                      dtype=dsDataType,
                                      chunks=True,
                                      compression=compression)
                print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                       ' with compression = {c}').format(d=dsName,
                                                         w=maxDigit,
                                                         t=str(dsDataType),
                                                         s=dsShape,
                                                         c=str(compression)))

                print('read coarse grid baseline files and linear interpolate into full resolution ...')
                prog_bar = ptime.progressBar(maxValue=self.numDate)
                for i in range(self.numDate):
                    fname = self.datasetDict[dsName][self.dateList[i]]
                    data = read_isce_bperp_file(fname=fname,
                                                full_shape=self.get_size(),
                                                box=box,
                                                xstep=xstep,
                                                ystep=ystep)
                    ds[i, :, :] = data
                    prog_bar.update(i+1, suffix=self.dateList[i])
                prog_bar.close()

                # Write 1D dataset date
                dsName = 'date'
                dsShape = (self.numDate,)
                dsDataType = np.string_
                print(('create dataset /{d:<{w}} of {t:<25}'
                       ' in size of {s}').format(d=dsName,
                                                 w=maxDigit,
                                                 t=str(dsDataType),
                                                 s=dsShape))
                data = np.array(self.dateList, dtype=dsDataType)
                if not dsName in f.keys():
                    ds = f.create_dataset(dsName, data=data)

            # 2D datasets containing height, latitude, incidenceAngle, shadowMask, etc.
            else:
                dsDataType = dataType
                if dsName.lower().endswith('mask'):
                    dsDataType = np.bool_
                dsShape = (length, width)
                print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                       ' with compression = {c}').format(d=dsName,
                                                         w=maxDigit,
                                                         t=str(dsDataType),
                                                         s=dsShape,
                                                         c=str(compression)))

                data = np.array(self.read(family=dsName, box=box)[0], dtype=dsDataType)
                if not dsName in f.keys():
                    ds = f.create_dataset(dsName,
                                          data=data,
                                          chunks=True,
                                          compression=compression)

        ###############################
        # Generate Dataset if not existed in binary file: incidenceAngle, slantRangeDistance
        for dsName in [i for i in ['incidenceAngle', 'slantRangeDistance']
                       if i not in self.dsNames]:
            # Calculate data
            data = None
            if dsName == 'incidenceAngle':
                data = self.get_incidence_angle(box=box)
            elif dsName == 'slantRangeDistance':
                data = self.get_slant_range_distance(box=box)

            # Write dataset
            if data is not None:
                dsShape = data.shape
                dsDataType = dataType
                print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                       ' with compression = {c}').format(d=dsName,
                                                         w=maxDigit,
                                                         t=str(dsDataType),
                                                         s=dsShape,
                                                         c=str(compression)))
                if not dsName in f.keys():
                    ds = f.create_dataset(dsName,
                                          data=data,
                                          dtype=dataType,
                                          chunks=True,
                                          compression=compression)

        ###############################
        # Attributes
        self.get_metadata()
        if extra_metadata:
            self.metadata.update(extra_metadata)
            print('add extra metadata: {}'.format(extra_metadata))
        self.metadata = attr.update_attribute4subset(self.metadata, box)
        self.metadata['FILE_TYPE'] = self.name
        for key, value in self.metadata.items():
            f.attrs[key] = value

        f.close()
        print('Finished writing to {}'.format(self.outputFile))
        return self.outputFile
