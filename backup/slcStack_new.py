#! /usr/bin/env python
###############################################################################
#  slcStack.py
#  Author:   Sara Mirzaee
###############################################################################

import h5py
import time
import os
import numpy as np

class slcStack:
    """
    Stack of SLCs
    written based on MintPy stack objects
    """
    def __init__(self, file=None):
        self.file = file
        self.name = 'slc'
        
    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print('close slcStack file: {}'.format(os.path.basename(self.file)))
        except:
            pass
        return None
    
    def open_hdf5(self, mode='a'):
        print('open {} in {} mode'.format(self.file, mode))
        self.f = h5py.File(self.file, mode)
        return self.f
    
    def open(self, print_msg=True):
        if print_msg:
            print('open {} file: {}'.format(self.name, os.path.basename(self.file)))
        self.get_metadata()
        self.get_size()
        self.get_date_list()
        self.numPixel = self.length * self.width

        self.times = np.array([dt(*time.strptime(i, "%Y%m%d")[0:5]) for i in self.dateList])
        
        return None
    
    def get_metadata(self):
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
            dates = f['dates'][:]
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value

        # ref_date/index
        dateList = [i.decode('utf8') for i in dates]
        self.metadata['START_DATE'] = dateList[0]
        self.metadata['END_DATE'] = dateList[-1]
        return self.metadata
    
    def get_size(self):
        with h5py.File(self.file, 'r') as f:
            self.numDate, self.length, self.width = f[self.name].shape
        return self.numDate, self.length, self.width

    def get_date_list(self):
        with h5py.File(self.file, 'r') as f:
            self.dateList = [i.decode('utf8') for i in f['dates'][:]]
        return self.dateList
    
    def read(self, datasetName=None, box=None, print_msg=True):
        """Read dataset from slc file
        Parameters: self : slcStack object
                    datasetName : (list of) string in YYYYMMDD format
                    box : tuple of 4 int, indicating x0,y0,x1,y1 of range
        Returns:    data : 2D or 3D dataset
        Examples:   
                    tsobj = slcStack('slcStack.h5')
                    data = tsobj.read(datasetName='20161020')
                    data = tsobj.read(datasetName='20161020', box=(100,300,500,800))
                    data = tsobj.read(datasetName=['20161020','20161026','20161101'])
                    data = tsobj.read(box=(100,300,500,800))
        """
        if print_msg:
            print('reading {} data from file: {} ...'.format(self.name, self.file))
        self.open(print_msg=False)

        # convert input datasetName into list of dates
        if not datasetName or datasetName == 'slc':
            datasetName = []
        elif isinstance(datasetName, str):
            datasetName = [datasetName]
        datasetName = [i.replace('slc', '').replace('-', '') for i in datasetName]

        with h5py.File(self.file, 'r') as f:
            ds = f[self.name]
            if isinstance(ds, h5py.Group):  # support for old mintpy files
                ds = ds[self.name]

            # Get dateFlag - mark in time/1st dimension
            dateFlag = np.zeros((self.numDate), dtype=np.bool_)
            if not datasetName:
                dateFlag[:] = True
            else:
                for e in datasetName:
                    dateFlag[self.dateList.index(e)] = True

            # Get Index in space/2_3 dimension
            if box is None:
                box = [0, 0, self.width, self.length]

            data = ds[dateFlag, box[1]:box[3], box[0]:box[2]]
            data = np.squeeze(data)
        return data
    
    def layout_hdf5(self, dsNameDict, metadata, compression=None):
        print('-'*50)
        print('create HDF5 file {} with w mode'.format(self.file))
        f = h5py.File(self.file, "w")

        for key in dsNameDict.keys():
            print("create dataset: {d:<25} of {t:<25} in size of {s}".format(
                d=key,
                t=str(dsNameDict[key][0]),
                s=dsNameDict[key][1]))

            f.create_dataset(key,
                             shape=dsNameDict[key][1],
                             dtype=dsNameDict[key][0],
                             chunks=True,
                             compression=compression)

        # write attributes
        metadata = dict(metadata)
        metadata['FILE_TYPE'] = self.name
        for key in metadata.keys():
            f.attrs[key] = metadata[key]

        print('close HDF5 file {}'.format(self.file))
        f.close()
        return self.file
    
    def write2hdf5_block(self, data, datasetName, block=None, mode='a'):
        """Write data to existing HDF5 dataset in disk block by block.
        Parameters: data : np.ndarray 1/2/3D matrix
                    datasetName : str, dataset name
                    block : list of 2/4/6 int, for
                        [zStart, zEnd,
                         yStart, yEnd,
                         xStart, xEnd]
                    mode : str, open mode
        Returns: self.file
        """
        if block is None:
            # data shape
            if isinstance(data, list):
                shape=(len(data),)
            else:
                shape = data.shape

            if len(shape) ==1:
                block = [0, shape[0]]
            elif len(shape) == 2:
                block = [0, shape[0],
                         0, shape[1]]
            elif len(shape) == 3:
                block = [0, shape[0],
                         0, shape[1],
                         0, shape[2]]

        print('open {} in {} mode'.format(self.file, mode))
        f = h5py.File(self.file, mode)

        print("writing dataset /{:<25} block: {}".format(datasetName, block))
        if len(block) == 6:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3],
                           block[4]:block[5]] = data

        elif len(block) == 4:
            f[datasetName][block[0]:block[1],
                           block[2]:block[3]] = data

        elif len(block) == 2:
            f[datasetName][block[0]:block[1]] = data

        f.close()
        print('close HDF5 file {}'.format(self.file))
        return self.file
    
    def write2hdf5(self, data, outFile=None, dates=None, metadata=None, compression=None):
        """
        Parameters: data  : 3D array of float32
                    dates : 1D array/list of string in YYYYMMDD format
                    metadata : dict
                    outFile : string
                    compression : string or None
        Returns: outFile : string
        Examples:
            ##Generate a new slcStack file
            tsobj = slcStack('slcStack.h5')
            tsobj.write2hdf5(data, dates=dateList, metadata=atr)
        """

        if not outFile:
            outFile = self.file
        data = np.array(data, dtype='c16')
        dates = np.array(dates, dtype=np.string_)
        metadata = dict(metadata)
        metadata['FILE_TYPE'] = self.name

        # 3D dataset - slcStack
        print('create slcStack HDF5 file: {} with w mode'.format(outFile))
        f = h5py.File(outFile, 'w')
        print(('create dataset /slcStack of {t:<10} in size of {s} '
               'with compression={c}').format(t=str(data.dtype),
                                              s=data.shape,
                                              c=compression))
        f.create_dataset('slc', data=data, chunks=True, compression=compression)

        # 1D dataset - date / bperp
        print('create dataset /dates      of {:<10} in size of {}'.format(str(dates.dtype), dates.shape))
        f.create_dataset('date', data=dates)

        # Attributes
        for key, value in metadata.items():
            f.attrs[key] = str(value)

        f.close()
        print('finished writing to {}'.format(outFile))
        return outFile
    
    