#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Author:  Sara Mirzaee                                    #
############################################################
import os
import glob
import sys
import datetime
from mintpy.objects import (GEOMETRY_DSET_NAMES,
                            geometry,
                            sensor)
from mintpy.utils import readfile, ptime
import mintpy.load_data as mld
from miaplpy.objects.geometryStack import geometryDict
import miaplpy.objects.utils as mut
from miaplpy.objects.arg_parser import MiaplPyParser
import h5py
import numpy as np

#################################################################
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


#################################################################


def main(iargs=None):
    Parser = MiaplPyParser(iargs, script='load_slc')
    inps = Parser.parse()

    print(os.getcwd())

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1:-1])
        string = dateStr + " * " + msg
        print(string)

    os.chdir(inps.work_dir)

    # read input options
    iDict = mut.read_inps2dict(inps)
    
    # prepare metadata
    if not inps.no_metadata_check:
        mut.prepare_metadata(iDict)

    # skip data writing for aria as it is included in prep_aria
    if iDict['processor'] == 'aria':
        return

    iDict = mut.read_subset_box(iDict)
    #iDict = mld.read_subset_box(iDict)

    extraDict = mld.get_extra_metadata(iDict)

    # initiate objects
    stackObj = mut.read_inps_dict2slc_stack_dict_object(iDict)

    iDict['ds_name2key'] = datasetName2templateKey
    geomRadarObj, geomGeoObj = read_inps_dict2geometry_dict_object(iDict)

    # prepare write
    #updateMode, comp, box, boxGeo, xyStep, xyStepGeo = mut.print_write_setting(iDict)
    #updateMode, comp, box, boxGeo = mld.print_write_setting(iDict)
    updateMode = iDict['updateMode']
    comp = iDict['compression']
    box = iDict['box']
    if not iDict.get('geocoded', False):
         boxGeo = iDict['box4geo_lut']
    else:
         boxGeo = box

    if any([stackObj, geomRadarObj, geomGeoObj]) and not os.path.isdir(inps.out_dir):
        os.makedirs(inps.out_dir)
        print('create directory: {}'.format(inps.out_dir))

    # write
    if stackObj and mld.run_or_skip(inps.out_file[0], stackObj, box, updateMode=updateMode,
                                      xstep=iDict['xstep'], ystep=iDict['ystep']):
        print('-' * 50)
        stackObj.write2hdf5(outputFile=inps.out_file[0],
                            access_mode='a',
                            box=box,
                            xstep=iDict['xstep'],
                            ystep=iDict['ystep'],
                            compression=comp,
                            extra_metadata=extraDict)

    if geomRadarObj and mld.run_or_skip(inps.out_file[1], geomRadarObj, box, updateMode=updateMode,
                                          xstep=iDict['xstep'], ystep=iDict['ystep']):
        print('-' * 50)
        geomRadarObj.write2hdf5(outputFile=inps.out_file[1],
                                access_mode='a',
                                box=box,
                                xstep=iDict['xstep'],
                                ystep=iDict['ystep'],
                                compression='lzf',
                                extra_metadata=extraDict)

    if geomGeoObj and mld.run_or_skip(inps.out_file[2], geomGeoObj, boxGeo, updateMode=updateMode,
                                        xstep=iDict['xstep'], ystep=iDict['ystep']):
        print('-' * 50)
        geomGeoObj.write2hdf5(outputFile=inps.out_file[2],
                              access_mode='a',
                              box=boxGeo,
                              xstep=iDict['xstep'],
                              ystep=iDict['ystep'],
                              compression='lzf')
    if iDict['processor'] == 'gamma':
        add_lonlat_gamma(inps.out_file[2])
        dir_name = os.path.dirname(inps.out_file[2])
        this_dir = os.getcwd()
        os.chdir(dir_name)
        # run lookup_geo2radar.py to copy lon/lat to radar coordiantes
        script_name = 'lookup_geo2radar.py'
        print('-' * 50)
        cmd = f'{script_name} geometryGeo.h5'
        print(cmd)
        os.system(cmd)
        os.chdir(this_dir)

    return inps.out_file

#################################################################


def add_lonlat_gamma(h5file):
    """
    agg longitude/latitude to an existing geometry file
    :param h5file:
    """
    f = h5py.File(h5file, 'a')
    x_first = float(f.attrs['X_FIRST'])
    x_step = float(f.attrs['X_STEP'])
    y_first = float(f.attrs['Y_FIRST'])
    y_step = float(f.attrs['Y_STEP'])
    length = int(f.attrs['LENGTH'])
    width = int(f.attrs['WIDTH'])
    x_last = x_first+x_step*(width-1)
    y_last = y_first+y_step*(length-1)
    # x_last+x_step/100 to make sure the last element is not missing due to rounding
    x = np.arange(x_first, x_last+x_step/100, x_step)
    y = np.arange(y_first, y_last+y_step/100, y_step)
    lon, lat = np.meshgrid(x, y)

    compression = f['azimuthCoord'].compression
    for data, dsname in zip([lon, lat], ['longitude', 'latitude']):
        if dsname in f.keys():
            # TODO: add update mode
            raise Exception(f"name {dsname} already exists in {h5file}. Try deleting the file.")
        f.create_dataset(dsname,
                         data=data,
                         dtype='float32',
                         chunks=True,
                         compression=compression)
    f.close()
#########

def read_inps_dict2geometry_dict_object(inpsDict):
    # eliminate dsName by processor
    if not 'processor' in inpsDict and 'PROCESSOR'in inpsDict:
        inpsDict['processor'] = inpsDict['PROCESSOR']
    if inpsDict['processor'] in ['isce', 'doris']:
        datasetName2templateKey.pop('azimuthCoord')
        datasetName2templateKey.pop('rangeCoord')
        ending = '.xml'
    elif inpsDict['processor'] in ['roipac', 'gamma']:
        datasetName2templateKey.pop('latitude')
        datasetName2templateKey.pop('longitude')
        ending = ''
    elif inpsDict['processor'] in ['snap']:
        # check again when there is a SNAP product in radar coordiantes
        pass
    else:
        print('Un-recognized InSAR processor: {}'.format(inpsDict['processor']))

    # inpsDict --> dsPathDict
    print('-' * 50)
    print('searching geometry files info')
    print('input data files:')

    maxDigit = max([len(i) for i in list(datasetName2templateKey.keys())])
    dsPathDict = {}
    for dsName in [i for i in GEOMETRY_DSET_NAMES
                   if i in datasetName2templateKey.keys()]:
        key = datasetName2templateKey[dsName]
        if key in inpsDict.keys():
            files = sorted(glob.glob(str(inpsDict[key]) + ending))
            files = [item.split('.xml')[0] for item in files]
            if len(files) > 0:
                if dsName == 'bperp':
                    bperpDict = {}
                    for file in files:
                        date = ptime.yyyymmdd(os.path.basename(os.path.dirname(file)))
                        bperpDict[date] = file
                    dsPathDict[dsName] = bperpDict
                    print('{:<{width}}: {path}'.format(dsName,
                                                       width=maxDigit,
                                                       path=inpsDict[key]))
                    print('number of bperp files: {}'.format(len(list(bperpDict.keys()))))
                else:
                    dsPathDict[dsName] = files[0]
                    print('{:<{width}}: {path}'.format(dsName,
                                                       width=maxDigit,
                                                       path=files[0]))

    # Check required dataset
    dsName0 = GEOMETRY_DSET_NAMES[0]
    if dsName0 not in dsPathDict.keys():
        print('WARNING: No reqired {} data files found!'.format(dsName0))

    # metadata
    slcRadarMetadata = None
    slcKey = datasetName2templateKey['slc']
    if slcKey in inpsDict.keys():
        slcFiles = glob.glob(str(inpsDict[slcKey]))
        if len(slcFiles) > 0:
            atr = readfile.read_attribute(slcFiles[0])
            if 'Y_FIRST' not in atr.keys():
                slcRadarMetadata = atr.copy()

    # dsPathDict --> dsGeoPathDict + dsRadarPathDict
    dsNameList = list(dsPathDict.keys())
    dsGeoPathDict = {}
    dsRadarPathDict = {}
    for dsName in dsNameList:
        if dsName == 'bperp':
            atr = readfile.read_attribute(next(iter(dsPathDict[dsName].values())))
        else:
            if inpsDict['processor'] in ['gamma']:
                atr = mut.read_attribute(dsPathDict[dsName], metafile_ext='.rsc')
            else:
                atr = mut.read_attribute(dsPathDict[dsName].split('.xml')[0], metafile_ext='.xml')
        if 'Y_FIRST' in atr.keys():
            dsGeoPathDict[dsName] = dsPathDict[dsName]
        else:
            dsRadarPathDict[dsName] = dsPathDict[dsName]

    GeoMetadata = atr.copy()
    geomRadarObj = None
    geomGeoObj = None

    if len(dsRadarPathDict) > 0:
        geomRadarObj = geometryDict(processor=inpsDict['processor'],
                                    datasetDict=dsRadarPathDict,
                                    extraMetadata=slcRadarMetadata)
    if len(dsGeoPathDict) > 0:
        geomGeoObj = geometryDict(processor=inpsDict['processor'],
                                  datasetDict=dsGeoPathDict,
                                  extraMetadata=GeoMetadata)
    return geomRadarObj, geomGeoObj


#################################################################
if __name__ == '__main__':
    """
    loading a stack of InSAR pairs to and HDF5 file
    """
    main()
