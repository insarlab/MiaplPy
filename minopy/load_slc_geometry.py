#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee                                    #
############################################################
import os
import glob
import sys
import datetime
from mintpy.objects import (geometryDatasetNames,
                            geometry,
                            sensor)
from minopy.objects.slcStack import slcStack
from minopy.objects.geometryStack import geometryDict
from mintpy.utils import readfile, ptime, utils as ut
import mintpy.load_data as mld

import minopy.objects.utils as mut
from minopy.objects.arg_parser import MinoPyParser

#################################################################
datasetName2templateKey = {'slc': 'minopy.load.slcFile',
                           'unwrapPhase': 'minopy.load.unwFile',
                           'coherence': 'minopy.load.corFile',
                           'connectComponent': 'minopy.load.connCompFile',
                           'wrapPhase': 'minopy.load.intFile',
                           'iono': 'minopy.load.ionoFile',
                           'height': 'minopy.load.demFile',
                           'latitude': 'minopy.load.lookupYFile',
                           'longitude': 'minopy.load.lookupXFile',
                           'azimuthCoord': 'minopy.load.lookupYFile',
                           'rangeCoord': 'minopy.load.lookupXFile',
                           'incidenceAngle': 'minopy.load.incAngleFile',
                           'azimuthAngle': 'minopy.load.azAngleFile',
                           'shadowMask': 'minopy.load.shadowMaskFile',
                           'waterMask': 'minopy.load.waterMaskFile',
                           'bperp': 'minopy.load.bperpFile'
                           }


#################################################################


def main(iargs=None):
    Parser = MinoPyParser(iargs, script='load_slc')
    inps = Parser.parse()

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

    extraDict = mld.get_extra_metadata(iDict)

    # initiate objects
    stackObj = mut.read_inps_dict2slc_stack_dict_object(iDict)
    
    geomRadarObj, geomGeoObj = read_inps_dict2geometry_dict_object(iDict)

    # prepare write
    updateMode, comp, box, boxGeo, xyStep, xyStepGeo = mut.print_write_setting(iDict)

    if any([stackObj, geomRadarObj, geomGeoObj]) and not os.path.isdir(inps.out_dir):
        os.makedirs(inps.out_dir)
        print('create directory: {}'.format(inps.out_dir))

    # write
    if stackObj and update_object(inps.out_file[0], stackObj, box, updateMode=updateMode):
        print('-' * 50)
        stackObj.write2hdf5(outputFile=inps.out_file[0],
                            access_mode='a',
                            box=box,
                            xstep=xyStep[0],
                            ystep=xyStep[1],
                            compression=comp,
                            extra_metadata=extraDict)

    if geomRadarObj and update_object(inps.out_file[1], geomRadarObj, box, updateMode=updateMode):
        print('-' * 50)
        geomRadarObj.write2hdf5(outputFile=inps.out_file[1],
                                access_mode='a',
                                box=box,
                                xstep=xyStep[0],
                                ystep=xyStep[1],
                                compression='lzf',
                                extra_metadata=extraDict)

    if geomGeoObj and update_object(inps.out_file[2], geomGeoObj, boxGeo, updateMode=updateMode):
        print('-' * 50)
        geomGeoObj.write2hdf5(outputFile=inps.out_file[2],
                              access_mode='a',
                              box=boxGeo,
                              xstep=xyStepGeo[0],
                              ystep=xyStepGeo[1],
                              compression='lzf')
    return inps.out_file

#################################################################


def update_object(outFile, inObj, box, updateMode=True):
    """Do not write h5 file if: 1) h5 exists and readable,
                                2) it contains all date12 from slcStackDict,
                                            or all datasets from geometryDict"""
    write_flag = True
    if updateMode and ut.run_or_skip(outFile, check_readable=True) == 'skip':
        if inObj.name == 'slc':
            in_size = inObj.get_size(box=box)[1:]
            in_date_list = inObj.get_date_list()

            outObj = slcStack(outFile)
            out_size = outObj.get_size()[1:]
            # out_date12_list = outObj.get_date12_list(dropIfgram=False)
            out_date_list = outObj.get_date_list()

            if out_size == in_size and set(in_date_list).issubset(set(out_date_list)):
                print(('All date12   exists in file {} with same size as required,'
                       ' no need to re-load.'.format(os.path.basename(outFile))))
                write_flag = False

        elif inObj.name == 'geometry':
            outObj = geometry(outFile)
            outObj.open(print_msg=False)
            if (outObj.get_size() == inObj.get_size(box=box)
                    and all(i in outObj.datasetNames for i in inObj.get_dataset_list())):
                print(('All datasets exists in file {} with same size as required,'
                       ' no need to re-load.'.format(os.path.basename(outFile))))
                write_flag = False
    return write_flag


def read_inps_dict2geometry_dict_object(inpsDict):
    # eliminate dsName by processor
    if not 'processor' in inpsDict and 'PROCESSOR'in inpsDict:
        inpsDict['processor'] = inpsDict['PROCESSOR']
    if inpsDict['processor'] in ['isce', 'doris']:
        datasetName2templateKey.pop('azimuthCoord')
        datasetName2templateKey.pop('rangeCoord')
    elif inpsDict['processor'] in ['roipac', 'gamma']:
        datasetName2templateKey.pop('latitude')
        datasetName2templateKey.pop('longitude')
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
    for dsName in [i for i in geometryDatasetNames
                   if i in datasetName2templateKey.keys()]:
        key = datasetName2templateKey[dsName]
        if key in inpsDict.keys():
            files = sorted(glob.glob(str(inpsDict[key]) + '.xml'))
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
    dsName0 = geometryDatasetNames[0]
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
            atr = mut.read_attribute(dsPathDict[dsName].split('.xml')[0], metafile_ext='.xml')
        if 'Y_FIRST' in atr.keys():
            dsGeoPathDict[dsName] = dsPathDict[dsName]
        else:
            dsRadarPathDict[dsName] = dsPathDict[dsName]

    geomRadarObj = None
    geomGeoObj = None

    if len(dsRadarPathDict) > 0:
        geomRadarObj = geometryDict(processor=inpsDict['processor'],
                                    datasetDict=dsRadarPathDict,
                                    extraMetadata=slcRadarMetadata)
    if len(dsGeoPathDict) > 0:
        geomGeoObj = geometryDict(processor=inpsDict['processor'],
                                  datasetDict=dsGeoPathDict,
                                  extraMetadata=None)
    return geomRadarObj, geomGeoObj


#################################################################
if __name__ == '__main__':
    """
    loading a stack of InSAR pairs to and HDF5 file
    """
    main()
