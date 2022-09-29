#!/usr/bin/env python3
########################################################################
# Program is part of MiaplPy and a modified version of Mintpy.load_data #
# Author (Modified by): Sara Mirzaee                                   #
########################################################################

import os
import sys
import glob
import argparse
import warnings
import shutil
from miaplpy.defaults import auto_path
from mintpy.objects import (GEOMETRY_DSET_NAMES,
                            geometry,
                            IFGRAM_DSET_NAMES,
                            ifgramStack,
                            sensor)
from mintpy import load_data as mld
from mintpy.objects.stackDict import (geometryDict,
                                      ifgramStackDict,
                                      ifgramDict)
from mintpy.utils import readfile, ptime, utils as ut
from miaplpy.objects.utils import check_template_auto_value, read_subset_template2box
from mintpy import subset
import datetime

#################################################################
datasetName2templateKey = {'unwrapPhase'     : 'miaplpy.load.unwFile',
                           'coherence'       : 'miaplpy.load.corFile',
                           'connectComponent': 'miaplpy.load.connCompFile',
                           'wrapPhase'       : 'miaplpy.load.intFile',
                           'iono'            : 'miaplpy.load.ionoFile',
                           'height'          : 'miaplpy.load.demFile',
                           'latitude'        : 'miaplpy.load.lookupYFile',
                           'longitude'       : 'miaplpy.load.lookupXFile',
                           'azimuthCoord'    : 'miaplpy.load.lookupYFile',
                           'rangeCoord'      : 'miaplpy.load.lookupXFile',
                           'incidenceAngle'  : 'miaplpy.load.incAngleFile',
                           'azimuthAngle'    : 'miaplpy.load.azAngleFile',
                           'shadowMask'      : 'miaplpy.load.shadowMaskFile',
                           'waterMask'       : 'miaplpy.load.waterMaskFile',
                           'bperp'           : 'miaplpy.load.bperpFile'
                           }

DEFAULT_TEMPLATE = """template:
########## 1. Load ifgrams (--load to exit after this step)
{}\n
{}\n
{}\n
{}\n
""".format(auto_path.isceTopsAutoPath,
           auto_path.isceStripmapAutoPath,
           auto_path.roipacAutoPath,
           auto_path.gammaAutoPath)

TEMPLATE = """template:
########## 1. Load interferograms
## auto - automatic path pattern for Univ of Miami file structure
## load_data.py -H to check more details and example inputs.
## compression to save disk usage for ifgramStack.h5 file:
## no   - save   0% disk usage, fast [default]
## lzf  - save ~57% disk usage, relative slow
## gzip - save ~62% disk usage, very slow [not recommend]

miaplpy.load.processor      = auto  #[isce,snap,gamma,roipac], auto for isceTops
miaplpy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
miaplpy.load.compression    = auto  #[gzip / lzf / no], auto for no.
miaplpy.load.autoPath       = auto    # [yes, no] auto for no

##---------interferogram datasets:
miaplpy.load.unwFile        = auto  #[path2unw_file]
miaplpy.load.corFile        = auto  #[path2cor_file]
miaplpy.load.connCompFile   = auto  #[path2conn_file], optional
miaplpy.load.intFile        = auto  #[path2int_file], optional
miaplpy.load.ionoFile       = auto  #[path2iono_file], optional
"""

EXAMPLE = """example:
  load_ifgram.py -t PichinchaSenDT142.tempalte
  load_ifgram.py -t miaplpyApp.cfg
  load_ifgram.py -t miaplpyApp.cfg PichinchaSenDT142.txt --project PichinchaSenDT142
  load_ifgram.py -H #Show example input template for ISCE/ROI_PAC/GAMMA products
"""


def create_parser():
    """Create command line parser."""
    parser = argparse.ArgumentParser(description='Saving a stack of Interferograms to an HDF5 file',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=TEMPLATE+'\n'+EXAMPLE)
    parser.add_argument('-H', dest='print_example_template', action='store_true',
                        help='Print/Show the example template file for loading.')
    parser.add_argument('-t', '--template', type=str, nargs='+', dest='template_file',
                        help='template file with path info.')

    parser.add_argument('--project', type=str, dest='PROJECT_NAME',
                        help='project name of dataset for INSARMAPS Web Viewer')
    parser.add_argument('--processor', type=str, dest='processor',
                        choices={'isce', 'snap', 'gamma', 'roipac', 'doris', 'gmtsar'},
                        help='InSAR processor/software of the file', default='isce')
    parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                        help='Disable the update mode, or skip checking dataset already loaded.')
    parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                        help='compress loaded geometry while writing HDF5 file, default: None.')
    parser.add_argument('-o', '--output', type=str, nargs=3, dest='outfile',
                        default=['./inputs/ifgramStack.h5',
                                 './inputs/geometryRadar.h5',
                                 './inputs/geometryGeo.h5'],
                        help='output HDF5 file')

    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if inps.template_file:
        pass
    elif inps.print_example_template:
        raise SystemExit(DEFAULT_TEMPLATE)
    else:
        parser.print_usage()
        print(('{}: error: one of the following arguments are required:'
               ' -t/--template, -H'.format(os.path.basename(__file__))))
        print('{} -H to show the example template file'.format(os.path.basename(__file__)))
        sys.exit(1)

    for i, file in enumerate(inps.outfile):
        inps.outfile[i] = os.path.abspath(file)

    inps.outdir = os.path.dirname(inps.outfile[0])

    return inps


#################################################################
def read_inps2dict(inps):
    """Read input Namespace object info into inpsDict"""
    # Read input info into inpsDict
    inpsDict = vars(inps)
    inpsDict['PLATFORM'] = None
    auto_template = os.path.join(os.path.dirname(__file__), 'defaults/miaplpyApp_auto.cfg')

    # Read template file
    template = {}
    for fname in list(inps.template_file):
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

    msg = 'SAR platform/sensor : '
    sensor_name = sensor.project_name2sensor_name(str(inpsDict['PROJECT_NAME']))[0]
    if sensor_name:
        msg += str(sensor_name)
        inpsDict['PLATFORM'] = str(sensor_name)
    else:
        msg += 'unknown from project name "{}"'.format(inpsDict['PROJECT_NAME'])
    print(msg)
    
    # Here to insert code to check default file path for miami user
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(inpsDict['outfile'][0])))
    if inpsDict.get('autoPath', False):
        print(('check auto path setting for Univ of Miami users'
               ' for processor: {}'.format(inpsDict['processor'])))
        inpsDict = auto_path.get_auto_path(processor=inpsDict['processor'],
                                           work_dir=work_dir,
                                           template=inpsDict)
    return inpsDict


def read_subset_box(inpsDict):
    # Read subset info from template
    inpsDict['box'] = None
    inpsDict['box4geo_lut'] = None
    pix_box, geo_box = read_subset_template2box(inpsDict['template_file'][0])

    # Grab required info to read input geo_box into pix_box
    try:
        lookupFile = [glob.glob(str(inpsDict['miaplpy.load.lookupYFile']))[0],
                      glob.glob(str(inpsDict['miaplpy.load.lookupXFile']))[0]]
    except:
        lookupFile = None

    try:
        pathKey = [i for i in datasetName2templateKey.values()
                   if i in inpsDict.keys()][0]
        file = glob.glob(str(inpsDict[pathKey]))[0]
        atr = readfile.read_attribute(file)
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
            fnames = ut.get_file_list(inpsDict['miaplpy.load.unwFile'])
            pix_box = mld.update_box4files_with_inconsistent_size(fnames)

        if not pix_box:
            return inpsDict

    # geo_box --> pix_box
    coord = ut.coordinate(atr, lookup_file=lookupFile)

    if geo_box is not None:
        pix_box = (0, 0, int(atr['width']), int(atr['length']))    # coord.bbox_geo2radar(geo_box)
        pix_box = coord.check_box_within_data_coverage(pix_box)
        print('input bounding box of interest in lalo: {}'.format((geo_box[1], geo_box[0], geo_box[3], geo_box[2])))
    print('box to read for datasets in y/x: {}'.format(pix_box))

    # Get box for geocoded lookup table (for gamma/roipac)
    box4geo_lut = None
    if lookupFile is not None:
        atrLut = readfile.read_attribute(lookupFile[0])
        if not geocoded and 'Y_FIRST' in atrLut.keys():
            geo_box = coord.bbox_radar2geo(pix_box)
            box4geo_lut = ut.coordinate(atrLut).bbox_geo2radar(geo_box)
            print('box to read for geocoded lookup file in y/x: {}'.format(box4geo_lut))

    inpsDict['box'] = pix_box
    inpsDict['box4geo_lut'] = box4geo_lut
    return inpsDict


def prepare_metadata(inpsDict):
    processor = inpsDict['processor']
    script_name = 'prep_{}.py'.format(processor)
    print('-'*50)
    print('prepare metadata files for {} products'.format(processor))

    if processor in ['gamma', 'roipac', 'snap']:
        for key in [i for i in inpsDict.keys() if (i.startswith('miaplpy.load.') and i.endswith('File'))]:
            if len(glob.glob(str(inpsDict[key]))) > 0:
                cmd = '{} {}'.format(script_name, inpsDict[key])
                print(cmd)
                os.system(cmd)

    elif processor == 'isce':
        meta_files = sorted(glob.glob(inpsDict['miaplpy.load.metaFile']))
        if len(meta_files) < 1:
            warnings.warn('No input metadata file found: {}'.format(inpsDict['miaplpy.load.metaFile']))
        try:
            # metadata and auxliary data
            meta_file = meta_files[0]
            baseline_dir = inpsDict['miaplpy.load.baselineDir']
            geom_dir = os.path.dirname(inpsDict['miaplpy.load.demFile'])
            # observation
            obs_keys = ['miaplpy.load.unwFile', 'miaplpy.load.azOffFile']
            obs_keys = [i for i in obs_keys if i in inpsDict.keys()]
            obs_paths = [inpsDict[key] for key in obs_keys if inpsDict[key].lower() != 'auto']
            if len(obs_paths) > 0:
                obs_dir = os.path.dirname(os.path.dirname(obs_paths[0]))
                obs_file = os.path.basename(obs_paths[0])
            else:
                obs_dir = None
                obs_file = None

            # command line
            cmd = '{s} -m {m} -g {g}'.format(s=script_name, m=meta_file, g=geom_dir)
            if baseline_dir:
                cmd += ' -b {b} '.format(b=baseline_dir)
            if obs_dir is not None:
                cmd += ' -f {f} '.format(f=obs_dir + '/*/' + obs_file)
            print(cmd)
            os.system(cmd)
        except:
            pass
    return


def print_write_setting(inpsDict):
    updateMode = inpsDict['updateMode']
    comp = inpsDict['compression']
    print('-'*50)
    print('updateMode : {}'.format(updateMode))
    print('compression: {}'.format(comp))
    box = inpsDict['box']
    boxGeo = inpsDict['box4geo_lut']
    return updateMode, comp, box, boxGeo


def get_extra_metadata(inpsDict):
    """Extra metadata to be written into stack file"""
    extraDict = {}
    for key in ['PROJECT_NAME', 'PLATFORM']:
        if inpsDict[key]:
            extraDict[key] = inpsDict[key]
    for key in ['SUBSET_XMIN', 'SUBSET_YMIN']:
        if key in inpsDict.keys():
            extraDict[key] = inpsDict[key]
    return extraDict


#################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')
    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    work_dir = os.path.dirname(inps.outdir)
    #os.chdir(work_dir)

    # read input options
    inpsDict = read_inps2dict(inps)
    prepare_metadata(inpsDict)

    inpsDict = read_subset_box(inpsDict)
    extraDict = get_extra_metadata(inpsDict)
    
    if not 'PLATFORM' in extraDict:
        slcStack = os.path.join(os.path.dirname(work_dir), 'inputs/slcStack.h5')
        atr = readfile.read_attribute(slcStack)
        if 'PLATFORM' in atr:
            extraDict['PLATFORM'] = atr['PLATFORM']

    # initiate objects
    inpsDict['dset_name2template_key'] = datasetName2templateKey
    inpsDict['only_load_geometry'] = False
    if 'miaplpy.load.unwFile' in datasetName2templateKey.values():
        datasetName2templateKey['unwrapMintPy'] = 'mintpy.load.unwFile'
    elif 'miaplpy.load.ionUnwFile' in datasetName2templateKey.values():
        datasetName2templateKey['ionMintPy'] = 'mintpy.load.ionUnwFile'
    stackObj = mld.read_inps_dict2ifgram_stack_dict_object(inpsDict, datasetName2templateKey)

    # prepare wirte
    updateMode, comp, box, boxGeo = print_write_setting(inpsDict)
    box = None
    boxGeo = None
    if stackObj and not os.path.isdir(inps.outdir):
        os.makedirs(inps.outdir)
        print('create directory: {}'.format(inps.outdir))
    # write
    if stackObj and mld.run_or_skip(inps.outfile[0], stackObj, box,
                                     updateMode=updateMode, xstep=inpsDict['xstep'],
                                     ystep=inpsDict['ystep']):
        print('-'*50)
        stackObj.write2hdf5(outputFile=inps.outfile[0],
                            access_mode='w',
                            box=box,
                            compression=comp,
                            extra_metadata=extraDict)

    geo_files = ['geometryRadar.h5', 'geometryGeo.h5']
    copy_file = False
    for geometry_file_2 in geo_files:
        geometry_file = os.path.join(os.path.dirname(work_dir), 'inputs', geometry_file_2)
        if os.path.exists(geometry_file):
            copy_file = True
            break

    if copy_file:
        if not os.path.exists(os.path.join(work_dir, 'inputs/{}'.format(geometry_file_2))):
            shutil.copyfile(geometry_file, os.path.join(work_dir, 'inputs/{}'.format(geometry_file_2)))
    else:
        geomRadarObj, geomGeoObj = mld.read_inps_dict2geometry_dict_object(inpsDict)
        if geomRadarObj and mld.run_or_skip(inps.outfile[1], geomRadarObj, box,
                                          updateMode=updateMode,
                                          xstep=inpsDict['xstep'],
                                          ystep=inpsDict['ystep']):
            print('-' * 50)
            geomRadarObj.write2hdf5(outputFile=inps.outfile[1],
                                    access_mode='w',
                                    box=box,
                                    xstep=inpsDict['xstep'],
                                    ystep=inpsDict['ystep'],
                                    compression='lzf',
                                    extra_metadata=extraDict)

        if geomGeoObj and mld.run_or_skip(inps.outfile[2], geomGeoObj, boxGeo,
                                        updateMode=updateMode,
                                        xstep=inpsDict['xstep'],
                                        ystep=inpsDict['ystep']):
            print('-' * 50)
            geomGeoObj.write2hdf5(outputFile=inps.outfile[2],
                                  access_mode='w',
                                  box=boxGeo,
                                  xstep=inpsDict['xstep'],
                                  ystep=inpsDict['ystep'],
                                  compression='lzf')


    # check loading result
    if not os.path.exists(os.path.join(work_dir, 'smallbaselineApp.cfg')):
        shutil.copyfile(os.path.join(os.path.dirname(work_dir), 'custom_smallbaselineApp.cfg'),
                        os.path.join(work_dir, 'smallbaselineApp.cfg'))
    load_complete, stack_file, geom_file = ut.check_loaded_dataset(work_dir=work_dir, print_msg=True)[0:3]

    # add custom metadata (optional)
    customTemplate = inps.template_file[0]
    if customTemplate:
        print('updating {}, {} metadata based on custom template file: {}'.format(
            os.path.basename(stack_file),
            os.path.basename(geom_file),
            os.path.basename(customTemplate)))
        # use ut.add_attribute() instead of add_attribute.py because of
        # better control of special metadata, such as SUBSET_X/YMIN
        ut.add_attribute(stack_file, inpsDict)
        ut.add_attribute(geom_file, inpsDict)

    ut.add_attribute(stack_file, extraDict)

    # if not load_complete, plot and raise exception
    if not load_complete:
        # go back to original directory
        print('Go back to directory:', work_dir)
        os.chdir(work_dir)

        # raise error
        msg = 'step load_ifgram: NOT all required dataset found, exit.'
        raise SystemExit(msg)

    return inps.outfile


#################################################################
if __name__ == '__main__':
    """
    loading a stack of InSAR pairs to and HDF5 file
    """
    main()
