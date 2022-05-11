#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Author:  Sara Mirzaee, Zhang Yunjun, Heresh Fattahi                    #
############################################################
# Modified from prep4timeseries.py in ISCE-2.2.0/contrib/stack/topsStack

import os, sys
import warnings
import logging
warnings.filterwarnings("ignore")

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()

from isceobj.Planet.Planet import Planet
import glob
import shelve
import argparse
import numpy as np
from miaplpy.objects.utils import read_attribute, read
from mintpy.utils import isce_utils, ptime, readfile, writefile, utils as ut
enablePrint()


EXAMPLE = """example:
  prep_slc_isce.py -s ./merged/SLC -m ./reference/IW1.xml -b ./baselines -g ./merged/geom_reference  #for topsStack
  prep_slc_isce.py -s ./merged/SLC -m .merged/SLC/20190510/referenceShelve/data.dat -b ./baselines -g ./merged/geom_reference  #for stripmapStack
  """

GEOMETRY_PREFIXS = ['hgt', 'lat', 'lon', 'los', 'shadowMask', 'waterMask', 'incLocal']

def create_parser():
    """Command line parser."""
    parser = argparse.ArgumentParser(description='Prepare ISCE metadata files.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('-s', '--slc-dir', dest='slcDir', type=str, default=None,
                        help='The directory which contains all SLCs\n'+
                             'e.g.: $PROJECT_DIR/merged/SLC')
    parser.add_argument('-f', '--file-pattern', nargs = '+', dest='slcFiles', type=str,
                        default=['*.slc.full'],
                        help='A list of files that will be used in miaplpy\n'
                             'e.g.: 20180705.slc.full')
    parser.add_argument('-m', '--meta-file', dest='metaFile', type=str, default=None,
                        help='Metadata file to extract common metada for the stack:\n'
                             'e.g.: for ISCE/topsStack: reference/IW3.xml')
    parser.add_argument('-b', '--baseline-dir', dest='baselineDir', type=str, default=None,
                        help=' directory with baselines ')
    parser.add_argument('-g', '--geometry-dir', dest='geometryDir', type=str, default=None,
                        help=' directory with geometry files ')
    parser.add_argument('--geom-files', dest='geometryFiles', type=str, nargs='*',
                        default=['{}.rdr'.format(i) for i in GEOMETRY_PREFIXS],
                        help='List of geometry file basenames. Default: %(default)s.\n'
                             'All geometry files need to be in the same directory.')
    parser.add_argument('--force', dest='update_mode', action='store_false',
                        help='Force to overwrite all .rsc metadata files.')
    return parser


def cmd_line_parse(iargs = None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if all(not i for i in [inps.slcDir, inps.geometryDir, inps.metaFile]):
        parser.print_usage()
        raise SystemExit('error: at least one of the following arguments are required: -s, -g, -m')
    return inps


#########################################################################
def load_product(xmlname):
    """Load the product using Product Manager."""
    from iscesys.Component.ProductManager import ProductManager as PM
    pm = PM()
    pm.configure()
    obj = pm.loadProduct(xmlname)
    return obj


'''
def extract_tops_metadata(xml_file):
    """Read metadata from xml file for Sentinel-1/TOPS
    Parameters: xml_file : str, path of the .xml file, i.e. reference/IW1.xml
    Returns:    meta     : dict, metadata
    """
    import isce
    from isceobj.Planet.Planet import Planet

    obj = load_product(xml_file)
    burst = obj.bursts[0]
    burstEnd = obj.bursts[-1]

    metadata = {}
    metadata['prf'] = burst.prf
    metadata['startUTC'] = burst.burstStartUTC
    metadata['stopUTC'] = burstEnd.burstStopUTC
    metadata['radarWavelength'] = burst.radarWavelength
    metadata['rangePixelSize'] = burst.rangePixelSize
    metadata['startingRange'] = burst.startingRange
    metadata['passDirection'] = burst.passDirection
    metadata['polarization'] = burst.polarization
    metadata['trackNumber'] = burst.trackNumber
    metadata['orbitNumber'] = burst.orbitNumber

    time_seconds = (burst.burstStartUTC.hour * 3600.0 +
                    burst.burstStartUTC.minute * 60.0 +
                    burst.burstStartUTC.second)
    metadata['CENTER_LINE_UTC'] = time_seconds

    orbit = burst.orbit
    peg = orbit.interpolateOrbit(burst.sensingMid, method='hermite')

    Vs = np.linalg.norm(peg.getVelocity())
    metadata['satelliteSpeed'] = Vs
    metadata['azimuthPixelSize'] = Vs*burst.azimuthTimeInterval

    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    refElp.setSCH(llh[0], llh[1], orbit.getENUHeading(burst.sensingMid))
    metadata['earthRadius'] = refElp.pegRadCur
    metadata['altitude'] = llh[2]

    # for Sentinel-1
    metadata['beam_mode'] = 'IW'
    metadata['swathNumber'] = burst.swathNumber

    # Sentinel-1 TOPS spatial resolution
    iw_str = 'IW2'
    if os.path.basename(xml_file).startswith('IW'):
        iw_str = os.path.splitext(os.path.basename(xml_file))[0]
    metadata['azimuthResolution'] = isce_utils.S1_TOPS_RESOLUTION[iw_str]['azimuthResolution']
    metadata['rangeResolution'] = isce_utils.S1_TOPS_RESOLUTION[iw_str]['rangeResolution']

    # 1. multipel subswaths
    xml_files = glob.glob(os.path.join(os.path.dirname(xml_file), 'IW*.xml'))
    if len(xml_files) > 1:
        swath_num = [load_product(fname).bursts[0].swathNumber for fname in xml_files]
        metadata['swathNumber'] = ''.join(str(i) for i in sorted(swath_num))

    # 2. calculate ASF frame number for Sentinel-1
    metadata['firstFrameNumber'] = int(0.2 * (burst.burstStartUTC - obj.ascendingNodeTime).total_seconds())
    metadata['lastFrameNumber'] = int(0.2 * (burstEnd.burstStopUTC - obj.ascendingNodeTime).total_seconds())
    return metadata, burst


def extract_stripmap_metadata(meta_file):
    """Read metadata from shelve file for StripMap stack from ISCE
    Parameters: meta_file : str, path of the shelve file, i.e. referenceShelve/data.dat
    Returns:    meta      : dict, metadata
    """

    if os.path.basename(meta_file) == "data.dat":    #shelve file from stripmapStack
        fbase = os.path.splitext(meta_file)[0]
        with shelve.open(fbase, flag='r') as mdb:
            frame = mdb['frame']

    elif meta_file.endswith(".xml"):   #XML file from stripmapApp
        frame = load_product(meta_file)

    else:
        raise ValueError('un-recognized isce/stripmap metadata file: {}'.format(meta_file))

    metadata = {}
    metadata['prf'] = frame.PRF
    metadata['startUTC'] = frame.sensingStart
    metadata['stopUTC'] = frame.sensingStop
    metadata['radarWavelength'] = frame.radarWavelegth
    metadata['rangePixelSize'] = frame.instrument.rangePixelSize
    metadata['startingRange'] = frame.startingRange
    metadata['polarization'] = str(frame.polarization).replace('/', '')
    if metadata['polarization'].startswith("b'"):
        metadata['polarization'] = metadata['polarization'][2:4]
    metadata['trackNumber'] = frame.trackNumber
    metadata['orbitNumber'] = frame.orbitNumber

    time_seconds = (frame.sensingStart.hour * 3600.0 +
                    frame.sensingStart.minute * 60.0 +
                    frame.sensingStart.second)
    metadata['CENTER_LINE_UTC'] = time_seconds

    orbit = frame.orbit
    peg = orbit.interpolateOrbit(frame.sensingMid, method='hermite')

    Vs = np.linalg.norm(peg.getVelocity())
    metadata['satelliteSpeed'] = Vs
    metadata['azimuthPixelSize'] = Vs/frame.PRF

    refElp = Planet(pname='Earth').ellipsoid
    llh = refElp.xyz_to_llh(peg.getPosition())
    refElp.setSCH(llh[0], llh[1], orbit.getENUHeading(frame.sensingMid))
    metadata['earthRadius'] = refElp.pegRadCur
    metadata['altitude'] = llh[2]

    # for StripMap
    metadata['beam_mode'] = 'SM'
    return metadata, frame

'''

def extract_multilook_number(geom_dir, metadata=dict()):

    for fbase in ['hgt','lat','lon','los']:
        fbase = os.path.join(geom_dir, fbase)
        fnames = glob.glob('{}*.rdr'.format(fbase)) + glob.glob('{}*.geo'.format(fbase))
        if len(fnames) > 0:
            fullXmlFile = '{}.full.xml'.format(fnames[0])
            if os.path.isfile(fullXmlFile):
                fullXmlDict = readfile.read_isce_xml(fullXmlFile)
                xmlDict = readfile.read_attribute(fnames[0])
                metadata['ALOOKS'] = int(int(fullXmlDict['LENGTH']) / int(xmlDict['LENGTH']))
                metadata['RLOOKS'] = int(int(fullXmlDict['WIDTH']) / int(xmlDict['WIDTH']))
                break
    '''
    # default
    for key in ['ALOOKS', 'RLOOKS']:
        if key not in metadata:
            metadata[key] = 1
    return metadata
'''

def extract_isce_metadata(meta_file, geom_dir=None, rsc_file=None, update_mode=True):
    """Extract metadata from ISCE stack products
    Parameters: meta_file : str, path of metadata file, reference/IW1.xml or referenceShelve/data.dat
                geom_dir  : str, path of geometry directory.
                rsc_file  : str, output file name of ROIPAC format rsc file
    Returns:    metadata  : dict
    """

    if not rsc_file:
        rsc_file = os.path.join(os.path.dirname(meta_file), 'data.rsc')

    # check existing rsc_file
    if update_mode and ut.run_or_skip(rsc_file, in_file=meta_file, check_readable=False) == 'skip':
        return readfile.read_roipac_rsc(rsc_file)

    # 1. extract metadata from XML / shelve file
    processor = isce_utils.get_processor(meta_file)

    if processor == 'tops':
        print('extract metadata from ISCE/topsStack xml file:', meta_file)
        metadata, frame = isce_utils.extract_tops_metadata(meta_file)
        metadata['sensor_type'] = 'tops'

    elif processor == 'alosStack':
        print('extract metadata from ISCE/stripmapStack shelve file:', meta_file)
        metadata, frame = isce_utils.extract_stripmap_metadata(meta_file)
        metadata['sensor_type'] = 'stripmap'

    elif processor == 'stripmaps':
        print('extract metadata from ISCE/stripmapStack data file:', meta_file)
        metadata, frame = isce_utils.extract_stripmap_metadata(meta_file)[0]

    else:
        raise ValueError("unrecognized ISCE metadata file: {}".format(meta_file))

    # 2. extract metadata from geometry file
    if geom_dir:
        if processor != 'alosStack':
            metadata = isce_utils.extract_geometry_metadata(geom_dir, metadata)

    # update pixel_size for full resolution data
    metadata['rangePixelSize'] /= metadata['RLOOKS']
    metadata['azimuthPixelSize'] /= metadata['ALOOKS']
    metadata['RLOOKS'] = 1
    metadata['ALOOKS'] = 1
    # NCORRLOOKS for coherence calibration
    rgfact = float(metadata['rangeResolution']) / float(metadata['rangePixelSize'])
    azfact = float(metadata['azimuthResolution']) / float(metadata['azimuthPixelSize'])
    metadata['NCORRLOOKS'] = metadata['RLOOKS'] * metadata['ALOOKS'] / (rgfact * azfact)
    ##

    # 3. common metadata
    metadata['PROCESSOR'] = 'isce'
    if 'ANTENNA_SIDE' not in metadata.keys():
        metadata['ANTENNA_SIDE'] = '-1'

    # convert all value to string format
    for key, value in metadata.items():
        metadata[key] = str(value)

    # write to .rsc file
    metadata = readfile.standardize_metadata(metadata)
    if rsc_file:
        print('writing ', rsc_file)
        writefile.write_roipac_rsc(metadata, rsc_file)
    return metadata


def add_slc_metadata(metadata_in, dates=[], baseline_dict={}):
    """Add metadata unique for each interferogram
    Parameters: metadata_in   : dict, input common metadata for the entire dataset
                dates         : list of str in YYYYMMDD or YYMMDD format
                baseline_dict : dict, output of baseline_timeseries()
    Returns:    metadata      : dict, updated metadata
    """
    # make a copy of input metadata
    metadata = {}
    for k in metadata_in.keys():
        metadata[k] = metadata_in[k]
    metadata['DATE'] = '{}'.format(dates[1])
    if baseline_dict:
        bperp_top = baseline_dict[dates[1]][0] - baseline_dict[dates[0]][0]
        bperp_bottom = baseline_dict[dates[1]][1] - baseline_dict[dates[0]][1]
        metadata['P_BASELINE_TOP_HDR'] = str(bperp_top)
        metadata['P_BASELINE_BOTTOM_HDR'] = str(bperp_bottom)
    return metadata


#########################################################################
def read_tops_baseline(baseline_file):
    bperps = []
    with open(baseline_file, 'r') as f:
        for line in f:
            l = line.split(":")
            if l[0] == "Bperp (average)":
                bperps.append(float(l[1]))
    bperp_top = np.mean(bperps)
    bperp_bottom = np.mean(bperps)
    return [bperp_top, bperp_bottom]


def read_stripmap_baseline(baseline_file):
    fDict = readfile.read_template(baseline_file, delimiter=' ')
    bperp_top = float(fDict['PERP_BASELINE_TOP'])
    bperp_bottom = float(fDict['PERP_BASELINE_BOTTOM'])
    return [bperp_top, bperp_bottom]


def read_baseline_timeseries(baseline_dir, beam_mode='IW'):
    """Read bperp time-series from files in baselines directory
    Parameters: baseline_dir : str, path to the baselines directory
                beam_mode    : str, IW for Sentinel-1/TOPS
                                    SM for StripMap data
    Returns:    bDict : dict, in the following format:
                    {'20141213': [0.0, 0.0],
                     '20141225': [104.6, 110.1],
                     ...
                    }
    """
    print('read perp baseline time-series from {}'.format(baseline_dir))
    # grab all existed baseline files
    bFiles = sorted(glob.glob(os.path.join(baseline_dir, '*/*.txt')))  #for TOPS
    bFiles += sorted(glob.glob(os.path.join(baseline_dir, '*.txt')))   #for stripmap

    # read files into dict
    bDict = {}
    for bFile in bFiles:
        dates = os.path.basename(bFile).split('.txt')[0].split('_')
        if beam_mode == 'IW':
            bDict[dates[1]] = read_tops_baseline(bFile)
        elif beam_mode == 'SM':
            bDict[dates[1]] = read_stripmap_baseline(bFile)
        else:
            raise ValueError('Unrecognized beam_mode/processor: {}'.format(beam_mode))
        bDict[dates[0]] = [0, 0]
    return bDict


#########################################################################
def prepare_geometry(geom_dir, geom_files=[], metadata=dict(), processor='tops', update_mode=True):
    """Prepare and extract metadata from geometry files"""

    print('prepare .rsc file for geometry files')
    # grab all existed files

    # default file basenames
    if not geom_files:
        if processor in ['tops', 'stripmap']:
            geom_files = ['{}.rdr.full.xml'.format(i) for i in GEOMETRY_PREFIXS]

        elif processor in ['alosStack']:
            alooks = metadata['ALOOKS']
            rlooks = metadata['RLOOKS']
            fexts = ['.hgt', '.lat', '.lon', '.los', '.wbd']
            geom_files = ['*_{}rlks_{}alks{}.full'.format(rlooks, alooks, fext) for fext in fexts]

        else:
            raise Exception('unknown processor: {}'.format(processor))

    # get absolute file paths
    geom_files = [os.path.join(geom_dir, i) for i in geom_files]

    if not os.path.exists(geom_files[0]):
        geom_files = [os.path.join(os.path.abspath(geom_dir), x + '.rdr.xml') for x in GEOMETRY_PREFIXS]

    geom_files = [i for i in geom_files if os.path.isfile(i)]

    # write rsc file for each file
    for geom_file in geom_files:
        # prepare metadata for current file
        geom_metadata = read_attribute(geom_file.split('.xml')[0], metafile_ext='.xml')
        geom_metadata.update(metadata)
        # write .rsc file
        rsc_file = geom_file.split('.xml')[0]+'.rsc'
        writefile.write_roipac_rsc(geom_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=True)
    return metadata


def prepare_stack(inputDir, filePattern, processor='tops', metadata=dict(), baseline_dict=dict(), update_mode=True):

    if not os.path.exists(glob.glob(os.path.join(os.path.abspath(inputDir), '*', filePattern + '.xml'))[0]):
        filePattern = filePattern.split('.full')[0]
    print('preparing RSC file for ', filePattern)

    if processor in ['tops', 'stripmap']:
        isce_files = sorted(glob.glob(os.path.join(os.path.abspath(inputDir), '*', filePattern + '.xml')))
    elif processor == 'alosStack':
        isce_files = sorted(glob.glob(os.path.join(os.path.abspath(inputDir), filePattern + '.xml')))    # not sure
    else:
        raise ValueError('Un-recognized ISCE stack processor: {}'.format(processor))

    if len(isce_files) == 0:
        raise FileNotFoundError('no file found in pattern: {}'.format(filePattern))

    # write .rsc file for each interferogram file
    num_file = len(isce_files)
    slc_dates = np.sort(os.listdir(inputDir))
    prog_bar = ptime.progressBar(maxValue=num_file)
    for i in range(num_file):
        # prepare metadata for current file
        isce_file = isce_files[i].split('.xml')[0]
        dates = [slc_dates[0], os.path.basename(os.path.dirname(isce_file))]
        slc_metadata = read_attribute(isce_file, metafile_ext='.xml')
        slc_metadata.update(metadata)
        slc_metadata = add_slc_metadata(slc_metadata, dates, baseline_dict)

        # write .rsc file
        rsc_file = isce_file + '.rsc'
        writefile.write_roipac_rsc(slc_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=False)
        prog_bar.update(i + 1, suffix='{}_{}'.format(dates[0], dates[1]))
    prog_bar.close()


    return


def gen_random_baseline_timeseries(dset_dir, dset_file, max_bperp=10):
    """Generate a baseline time series with random values.
    """
    # list of dates
    fnames = glob.glob(os.path.join(dset_dir, '*', dset_file))
    date12s = sorted([os.path.basename(os.path.dirname(x)) for x in fnames])
    date1s = [x.split('_')[0] for x in date12s]
    date2s = [x.split('_')[1] for x in date12s]
    date_list = sorted(list(set(date1s + date2s)))

    # list of bperp
    bperp_list = [0] + np.random.randint(-max_bperp, max_bperp, len(date_list)-1).tolist()

    # prepare output
    bDict = {}
    for date_str, bperp in zip(date_list, bperp_list):
        bDict[date_str] = [bperp, bperp]

    return bDict

#########################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    inps.processor = isce_utils.get_processor(inps.metaFile)

    # read common metadata
    metadata = {}
    if inps.metaFile:
        rsc_file = os.path.join(os.path.dirname(inps.metaFile), 'data.rsc')
        metadata = extract_isce_metadata(inps.metaFile,
                                         geom_dir=inps.geometryDir,
                                         rsc_file=rsc_file,
                                         update_mode=inps.update_mode)

    # prepare metadata for geometry file
    if inps.geometryDir:
        metadata = prepare_geometry(inps.geometryDir,
                                    metadata=metadata,
                                    processor=inps.processor,
                                    update_mode=inps.update_mode)

    # read baseline info
    baseline_dict = {}
    if inps.baselineDir:
        baseline_dict = isce_utils.read_baseline_timeseries(inps.baselineDir,
                                                            processor=inps.processor)

    ''' 
    # read baseline info
    baseline_dict = {}
    if inps.baselineDir:
        baseline_dict = read_baseline_timeseries(inps.baselineDir,
                                                 beam_mode=metadata['beam_mode'])
    '''

    # prepare metadata for ifgram file
    if inps.slcDir and inps.slcFiles:
        for namePattern in inps.slcFiles:
            prepare_stack(inps.slcDir, namePattern,
                          metadata=metadata,
                          baseline_dict=baseline_dict,
                          processor=inps.processor,
                          update_mode=inps.update_mode)
    print('Done.')
    return


#########################################################################
if __name__ == '__main__':
    """Main driver."""
    main()
