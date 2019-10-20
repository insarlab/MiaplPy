#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee, Zhang Yunjun, Heresh Fattahi                    #
############################################################
# Modified from prep4timeseries.py in ISCE-2.2.0/contrib/stack/topsStack
#


import os
import glob
import shelve
import argparse
import numpy as np
from mintpy.utils import ptime, readfile, writefile, utils as ut
from mintpy.objects import datasetUnitDict
import h5py

EXAMPLE = """example:
  prep_slc_isce.py -s ./merged/SLC -m ./master/IW1.xml -b ./baselines -g ./merged/geom_master  #for topsStack
  """


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
                        help='A list of files that will be used in minopy\n'
                             'e.g.: 20180705.slc.full')
    parser.add_argument('-m', '--meta-file', dest='metaFile', type=str, default=None,
                        help='Metadata file to extract common metada for the stack:\n'
                             'e.g.: for ISCE/topsStack: master/IW3.xml')
    parser.add_argument('-b', '--baseline-dir', dest='baselineDir', type=str, default=None,
                        help=' directory with baselines ')
    parser.add_argument('-g', '--geometry-dir', dest='geometryDir', type=str, default=None,
                        help=' directory with geometry files ')
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


def extract_tops_metadata(xml_file):
    """Read metadata from xml file for Sentinel-1/TOPS
    Parameters: xml_file : str, path of the .xml file, i.e. master/IW1.xml
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
    Parameters: meta_file : str, path of the shelve file, i.e. masterShelve/data.dat
    Returns:    meta      : dict, metadata
    """
    import isce
    import isceobj
    import isceobj.StripmapProc.StripmapProc as St
    from isceobj.Planet.Planet import Planet

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


def extract_multilook_number(geom_dir, metadata=dict()):
    '''
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


def extract_geometry_metadata(geom_dir, metadata=dict()):
    """extract metadata from geometry files"""

    def get_nonzero_row_number(data, buffer=2):
        """Find the first and last row number of rows without zero value
        for multiple swaths data
        """
        if np.all(data):
            r0, r1 = 0 + buffer, -1 - buffer
        else:
            row_flag = np.sum(data != 0., axis=1) == data.shape[1]
            row_idx = np.where(row_flag)[0]
            r0, r1 = row_idx[0] + buffer, row_idx[-1] - buffer
        return r0, r1

    # grab existing files
    geom_files = [os.path.join(os.path.abspath(geom_dir), '{}.rdr.full'.format(i))
                  for i in ['hgt','lat','lon','los']]
    geom_files = [i for i in geom_files if os.path.isfile(i)]
    print('extract metadata from geometry files: {}'.format(
        [os.path.basename(i) for i in geom_files]))

    # get A/RLOOKS
    metadata = extract_multilook_number(geom_dir, metadata)

    # update pixel_size for multilooked data
    metadata['rangePixelSize'] *= metadata['RLOOKS']
    metadata['azimuthPixelSize'] *= metadata['ALOOKS']

    # get LAT/LON_REF1/2/3/4 and HEADING into metadata
    for geom_file in geom_files:
        if 'lat' in os.path.basename(geom_file):
            data = readfile.read(geom_file)[0]
            r0, r1 = get_nonzero_row_number(data)
            metadata['LAT_REF1'] = str(data[r0, 0])
            metadata['LAT_REF2'] = str(data[r0, -1])
            metadata['LAT_REF3'] = str(data[r1, 0])
            metadata['LAT_REF4'] = str(data[r1, -1])

        if 'lon' in os.path.basename(geom_file):
            data = readfile.read(geom_file)[0]
            r0, r1 = get_nonzero_row_number(data)
            metadata['LON_REF1'] = str(data[r0, 0])
            metadata['LON_REF2'] = str(data[r0, -1])
            metadata['LON_REF3'] = str(data[r1, 0])
            metadata['LON_REF4'] = str(data[r1, -1])

        if 'los' in os.path.basename(geom_file):
            data = readfile.read(geom_file, datasetName='az')[0]
            data[data == 0.] = np.nan
            az_angle = np.nanmean(data)
            # convert isce azimuth angle to roipac orbit heading angle
            head_angle = -1 * (270 + az_angle)
            head_angle -= np.round(head_angle / 360.) * 360.
            metadata['HEADING'] = str(head_angle)
    return metadata


def extract_isce_metadata(meta_file, geom_dir=None, rsc_file=None, update_mode=True):
    """Extract metadata from ISCE stack products
    Parameters: meta_file : str, path of metadata file, master/IW1.xml or masterShelve/data.dat
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
    fbase = os.path.basename(meta_file)
    if fbase.startswith("IW"):
        print('extract metadata from ISCE/topsStack xml file:', meta_file)
        metadata = extract_tops_metadata(meta_file)[0]
    elif fbase.startswith("data"):
        print('extract metadata from ISCE/stripmapStack shelve file:', meta_file)
        metadata = extract_stripmap_metadata(meta_file)[0]
    elif fbase.endswith(".xml"):
        metadata = extract_stripmap_metadata(meta_file)[0]
    else:
        raise ValueError("unrecognized ISCE metadata file: {}".format(meta_file))

    # 2. extract metadata from geometry file
    if geom_dir:
        metadata = extract_geometry_metadata(geom_dir, metadata)

    # 3. common metadata
    metadata['PROCESSOR'] = 'isce'
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
def prepare_geometry(geom_dir, metadata=dict(), update_mode=True):
    """Prepare and extract metadata from geometry files"""
    print('prepare .rsc file for geometry files')
    # grab all existed files
    isce_files = [os.path.join(os.path.abspath(geom_dir), '{}.rdr.full'.format(i))
                  for i in ['hgt','lat','lon','los','shadowMask','incLocal']]
    isce_files = [i for i in isce_files if os.path.isfile(i)]

    # write rsc file for each file
    for isce_file in isce_files:
        # prepare metadata for current file
        geom_metadata = readfile.read_attribute(isce_file, metafile_ext='.xml')
        geom_metadata.update(metadata)

        # write .rsc file
        rsc_file = isce_file+'.rsc'
        writefile.write_roipac_rsc(geom_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=True)
    return metadata


def prepare_stack(inputDir, filePattern, metadata=dict(), baseline_dict=dict(), update_mode=True):
    print('prepare .rsc file for ', filePattern)
    isce_files = sorted(glob.glob(os.path.join(os.path.abspath(inputDir), '*', filePattern + '.xml')))
    if len(isce_files) == 0:
        raise FileNotFoundError('no file found in pattern: {}'.format(filePattern))
    slc_dates = np.sort(os.listdir(inputDir))
    # write .rsc file for each interferogram file
    num_file = len(isce_files)
    prog_bar = ptime.progressBar(maxValue=num_file)
    for i in range(num_file):
        isce_file = isce_files[i].split('.xml')[0]
        # prepare metadata for current file
        slc_metadata = read_attribute(isce_file, metafile_ext='.xml')
        slc_metadata.update(metadata)
        dates = [slc_dates[0], os.path.basename(os.path.dirname(isce_file))]
        slc_metadata = add_slc_metadata(slc_metadata, dates, baseline_dict)

        # write .rsc file
        rsc_file = isce_file+'.rsc'
        writefile.write_roipac_rsc(slc_metadata, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=False)
        prog_bar.update(i+1, suffix='{}_{}'.format(dates[0], dates[1]))
    prog_bar.close()
    return


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
    if k == 'slcStack':
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




#########################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    # read common metadata
    metadata = {}
    if inps.metaFile:
        metadata = extract_isce_metadata(inps.metaFile,
                                         geom_dir=inps.geometryDir,
                                         update_mode=inps.update_mode)

    # prepare metadata for geometry file
    if inps.geometryDir:
        metadata = prepare_geometry(inps.geometryDir,
                                    metadata=metadata,
                                    update_mode=inps.update_mode)

    # read baseline info
    baseline_dict = {}
    if inps.baselineDir:
        baseline_dict = read_baseline_timeseries(inps.baselineDir,
                                                 beam_mode=metadata['beam_mode'])

    # prepare metadata for ifgram file
    if inps.slcDir and inps.slcFiles:
        for namePattern in inps.slcFiles:
            prepare_stack(inps.slcDir, namePattern,
                          metadata=metadata,
                          baseline_dict=baseline_dict,
                          update_mode=inps.update_mode)
    print('Done.')
    return


#########################################################################
if __name__ == '__main__':
    """Main driver."""
    main()
