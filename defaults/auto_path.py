#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright(c) 2018-2019, Zhang Yunjun                     #
# Author:  Zhang Yunjun, 2018 Mar                          #
############################################################


import os
import re
import glob
import numpy as np
import mintpy
from mintpy.defaults.auto_path import read_str2dict, get_master_date12

# Auto setting for file structure of Univ. of Miami, as shown below.
# It required 3 conditions: 1) autoPath = True
#                           2) $SCRATCHDIR is defined in environmental variable
#                           3) input custom template with basename same as project_name
# Change it to False if you are not using it.
autoPath = True


# Default path of data files from different InSAR processors to be loaded into MintPy
isceAutoPath = '''##----------Default file path of ISCE-topsStack products
mintpy.load.processor      = isce
mintpy.load.metaFile       = ${PROJECT_DIR}/master/IW*.xml
mintpy.load.baselineDir    = ${PROJECT_DIR}/baselines

mintpy.load.slcFile        = ${PROJECT_DIR}/merged/SLC/*/*.slc.full
mintpy.load.unwFile        = ${PROJECT_DIR}/minopy/interferograms/*/filt*.unw
mintpy.load.corFile        = ${PROJECT_DIR}/minopy/interferograms/*/filt*.cor
mintpy.load.connCompFile   = ${PROJECT_DIR}/minopy/interferograms/*/filt*.unw.conncomp
mintpy.load.ionoFile       = None
mintpy.load.intFile        = None

mintpy.load.demFile        = ${PROJECT_DIR}/merged/geom_master/hgt.rdr.full
mintpy.load.lookupYFile    = ${PROJECT_DIR}/merged/geom_master/lat.rdr.full
mintpy.load.lookupXFile    = ${PROJECT_DIR}/merged/geom_master/lon.rdr.full
mintpy.load.incAngleFile   = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
mintpy.load.azAngleFile    = ${PROJECT_DIR}/merged/geom_master/los.rdr.full
mintpy.load.shadowMaskFile = ${PROJECT_DIR}/merged/geom_master/shadowMask.rdr.full
mintpy.load.bperpFile      = None

'''

roipacAutoPath = '''##----------Default file path of ROI_PAC products
mintpy.load.processor      = roipac
mintpy.load.unwFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*.unw
mintpy.load.corFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*.cor
mintpy.load.connCompFile   = ${PROJECT_DIR}/PROCESS/DONE/IFG*/filt*snap_connect.byt
mintpy.load.intFile        = None

mintpy.load.demFile        = ${PROJECT_DIR}/PROCESS/DONE/*${m_date12}*/radar_*rlks.hgt
mintpy.load.lookupYFile    = ${PROJECT_DIR}/PROCESS/GEO/geo_${m_date12}/geomap_*rlks.trans
mintpy.load.lookupXFile    = ${PROJECT_DIR}/PROCESS/GEO/geo_${m_date12}/geomap_*rlks.trans
mintpy.load.incAngleFile   = None
mintpy.load.azAngleFile    = None
mintpy.load.shadowMaskFile = None
mintpy.load.bperpFile      = None
'''

gammaAutoPath = '''##----------Default file path of GAMMA products
mintpy.load.processor      = gamma
mintpy.load.unwFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/diff*rlks.unw
mintpy.load.corFile        = ${PROJECT_DIR}/PROCESS/DONE/IFG*/*filt*rlks.cor
mintpy.load.connCompFile   = None
mintpy.load.intFile        = None

mintpy.load.demFile        = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.hgt_sim
mintpy.load.lookupYFile    = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.UTM_TO_RDC
mintpy.load.lookupXFile    = ${PROJECT_DIR}/PROCESS/SIM/sim_${m_date12}/sim*.UTM_TO_RDC
mintpy.load.incAngleFile   = None
mintpy.load.azAngleFile    = None
mintpy.load.shadowMaskFile = None
mintpy.load.bperpFile      = ${PROJECT_DIR}/merged/baselines/*/*.base_perp
'''

autoPathDict = {
    'isce'  : isceAutoPath,
    'roipac': roipacAutoPath,
    'gamma' : gammaAutoPath,
}

prefix = 'mintpy.load.'


class PathFind:
    def __init__(self):
        self.scratchdir = os.getenv('SCRATCHDIR')
        self.defaultdir = os.path.expandvars('${MINOPY_HOME}/minopy/defaults')
        self.geomasterdir = 'subset/geom_master'
        self.patchdir = 'patches'
        self.minopydir = 'minopy'
        self.rundir = 'run_files'
        self.configdir = 'configs'
        self.mergeddir = 'subset'
        self.intdir = 'interferograms'
        self.auto_template = self.defaultdir + '/minopy_template.cfg'
        self.wrappercommandtops = 'SentinelWrapper.py -c '
        self.wrappercommandstripmap = 'stripmapWrapper.py -c '
        return

##----------------------------------------------------------------------------------------##


def get_auto_path(processor, project_name, template=dict()):
    """Update template options with auto path defined in autoPathDict
    Parameters: processor : str, isce / roipac / gamma
                project_name : str, Project name, e.g. GalapagosSenDT128
                template : dict,
    Returns:    template : dict,
    """
    # read auto_path_dict
    auto_path_dict = read_str2dict(autoPathDict[processor], print_msg=False)

    # grab variable value: SCRATCHDIR, m_date12
    project_dir = os.path.join(os.getenv('SCRATCHDIR'), project_name)
    m_date12 = None
    if processor in ['roipac', 'gamma']:
        m_date12 = get_master_date12(project_dir, processor=processor)
        if m_date12 and processor == 'roipac':
            # determine nlooks in case both radar_2rlks.hgt and radar_8rlks.hgt exist.
            lookup_file = os.path.join(project_dir, 'PROCESS/GEO/geo_{}/geomap*.trans'.format(m_date12))
            lks = re.findall('_\d+rlks', glob.glob(lookup_file)[0])[0]
            dem_file = os.path.join('${PROJECT_DIR}/PROCESS/DONE/*${m_date12}*', 'radar{}.hgt'.format(lks))
            auto_path_dict[prefix+'demFile'] = dem_file

    var_dict = {}
    var_dict['${PROJECT_DIR}'] = project_dir
    if m_date12:
        var_dict['${m_date12}'] = m_date12

    # update auto_path_dict
    for key, value in auto_path_dict.items():
        if value:
            for var1, var2 in var_dict.items():
                value = value.replace(var1, var2)
            auto_path_dict[key] = value

    # update template option with auto value
    for key, value in auto_path_dict.items():
        if value and template[key] == 'auto':
            template[key] = value
    return template


