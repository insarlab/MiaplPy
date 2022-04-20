#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Author:  Sara Mirzaee                                    #
# unwrapping based on snaphu                               #
# snaphu version 2.0.3                                     #
############################################################

import os
import sys
import shutil

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
#import isce
import isceobj
from isceobj.Util.ImageUtil import ImageLib as IML
from contrib.UnwrapComp.unwrapComponents import UnwrapComponents
from miaplpy.objects.arg_parser import MiaplPyParser
import numpy as np
from osgeo import gdal
import subprocess
import glob
import time
import datetime
enablePrint()

def main(iargs=None):
    """
        Unwrap interferograms.
    """
    Parser = MiaplPyParser(iargs, script='unwrap_miaplpy')
    inps = Parser.parse()
    if not 'unwrap_2stage' in inps:
        inps.unwrap_2stage = False

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    time0 = time.time()

    inps.work_dir = os.path.dirname(inps.input_ifg)
    if not os.path.exists(inps.work_dir + '/filt_fine.unw.conncomp.vrt'):

        unwObj = Snaphu(inps)
        do_tiles, metadata = unwObj.need_to_split_tiles()

        try:
            if do_tiles:
                #print('1')
                unwObj.unwrap_tile()
            else:
                #print('2')
                unwObj.unwrap()

        except:
            #print('3')
            runUnwrap(inps.input_ifg, inps.unwrapped_ifg, inps.input_cor, metadata, inps.unwrap_2stage)

    if inps.unwrap_2stage:
        temp_unwrap = os.path.dirname(inps.unwrapped_ifg) + '/temp_filt_fine.unw'
        inpFile = temp_unwrap
        ccFile = glob.glob(os.path.dirname(inps.unwrapped_ifg) + '/*conncomp')[0]
        outFile = inps.unwrapped_ifg
        unwrap_2stage(inpFile, ccFile, outFile, unwrapper_2stage_name=None, solver_2stage=None)

    if inps.remove_filter_flag and not os.path.exists(inps.unwrapped_ifg + '.old'):
        input_ifg_nofilter = os.path.join(os.path.dirname(inps.input_ifg), 'fine.int')
        remove_filter(input_ifg_nofilter, inps.input_ifg, inps.unwrapped_ifg)

    print('Time spent: {} m'.format((time.time() - time0)/60))

    return


class Snaphu:

    def __init__(self, inps):

        self.config_file = os.path.join(inps.work_dir, 'config_all')
        LENGTH = inps.ref_length
        WIDTH = inps.ref_width
        self.num_tiles = inps.num_tiles
        self.out_unwrapped = inps.unwrapped_ifg
        self.inp_wrapped = inps.input_ifg
        self.conncomp = inps.unwrapped_ifg + '.conncomp'
        if inps.unwrap_2stage:
            self.out_unwrapped = os.path.dirname(inps.unwrapped_ifg) + '/temp_filt_fine.unw'

        self.length, self.width = self.get_image_size()

        azlooks = int(np.ceil(LENGTH / self.length))
        rglooks = int(np.ceil(WIDTH / self.width))

        self.metadata = {'defomax': inps.defo_max,
                         'init_method': inps.init_method,
                         'wavelength': inps.wavelength,
                         'earth_radius': inps.earth_radius,
                         'height': inps.height,
                         'azlooks': azlooks,
                         'rglooks': rglooks}

        CONFIG_FILE = os.path.abspath(os.path.dirname(inps.work_dir) + '/../../conf.full')
        if not os.path.exists(CONFIG_FILE):
            CONFIG_FILE_def = os.path.dirname(os.path.abspath(__file__)) + '/defaults/conf.full'
            shutil.copy2(CONFIG_FILE_def, CONFIG_FILE)

        with open(CONFIG_FILE, 'r') as f:
            self.config_default = f.readlines()

        self.config_default.append('DEFOMAX_CYCLE   {}\n'.format(inps.defo_max))
        self.config_default.append('CORRFILE   {}\n'.format(inps.input_cor))
        self.config_default.append('CONNCOMPFILE   {}\n'.format(self.conncomp))
        self.config_default.append('NLOOKSRANGE {}\n'.format(rglooks))
        self.config_default.append('NLOOKSAZ {}\n'.format(azlooks))
        self.config_default.append('ALTITUDE   {}\n'.format(inps.height))
        self.config_default.append('LAMBDA   {}\n'.format(inps.wavelength))
        self.config_default.append('EARTHRADIUS   {}\n'.format(inps.earth_radius))
        self.config_default.append('INITMETHOD   {}\n'.format(inps.init_method))
        if inps.copy_to_tmp:
            os.system('rm -rf /tmp/{}'.format(os.path.basename(inps.work_dir)))
            self.config_default.append('TILEDIR   /tmp/{}\n'.format(os.path.basename(inps.work_dir)))
        if not inps.unwrap_mask is None:
            self.config_default.append('BYTEMASKFILE   {}\n'.format(inps.unwrap_mask))

        return

    def get_image_size(self):
        dg = gdal.Open(self.inp_wrapped, gdal.GA_ReadOnly)
        length = dg.RasterYSize
        width = dg.RasterXSize
        del dg
        return length, width

    def need_to_split_tiles(self):

        do_tiles, self.y_tile, self.x_tile = self.get_nproc_tile()

        if do_tiles:
            for indx, line in enumerate(self.config_default):
                if 'SINGLETILEREOPTIMIZE' in line:
                    self.config_default[indx] = 'SINGLETILEREOPTIMIZE  TRUE\n'

        with open(self.config_file, 'w+') as file:
            file.writelines(self.config_default)

        return do_tiles, self.metadata


    def get_nproc_tile(self):

        if self.num_tiles > 1:
            do_tiles = True
            if np.mod(self.num_tiles, 2) == 0:
                y_tile = int(np.sqrt(self.num_tiles))
            else:
                y_tile = int(np.sqrt(self.num_tiles + 1))
            x_tile = self.num_tiles // y_tile
        else:
            do_tiles = False
            x_tile = 1
            y_tile = 1

        return do_tiles, y_tile, x_tile

    def unwrap(self):

        cmd = 'snaphu -f {config_file} -d {wrapped_file} {line_length} -o ' \
              '{unwrapped_file}'.format(config_file=self.config_file, wrapped_file=self.inp_wrapped,
                                        line_length=self.width, unwrapped_file=self.out_unwrapped)

        print(cmd)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p.communicate()
        print(error)
        if 'ERROR' in error.decode('UTF-8') or 'Error' in error.decode('UTF-8'): # or len(error.decode('UTF-8'))>0:
            raise RuntimeError(error)
        
        if os.path.exists(self.out_unwrapped):

            IML.renderISCEXML(self.out_unwrapped, bands=2, nyy=self.length, nxx=self.width,
                              datatype='float32', scheme='BIL')

            IML.renderISCEXML(self.conncomp, bands=1, nyy=self.length, nxx=self.width,
                              datatype='BYTE', scheme='BIL')

        return 

    def unwrap_tile(self):

        cmd = 'snaphu -f {config_file} -d {wrapped_file} {line_length} -o ' \
              '{unwrapped_file} --tile {ytile} {xtile} 500 500 ' \
              '--nproc {num_proc}'.format(config_file=self.config_file, wrapped_file=self.inp_wrapped,
                                          line_length=self.width, unwrapped_file=self.out_unwrapped, ytile=self.y_tile,
                                          xtile=self.x_tile, num_proc=self.num_tiles)
        print(cmd)
        
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = p.communicate()
        print(error)

        if 'ERROR' in error.decode('UTF-8') or 'Error' in error.decode('UTF-8'):  # or len(error.decode('UTF-8'))>0:
           raise RuntimeError(error)  
        
        if os.path.exists(self.out_unwrapped):
  
            IML.renderISCEXML(self.out_unwrapped, bands=2, nyy=self.length, nxx=self.width,
                              datatype='float32', scheme='BIL')

            IML.renderISCEXML(self.conncomp, bands=1, nyy=self.length, nxx=self.width,
                              datatype='BYTE', scheme='BIL')

        return


def runUnwrap(infile, outfile, corfile, config, unwrap_2stage=False):
    from contrib.Snaphu.Snaphu import Snaphu

    costMode = 'DEFO'
    initMethod = config['init_method']
    defomax = config['defomax']

    wrapName = infile
    unwrapName = outfile
    if unwrap_2stage:
        unwrapName = os.path.dirname(outfile) + '/temp_filt_fine.unw'

    img = isceobj.createImage()
    img.load(infile + '.xml')

    wavelength = float(config['wavelength'])
    width = img.getWidth()
    length = img.getLength()
    earthRadius = float(config['earth_radius'])
    altitude = float(config['height'])
    rangeLooks = int(config['rglooks'])
    azimuthLooks = int(config['azlooks'])

    snp = Snaphu()
    snp.setInitOnly(False)
    snp.setInput(wrapName)
    snp.setOutput(unwrapName)
    snp.setWidth(width)
    snp.setCostMode(costMode)
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corfile)
    snp.setInitMethod(initMethod)
    snp.setMaxComponents(100)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(rangeLooks)
    snp.setAzimuthLooks(azimuthLooks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    ######Render XML
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(unwrapName)
    outImage.setWidth(width)
    outImage.setLength(length)
    outImage.setAccessMode('read')
    # outImage.createImage()
    outImage.renderHdr()
    outImage.renderVRT()
    # outImage.finalizeImage()

    #####Check if connected components was created
    if snp.dumpConnectedComponents:
        connImage = isceobj.Image.createImage()
        connImage.setFilename(unwrapName + '.conncomp')
        # At least one can query for the name used
        connImage.setWidth(width)
        connImage.setLength(length)
        connImage.setAccessMode('read')
        connImage.setDataType('BYTE')
        #    connImage.createImage()
        connImage.renderHdr()
        connImage.renderVRT()
    #   connImage.finalizeImage()

    return


def unwrap_2stage(inpFile, ccFile, outFile, unwrapper_2stage_name=None, solver_2stage=None):
    if unwrapper_2stage_name is None:
        unwrapper_2stage_name = 'REDARC0'

    if solver_2stage is None:
        # If unwrapper_2state_name is MCF then solver is ignored
        # and relaxIV MCF solver is used by default
        solver_2stage = 'pulp'

    print('Unwrap 2 Stage Settings:')
    print('Name: %s' % unwrapper_2stage_name)
    print('Solver: %s' % solver_2stage)

    #inpFile = os.path.join(self._insar.mergedDirname, self._insar.unwrappedIntFilename)
    #ccFile = inpFile + '.conncomp'
    #outFile = os.path.join(self._insar.mergedDirname, self.insar.unwrapped2StageFilename)

    ds_conn = gdal.Open(ccFile + '.vrt', gdal.GA_ReadOnly)
    conn_comp = ds_conn.GetRasterBand(1).ReadAsArray()

    if np.nanmax(conn_comp) > 1:
        # Hand over to 2Stage unwrap
        unw = UnwrapComponents()
        unw.setInpFile(inpFile)
        unw.setConnCompFile(ccFile)
        unw.setOutFile(outFile)
        unw.setSolver(solver_2stage)
        unw.setRedArcs(unwrapper_2stage_name)
        unw.unwrapComponents()
    else:
        print('Single connected component in image. 2 Stage will not have effect')

    if os.path.exists(outFile):
        os.system('rm -rf {} {} {}'.format(inpFile, inpFile+'.vrt', inpFile+'.xml'))
    else:
        os.system('mv {} {}'.format(inpFile, outFile))
        os.system('mv {} {}'.format(inpFile + '.vrt', outFile + '.vrt'))
        os.system('mv {} {}'.format(inpFile + '.xml', outFile + '.xml'))

    return



#   Obsolete   ################################
# unfiltered_ifg = os.path.join(os.path.dirname(inps.input_ifg), 'fine.int')
# remove_filter(unfiltered_ifg, inps.input_ifg, inps.unwrapped_ifg)
# update_connect_component_mask(inps.unwrapped_ifg, inps.input_cor)


def update_connect_component_mask(unwrapped_file, temporal_coherence):

    ds_unw = gdal.Open(unwrapped_file + '.vrt', gdal.GA_ReadOnly)
    phas = ds_unw.GetRasterBand(2).ReadAsArray()

    ds_conn = gdal.Open(unwrapped_file + '.conncomp.vrt', gdal.GA_ReadOnly)
    conn_comp = ds_conn.GetRasterBand(1).ReadAsArray()

    factor_2pi = np.round(phas / (2 * np.pi)).astype(np.int) + conn_comp
    factor_2pi = factor_2pi - np.min(factor_2pi) + 1
    mask = conn_comp > 0

    if not temporal_coherence is None:
        ds_tcoh = gdal.Open(temporal_coherence.split('_msk')[0] + '.vrt', gdal.GA_ReadOnly)
        tcoh = ds_tcoh.GetRasterBand(1).ReadAsArray()
        mask2 = tcoh > 0.5
        mask *= mask2

    new_conn_comp = factor_2pi * mask
    length = new_conn_comp.shape[0]
    width = new_conn_comp.shape[1]
    out_connComp = np.memmap(unwrapped_file + '.conncomp', dtype=np.byte, mode='write', shape=(length, width))
    out_connComp[:, :] = new_conn_comp[:, :]

    del out_connComp

    return


def remove_filter(intfile, filtfile, unwfile):

    ds_unw = gdal.Open(unwfile + ".vrt", gdal.GA_ReadOnly)
    unwphas = ds_unw.GetRasterBand(2).ReadAsArray()
    unwamp = ds_unw.GetRasterBand(1).ReadAsArray()
    width = ds_unw.RasterXSize
    length = ds_unw.RasterYSize
    del ds_unw

    ds_ifg = gdal.Open(intfile + ".vrt", gdal.GA_ReadOnly)
    ifgphas = np.angle(ds_ifg.GetRasterBand(1).ReadAsArray())
    del ds_ifg

    oldunwf = unwfile.split('filt_fine.unw')[0] + 'old_filt_fine.unw'
    unwImage_o = isceobj.Image.createUnwImage()
    unwImage_o.setFilename(oldunwf)
    unwImage_o.setAccessMode('write')
    unwImage_o.setWidth(width)
    unwImage_o.setLength(length)
    unwImage_o.createImage()

    out_unw = unwImage_o.asMemMap(oldunwf)
    # print(out_unw.shape)
    out_unw[:, 0, :] = unwamp
    out_unw[:, 1, :] = unwphas
    # del ifgphas, fifgphas
    unwImage_o.renderHdr()
    unwImage_o.finalizeImage()

    ds_fifg = gdal.Open(filtfile + ".vrt", gdal.GA_ReadOnly)
    fifgphas = np.angle(ds_fifg.GetRasterBand(1).ReadAsArray())
    del ds_fifg

    integer_jumps = unwphas - fifgphas
    del fifgphas

    #shutil.copy2(unwfile, unwfile + '.old')
    #os.system('cp {} {}'.format(unwfile, unwfile+'.old'))

    unwImage = isceobj.Image.createUnwImage()
    unwImage.setFilename(unwfile)
    unwImage.setAccessMode('write')
    unwImage.setWidth(width)
    unwImage.setLength(length)
    unwImage.createImage()

    out_unw = unwImage.asMemMap(unwfile)
    #print(out_unw.shape)
    out_unw[:, 0, :] = unwamp
    out_unw[:, 1, :] = ifgphas + integer_jumps
    #del ifgphas, fifgphas
    unwImage.renderHdr()
    unwImage.finalizeImage()

    return


if __name__ == '__main__':
    main()
