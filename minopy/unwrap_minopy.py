#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee                                    #
# unwrapping based on snaphu                               #
# snaphu version 2.0.3                                     #
############################################################

import os
import sys
import multiprocessing
import isce
from isceobj.Util.ImageUtil import ImageLib as IML
from minopy.objects.arg_parser import MinoPyParser
import numpy as np
from osgeo import gdal
import subprocess
import time
import datetime

CONFIG_FILE = os.path.dirname(os.path.abspath(__file__)) + '/defaults/conf.full'


def main(iargs=None):
    """
        Unwrap interferograms.
    """
    Parser = MinoPyParser(iargs, script='unwrap_minopy')
    inps = Parser.parse()

    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d:%H%M%S')

    if not iargs is None:
        msg = os.path.basename(__file__) + ' ' + ' '.join(iargs[:])
        string = dateStr + " * " + msg
        print(string)
    else:
        msg = os.path.basename(__file__) + ' ' + ' '.join(sys.argv[1::])
        string = dateStr + " * " + msg
        print(string)

    unwObj = Snaphu(inps)
    do_tiles, metadata = unwObj.need_to_split_tiles()


    time0 = time.time()


    try:
        if do_tiles:
            print('1')
            unwObj.unwrap_tile()
        else:
            print('2')
            unwObj.unwrap()
    
    except:
        print('3')
        runUnwrap(inps.input_ifg, inps.unwrapped_ifg, inps.input_cor, metadata)


    print('Time spent: {} m'.format((time.time() - time0)/60))
    return


class Snaphu:

    def __init__(self, inps):

        work_dir = os.path.dirname(inps.input_ifg)
        if os.path.exists(work_dir + '/filt_fine.unw.conncomp.vrt'):
            sys.exit(1)
        self.config_file = os.path.join(work_dir, 'config_all')
        LENGTH = inps.ref_length
        WIDTH = inps.ref_width
        self.num_tiles = inps.num_tiles
        self.out_unwrapped = inps.unwrapped_ifg
        self.inp_wrapped = inps.input_ifg

        self.length, self.width = self.get_image_size()

        azlooks = int(LENGTH / self.length)
        rglooks = int(WIDTH / self.width)

        self.metadata = {'defomax': inps.defo_max,
                         'init_method': inps.init_method,
                         'wavelength': inps.wavelength,
                         'earth_radius': inps.earth_radius,
                         'height': inps.height,
                         'azlooks': azlooks,
                         'rglooks': rglooks}


        with open(CONFIG_FILE, 'r') as f:
            self.config_default = f.readlines()

        self.config_default.append('DEFOMAX_CYCLE   {}\n'.format(inps.defo_max))
        self.config_default.append('CORRFILE   {}\n'.format(inps.input_cor))
        self.config_default.append('CONNCOMPFILE   {}\n'.format(inps.unwrapped_ifg + '.conncomp'))
        self.config_default.append('NLOOKSRANGE {}\n'.format(rglooks))
        self.config_default.append('NLOOKSAZ {}\n'.format(azlooks))
        self.config_default.append('ALTITUDE   {}\n'.format(inps.height))
        self.config_default.append('LAMBDA   {}\n'.format(inps.wavelength))
        self.config_default.append('EARTHRADIUS   {}\n'.format(inps.earth_radius))
        self.config_default.append('INITMETHOD   {}\n'.format(inps.init_method))
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
            x_tile = int(np.sqrt(self.num_tiles)) + 1
            y_tile = x_tile
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

            IML.renderISCEXML(self.out_unwrapped + '.conncomp', bands=1, nyy=self.length, nxx=self.width,
                              datatype='BYTE', scheme='BIL')

        return 

    def unwrap_tile(self):

        cmd = 'snaphu -f {config_file} -d {wrapped_file} {line_length} -o ' \
              '{unwrapped_file} --tile {ytile} {xtile} 200 200 ' \
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

            IML.renderISCEXML(self.out_unwrapped + '.conncomp', bands=1, nyy=self.length, nxx=self.width,
                              datatype='BYTE', scheme='BIL')

        return


def runUnwrap(infile, outfile, corfile, config):
    import isceobj
    from contrib.Snaphu.Snaphu import Snaphu

    costMode = 'DEFO'
    initMethod = config['init_method']
    defomax = config['defomax']

    wrapName = infile
    unwrapName = outfile

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
    #snp.setCorrLooks(corrLooks)
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


if __name__ == '__main__':
    main()
