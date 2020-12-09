#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:  Sara Mirzaee                                    #
# unwrapping based on snaphu                               #
# snaphu version 2.0.3                                     #
############################################################

import os
import h5py
import multiprocessing
from isceobj.Util.ImageUtil import ImageLib as IML
from minopy.objects.arg_parser import MinoPyParser
import numpy as np
import gdal


CONFIG_FILE = os.path.dirname(os.path.abspath(__file__)) + '/defaults/conf.full'


def main(iargs=None):
    """
        Unwrap interferograms.
    """
    Parser = MinoPyParser(iargs, script='unwrap_minopy')
    inps = Parser.parse()

    unwObj = Snaphu(inps)
    do_tiles, metadata = unwObj.need_to_split_tiles()

    try:
        if do_tiles:
            unwObj.unwrap_tile()
        else:
            unwObj.unwrap()
    except:
        metadata['defomax'] = inps.defo_max
        metadata['init_method'] = inps.init_method
        runUnwrap(inps.input_ifg, inps.unwrapped_ifg, inps.input_cor, metadata)

    return


class Snaphu:

    def __init__(self, inps):

        work_dir = os.path.dirname(inps.input_ifg)
        self.config_file = os.path.join(work_dir, 'config_all')
        self.reference = inps.reference
        self.out_unwrapped = inps.unwrapped_ifg
        self.inp_wrapped = inps.input_ifg

        retry_times = 10
        for run in range(retry_times):
            try:
                self.metadata = self.get_metadata()
                break
            except:
                continue

        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])

        with open(CONFIG_FILE, 'r') as f:
            self.config_default = f.readlines()

        self.config_default.append('DEFOMAX_CYCLE   {}\n'.format(inps.defo_max))
        self.config_default.append('CORRFILE   {}\n'.format(inps.input_cor))
        self.config_default.append('CONNCOMPFILE   {}\n'.format(inps.unwrapped_ifg + '.conncomp'))
        self.config_default.append('NLOOKSRANGE {}\n'.format(self.metadata['RgLooks']))
        self.config_default.append('NLOOKSAZ {}\n'.format(self.metadata['AzLooks']))
        self.config_default.append('ALTITUDE   {}\n'.format(self.metadata['HEIGHT']))
        self.config_default.append('LAMBDA   {}\n'.format(self.metadata['WAVELENGTH']))
        self.config_default.append('EARTHRADIUS   {}\n'.format(self.metadata['EARTH_RADIUS']))
        self.config_default.append('INITMETHOD   {}\n'.format(inps.init_method))

        return

    def get_image_size(self):
        dg = gdal.Open(self.inp_wrapped, gdal.GA_ReadOnly)
        length = dg.RasterYSize
        width = dg.RasterXSize
        del dg
        return length, width

    def need_to_split_tiles(self):

        do_tiles, self.nproc, self.y_tile, self.x_tile = self.get_nproc_tile()

        if do_tiles:
            for indx, line in enumerate(self.config_default):
                if 'SINGLETILEREOPTIMIZE' in line:
                    self.config_default[indx] = 'SINGLETILEREOPTIMIZE  TRUE\n'

        with open(self.config_file, 'w+') as file:
            file.writelines(self.config_default)

        return do_tiles, self.metadata

    def get_metadata(self):
        with h5py.File(self.reference, 'r') as ds:
            metadata = dict(ds.attrs)
        
        length, width = self.get_image_size()
        LENGTH = int(metadata['LENGTH'])
        WIDTH = int(metadata['WIDTH'])

        azlooks = None
        rglooks = None

        for key in metadata:
            if 'azimuthLooks' in key:
                azlooks = int(metadata[key])
            if 'rangeLooks' in key:
                rglooks = int(metadata[key])
        if azlooks is None:
            azlooks = int(LENGTH / length)
        if rglooks is None:
            rglooks = int(WIDTH/width)

        metadata['RgLooks'] = rglooks
        metadata['AzLooks'] = azlooks
        metadata['LENGTH'] = length
        metadata['WIDTH'] = width
        return metadata

    def get_nproc_tile(self):

        nproc = np.min([64, multiprocessing.cpu_count()])

        npixels = self.length * self.width

        ntiles = npixels / 5000000

        if ntiles > 1:
            do_tiles = True
            x_tile = int(np.sqrt(ntiles)) + 1
            y_tile = x_tile
        else:
            do_tiles = False
            x_tile = 1
            y_tile = 1

        nproc = np.min([nproc, int(ntiles) + 1])

        return do_tiles, nproc, y_tile, x_tile

    def unwrap(self):

        cmd = 'snaphu -f {config_file} -d {wrapped_file} {line_length} -o ' \
              '{unwrapped_file}'.format(config_file=self.config_file, wrapped_file=self.inp_wrapped,
                                        line_length=self.width, unwrapped_file=self.out_unwrapped)

        #print(cmd)
        #os.system(cmd)
        os.system('echo {c}; {c}'.format(c=cmd))

        IML.renderISCEXML(self.out_unwrapped, bands=2, nyy=self.length, nxx=self.width,
                          datatype='float32', scheme='BIL')

        IML.renderISCEXML(self.out_unwrapped + '.conncomp', bands=1, nyy=self.length, nxx=self.width,
                          datatype='BYTE', scheme='BIL')

        return

    def unwrap_tile(self):

        cmd = 'snaphu -f {config_file} -d {wrapped_file} {line_length} -o ' \
              '{unwrapped_file} --tile {ytile} {xtile} 500 500 ' \
              '--nproc {num_proc}'.format(config_file=self.config_file, wrapped_file=self.inp_wrapped,
                                          line_length=self.width, unwrapped_file=self.out_unwrapped, ytile=self.y_tile,
                                          xtile=self.x_tile, num_proc=self.nproc)
        #print(cmd)
        #os.system(cmd)
        os.system('echo {c}; {c}'.format(c=cmd))

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

    wavelength = float(config['WAVELENGTH'])
    width = img.getWidth()
    length = img.getLength()
    earthRadius = float(config['EARTH_RADIUS'])
    altitude = float(config['HEIGHT'])
    rangeLooks = int(config['RgLooks'])
    azimuthLooks = int(config['AzLooks'])

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
