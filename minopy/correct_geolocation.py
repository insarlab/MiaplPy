#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
import h5py
from mintpy.utils import readfile, writefile, utils as ut

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='Correct for geolocation shift caused by DEM error')
    parser.add_argument('-g', '--geometry', dest='geometry_file', type=str,
                        help='Geometry stack File in radar coordinate, (geometryRadar.h5)')
    parser.add_argument('-d', '--demErr', dest='dem_error_file', type=str, help='DEM error file (demErr.h5)')
    parser.add_argument('--reverse', dest='reverse', action='store_true', help='Reverse geolocation Correction')

    inps = parser.parse_args(args=iargs)
    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    key = 'geolocation_corrected'

    with h5py.File(inps.geometry_file, 'r') as f:
        keys = f.attrs.keys()
        latitude = f['latitude'][:, :]
        longitude = f['longitude'][:, :]

        atr = readfile.read(inps.geometry_file, datasetName='azimuthAngle')[1]

        if not key in keys or atr[key] == 'no':
            status = 'run'
            print('Run geolocation correction ...')
        else:
            status = 'skip'
            print('Geolocation is already done, you may reverse it using --reverse. skip ...')

        if inps.reverse:
            if key in keys and atr[key] == 'yes':
                status = 'run'
                print('Run reversing geolocation correction ...')
            else:
                status = 'skip'
                print('The file is not corrected for geolocation. skip ...')

    if status == 'run':

        az_angle = np.deg2rad(np.float(atr['HEADING']))
        inc_angle = np.deg2rad(readfile.read(inps.geometry_file, datasetName='incidenceAngle')[0])

        dem_error = readfile.read(inps.dem_error_file, datasetName='dem')[0]

        dx = dem_error * (1/np.tan(inc_angle)) * np.cos(az_angle) / 111000  # converted to degree
        dy = dem_error * (1/np.tan(inc_angle)) * np.sin(az_angle) / 111000  # converted to degree


        if inps.reverse:

            latitude += dy
            longitude += dx
            atr[key] = 'no'
            block = [0, latitude.shape[0], 0, latitude.shape[1]]
            writefile.write_hdf5_block(inps.geometry_file,
                                       data=latitude,
                                       datasetName='latitude',
                                       block=block)

            writefile.write_hdf5_block(inps.geometry_file,
                                       data=longitude,
                                       datasetName='longitude',
                                       block=block)

            ut.add_attribute(inps.geometry_file, atr_new=atr)


        else:
            latitude -= dy
            longitude -= dx
            atr[key] = 'yes'
            block = [0, latitude.shape[0], 0, latitude.shape[1]]
            writefile.write_hdf5_block(inps.geometry_file,
                                       data=latitude,
                                       datasetName='latitude',
                                       block=block)
            writefile.write_hdf5_block(inps.geometry_file,
                                       data=longitude,
                                       datasetName='longitude',
                                       block=block)
            ut.add_attribute(inps.geometry_file, atr_new=atr)

    f.close()

    return


if __name__ == '__main__':
    main()
