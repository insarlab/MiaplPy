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


def measure_d(lat1, lat2, lon1, lon2):
    R = 6378137  # in meter
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = (np.sin(dLat/2)) ** 2 + (np.cos(lat1 * np.pi / 180)) ** 2 * (np.sin(dLon/2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d


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

        az_angle = np.deg2rad(float(atr['HEADING']))
        inc_angle = np.deg2rad(readfile.read(inps.geometry_file, datasetName='incidenceAngle')[0])

        dem_error = readfile.read(inps.dem_error_file, datasetName='dem')[0]

        rad_latitude = np.deg2rad(latitude)

        one_degree_latitude = 111132.92 - 559.82 * np.cos(2*rad_latitude) + \
                              1.175 * np.cos(4 * rad_latitude) - 0.0023 * np.cos(6 * rad_latitude)

        one_degree_longitude = 111412.84 * np.cos(rad_latitude) - \
                               93.5 * np.cos(3 * rad_latitude) + 0.118 * np.cos(5 * rad_latitude)

        print(np.mean(one_degree_latitude), np.mean(one_degree_longitude))

        #one_degree_latitude = measure_d(latitude, latitude+1, 0, 0)
        #one_degree_longitude = measure_d(latitude, latitude, 10, 11)

        #print(np.mean(one_degree_latitude), np.mean(one_degree_longitude))
        
        dx = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.cos(az_angle), one_degree_longitude)  # converted to degree
        dy = np.divide((dem_error) * (1 / np.tan(inc_angle)) * np.sin(az_angle), one_degree_latitude)  # converted to degree

        if inps.reverse:
            sign = np.sign(latitude)
            latitude -= sign * dy

            sign = np.sign(longitude)
            longitude -= sign * dx

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
            sign = np.sign(latitude)
            latitude += sign * dy

            sign = np.sign(longitude)
            longitude += sign * dx

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
