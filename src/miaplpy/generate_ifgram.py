#! /usr/bin/env python3
############################################################
# Copyright(c) 2017, Sara Mirzaee                          #
############################################################

import os
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
import datetime
import numpy as np
from miaplpy.objects.arg_parser import MiaplPyParser
import h5py
from math import sqrt, exp
from osgeo import gdal
import dask.array as da
from pyproj import CRS

enablePrint()

DEFAULT_ENVI_OPTIONS = (
        "INTERLEAVE=BIL",
        "SUFFIX=ADD"
    )


def main(iargs=None):
    """
        Overwrite filtered SLC images in Isce merged/SLC directory.
    """

    Parser = MiaplPyParser(iargs, script='generate_interferograms')
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

    print(inps.out_dir)
    os.makedirs(inps.out_dir, exist_ok=True)

    ifg_file = inps.out_dir + "/filt_fine.int"
    cor_file = inps.out_dir + "/filt_fine.cor"

    run_inreferogram(inps, ifg_file)

    window_size = (6, 12)
    estimate_correlation(ifg_file, cor_file, window_size)

    return


def run_inreferogram(inps, ifg_file):

    if os.path.exists(ifg_file):
        return

    with h5py.File(inps.stack_file, 'r') as ds:
        date_list = np.array([x.decode('UTF-8') for x in ds['date'][:]])
        ref_ind = np.where(date_list == inps.reference)[0]
        sec_ind = np.where(date_list == inps.secondary)[0]
        phase_series = ds['phase']

        length = phase_series.shape[1]
        width = phase_series.shape[2]

        box_size = 3000

        dtype = gdal.GDT_CFloat32
        driver = gdal.GetDriverByName('ENVI')
        out_raster = driver.Create(ifg_file, width, length, 1, dtype, DEFAULT_ENVI_OPTIONS)
        band = out_raster.GetRasterBand(1)

        for i in range(0, length, box_size):
            for j in range(0, width, box_size):
                ref_phase = phase_series[ref_ind, i:i+box_size, j:j+box_size].squeeze()
                sec_phase = phase_series[sec_ind, i:i+box_size, j:j+box_size].squeeze()

                ifg = np.exp(1j * np.angle(np.exp(1j * ref_phase) * np.exp(-1j * sec_phase)))
                band.WriteArray(ifg, j, i)

        band.SetNoDataValue(np.nan)
        out_raster.FlushCache()
        out_raster = None

    write_projection(inps.stack_file, ifg_file)

    return


def write_projection(src_file, dst_file) -> None:
    if src_file.endswith('.h5'):
        with h5py.File(src_file, 'r') as ds:
            attrs = dict(ds.attrs)
            if 'spatial_ref' in attrs.keys():
                projection = attrs['spatial_ref'][3:-1]
                geotransform = [attrs['X_FIRST'], attrs['X_STEP'], 0, attrs['Y_FIRST'], 0, attrs['Y_STEP']]
                geotransform = [float(x) for x in geotransform]
                nodata = np.nan
            else:
                geotransform = [0, 1, 0, 0, 0, -1]
                projection = CRS.from_epsg(4326).to_wkt()
                nodata = np.nan
    else:
        ds_src = gdal.Open(src_file, gdal.GA_Update)
        projection = ds_src.GetProjection()
        geotransform = ds_src.GetGeoTransform()
        nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    ds_dst = gdal.Open(dst_file, gdal.GA_Update)
    ds_dst.SetGeoTransform(geotransform)
    ds_dst.SetProjection(projection)
    ds_dst.GetRasterBand(1).SetNoDataValue(nodata)
    ds_src = ds_dst = None
    return


def estimate_correlation(ifg_file, cor_file, window_size):
    if os.path.exists(cor_file):
        return
    ds = gdal.Open(ifg_file)
    phase = ds.GetRasterBand(1).ReadAsArray()
    length, width = ds.RasterYSize, ds.RasterXSize
    nan_mask = np.isnan(phase)
    zero_mask = np.angle(phase) == 0
    image = np.exp(1j * np.nan_to_num(np.angle(phase)))

    col_size, row_size = window_size
    row_pad = row_size // 2
    col_pad = col_size // 2

    image_padded = np.pad(
        image, ((row_pad, row_pad), (col_pad, col_pad)), mode="constant"
    )

    integral_img = np.cumsum(np.cumsum(image_padded, axis=0), axis=1)

    window_mean = (
            integral_img[row_size:, col_size:]
            - integral_img[:-row_size, col_size:]
            - integral_img[row_size:, :-col_size]
            + integral_img[:-row_size, :-col_size]
    )
    window_mean /= row_size * col_size

    cor = np.clip(np.abs(window_mean), 0, 1)
    cor[nan_mask] = np.nan
    cor[zero_mask] = 0

    dtype = gdal.GDT_Float32
    driver = gdal.GetDriverByName('ENVI')
    out_raster = driver.Create(cor_file, width, length, 1, dtype, DEFAULT_ENVI_OPTIONS)
    band = out_raster.GetRasterBand(1)
    band.WriteArray(cor, 0, 0)
    band.SetNoDataValue(np.nan)
    out_raster.FlushCache()
    out_raster = None

    write_projection(ifg_file, cor_file)

    return


if __name__ == '__main__':
    main()


