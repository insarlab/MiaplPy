#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Copyright (c) 2022, Sara Mirzaee                          #
# Author: Sara Mirzaee                                      #
############################################################
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Optional, List, Tuple, Union
from pathlib import Path
import h5py
from osgeo import gdal, osr
from os import fspath
import datetime
from pyproj import CRS
from pyproj.transformer import Transformer

from mintpy.utils import ptime, attribute as attr
from miaplpy.objects.utils import read_attribute
import time
#import rioxarray
#import isce3
from typing import Optional, Any

from mintpy.objects import (DATA_TYPE_DICT,
                            GEOMETRY_DSET_NAMES,
                            DSET_UNIT_DICT)

BOOL_ZERO = np.bool_(0)
INT_ZERO = np.int16(0)
FLOAT_ZERO = np.float32(0.0)
CPX_ZERO = np.complex64(0.0)

dataType = np.complex64

slcDatasetNames = ['slc']
DSET_UNIT_DICT['slc'] = 'i'
gdal.SetCacheMax(2**30)


HDF5_OPTS = dict(
    # https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    chunks=(1280, 1280),
    compression="gzip",
    compression_opts=4,
    shuffle=True,
    dtype=np.complex64
)


def create_grid_mapping(group, crs: CRS, gt: list):
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    if not 'spatial_ref' in group.keys():
        dset = group.create_dataset('spatial_ref', (), dtype=int)
        dset.attrs.update(crs.to_cf())
        # Also add the GeoTransform
        gt_string = " ".join([str(x) for x in gt])
        dset.attrs.update(
            dict(
                GeoTransform=gt_string,
                units="unitless",
                long_name=(
                    "Dummy variable containing geo-referencing metadata in attributes"
                ),
            )
        )
    else:
        dset = group['spatial_ref']

    return dset


def create_tyx_dsets(
    group,
    gt: list,
    times: list,
    shape: tuple):
    """Create the time, y, and x coordinate datasets."""
    y, x = create_yx_arrays(gt, shape)
    times, calendar, units = create_time_array(times)

    #if not group.dimensions:
    #    group.dimensions = dict(time=times.size, y=y.size, x=x.size)
    # Create the datasets
    if not 'time' in group.keys():
        t_ds = group.create_dataset("time", (len(times),), data=times, dtype=float)
        y_ds = group.create_dataset("y", (len(y),), data=y, dtype=float)
        x_ds = group.create_dataset("x", (len(x),), data=x, dtype=float)

        t_ds.attrs["standard_name"] = "time"
        t_ds.attrs["long_name"] = "time"
        t_ds.attrs["calendar"] = calendar
        t_ds.attrs["units"] = units
    else:
        t_ds = group['time']
        y_ds = group['y']
        x_ds = group['x']

    for name, ds in zip(["y", "x"], [y_ds, x_ds]):
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name.replace('_', ' ')} coordinate of projection"
        ds.attrs["units"] = "m"

    return t_ds, y_ds, x_ds


def create_yx_arrays(
    gt: list, shape: tuple
) -> tuple:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt
    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    return y, x


def create_time_array(dates: datetime.datetime):
    # 'calendar': 'standard',
    # 'units': 'seconds since 2017-02-03 00:00:00.000000'
    # Create the time array
    times = [datetime.datetime.strptime(dd, '%Y%m%d') for dd in dates]
    since_time = times[0]
    time = np.array([(t - since_time).total_seconds() for t in times])
    calendar = "standard"
    units = f"seconds since {since_time.strftime('%Y-%m-%d %H:%M:%S.%f')}"
    return time, calendar, units


def add_complex_ctype(h5file: h5py.File):
    """Add the complex64 type to the root of the HDF5 file.
    This is required for GDAL to recognize the complex data type.
    """
    with h5py.File(h5file, "a") as hf:
        if "complex64" in hf["/"]:
            return
        ctype = h5py.h5t.py_create(np.complex64)
        ctype.commit(hf["/"].id, np.bytes_("complex64"))


def create_geo_dataset_3d(
    *,
    group,
    name: str,
    description: str,
    fillvalue: float,
    attrs: dict,
    timelength: int,
    dtype,):

    dimensions = ["time", "y", "x"]
    if attrs is None:
        attrs = {}
    attrs.update(long_name=description)

    options = HDF5_OPTS
    options["chunks"] = (timelength, *options["chunks"])
    options['dtype'] = dtype

    dset = group.create_dataset(
        name,
        ndim=3,
        fillvalue=fillvalue,
        **options,
    )
    dset.attrs.update(attrs)
    dset.attrs["grid_mapping"] = 'spatial_ref'
    return dset

def get_raster_bounds(xcoord, ycoord, utm_bbox=None):
    """Get common bounds among all data"""
    x_bounds = []
    y_bounds = []

    west = min(xcoord)
    east = max(xcoord)
    north = max(ycoord)
    south = min(ycoord)

    x_bounds.append([west, east])
    y_bounds.append([south, north])
    if not utm_bbox is None:
        x_bounds.append([utm_bbox[0], utm_bbox[2]])
        y_bounds.append([utm_bbox[1], utm_bbox[3]])

    bounds = max(x_bounds)[0], max(y_bounds)[0], min(x_bounds)[1], min(y_bounds)[1]
    return bounds


def bbox_to_utm_sco(bounds, src_epsg, dst_epsg):
    t = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    left, bottom, right, top = bounds
    bbox = (*t.transform(left, bottom), *t.transform(right, top))  # type: ignore
    return bbox


def bbox_to_utm(bbox, epsg_dst, epsg_src=4326):
    """Convert a list of points to a specified UTM coordinate system.
        If epsg_src is 4326 (lat/lon), assumes points_xy are in degrees.
    """
    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    xt, yt = t.transform(xs, ys)
    xys = list(zip(xt, yt))
    return *xys[0], *xys[1]


class cropSLC:
    def __init__(self, pairs_dict: Optional[List[Path]] = None,
                 geo_bbox: Optional[Tuple[float, float, float, float]] = None):
        self.pairsDict = pairs_dict
        self.name = 'slc'
        self.dates = sorted([date for date in self.pairsDict.keys()])
        self.dsNames = list(self.pairsDict[self.dates[0]].datasetDict.keys())
        self.dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        self.numSlc = len(self.pairsDict)
        self.bperp = np.zeros(self.numSlc)
        dsname0 = self.pairsDict[self.dates[0]].datasetDict['slc']
        self.geo_bbox = geo_bbox
        self.bb_utm = None
        self.rdr_bbox = None
        self.crs, self.geotransform, self.shape = self.get_transform(dsname0)
        self.length, self.width = self.shape

        self.lengthc, self.widthc = self.get_size()

    def get_transform(self, src_file):
        import pdb; pdb.set_trace()
        with h5py.File(src_file, 'r') as ds:
            dsg = ds['data']['projection'].attrs
            xcoord = ds['data']['x_coordinates'][()]
            ycoord = ds['data']['y_coordinates'][()]
            shape = (ds['data']['y_coordinates'].shape[0], ds['data']['x_coordinates'].shape[0])
            #crs = CRS.from_wkt(dsg['spatial_ref'].decode("utf-8"))
            crs = dsg['spatial_ref'].decode("utf-8")
            x_step = float(ds['data']['x_spacing'][()])
            y_step = float(ds['data']['y_spacing'][()])
            x_first = min(ds['data']['x_coordinates'][()])
            y_first = max(ds['data']['y_coordinates'][()])
            geotransform = (x_first, x_step, 0, y_first, 0, y_step)
        if self.geo_bbox is None:
            x_last = max(ds['data']['x_coordinates'][()])
            y_last = min(ds['data']['y_coordinates'][()])
            self.bb_utm = (x_first, y_first, x_last, y_last)
        else:
            self.bb_utm = bbox_to_utm(self.geo_bbox, epsg_src=4326, epsg_dst=crs.to_epsg())

        bounds = get_raster_bounds(xcoord, ycoord, self.bb_utm)

        xindex = np.where(np.logical_and(xcoord >= bounds[0], xcoord <= bounds[2]))[0]
        yindex = np.where(np.logical_and(ycoord >= bounds[1], ycoord <= bounds[3]))[0]
        row1, row2 = min(yindex), max(yindex)
        col1, col2 = min(xindex), max(xindex)

        self.rdr_bbox = (col1, row1, col2, row2)

        return crs, geotransform, shape

    def get_subset_transform(self):
        # Define the cropping extent
        width = self.rdr_bbox[2] - self.rdr_bbox[0]
        length = self.rdr_bbox[3] - self.rdr_bbox[1]
        crop_extent = (self.bb_utm[0], self.bb_utm[1], self.bb_utm[2], self.bb_utm[3])
        crop_transform = rasterio.transform.from_bounds(self.bb_utm[0],
                                                        self.bb_utm[1],
                                                        self.bb_utm[2],
                                                        self.bb_utm[3], width, length)
        return crop_transform, crop_extent

    def obs_get_rdr_bbox(self):
        # calculate the image coordinates
        col1 = int((self.bb_utm[0] - self.geotransform[0]) / self.geotransform[1])
        col2 = int((self.bb_utm[2] - self.geotransform[0]) / self.geotransform[1])
        row1 = int((self.bb_utm[3] - self.geotransform[3]) / self.geotransform[5])
        row2 = int((self.bb_utm[1] - self.geotransform[3]) / self.geotransform[5])

        if col2 > self.width:
            col2 = self.width
            bb_utm2 = col2 * self.geotransform[1] + self.geotransform[0]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], bb_utm2, self.bb_utm[3])

        if row2 > self.length:
            row2 = self.length
            bb_utm1 = row2 * self.geotransform[5] + self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], bb_utm1, self.bb_utm[2], self.bb_utm[3])

        if col1 < 0:
            col1 = 0
            bb_utm0 = self.geotransform[0]
            self.bb_utm = (bb_utm0, self.bb_utm[1], self.bb_utm[2], self.bb_utm[3])
        if row1 < 0:
            row1 = 0
            bb_utm3 = self.geotransform[3]
            self.bb_utm = (self.bb_utm[0], self.bb_utm[1], self.bb_utm[2], bb_utm3)

        print('crop_geo: ', [col1, row1, col2, row2])
        return col1, row1, col2, row2

    def get_size(self):
        length = self.rdr_bbox[3] - self.rdr_bbox[1]
        width = self.rdr_bbox[2] - self.rdr_bbox[0]
        return length, width


    def get_date_list(self):
        self.dateList = sorted([date for date in self.pairsDict.keys()])
        return self.dateList


    def read_subset(self, slc_file):
        with h5py.File(slc_file, 'r') as f:
            subset_slc = f['data']['VV'][self.rdr_bbox[1]:self.rdr_bbox[3],
                  self.rdr_bbox[0]:self.rdr_bbox[2]]

        return subset_slc

    def get_metadata(self):
        slcObj = [v for v in self.pairsDict.values()][0]
        self.metadata = slcObj.get_metadata()
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')
        return self.metadata

    def write2hdf5(self, outputFile='slcStack.h5', access_mode='a', compression=None, extra_metadata=None):

        dsNames = [i for i in slcDatasetNames if i in self.dsNames]
        maxDigit = max([len(i) for i in dsNames])
        self.outputFile = outputFile
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))
        dsName = 'slc'

        f = h5py.File(self.outputFile, access_mode)
        print('create HDF5 file {} with {} mode'.format(self.outputFile, access_mode))

        #create_grid_mapping(group=f, crs=self.crs, gt=list(self.geotransform))
        #create_tyx_dsets(group=f, gt=list(self.geotransform), times=self.dates, shape=(self.lengthc, self.widthc))

        dsShape = (self.numSlc, self.lengthc, self.widthc)
        dsDataType = dataType
        dsCompression = compression

        self.bperp = np.zeros(self.numSlc)

        print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
               ' with compression = {c}').format(d=dsName,
                                                 w=maxDigit,
                                                 t=str(dsDataType),
                                                 s=dsShape,
                                                 c=dsCompression))

        if dsName in f.keys():
            ds = f[dsName]
        else:
            ds = f.create_dataset(dsName,
                                  shape=dsShape,
                                  maxshape=(None, dsShape[1], dsShape[2]),
                                  dtype=dsDataType,
                                  chunks=True,
                                  compression=dsCompression)

            ds.attrs.update(long_name="SLC complex data")
            ds.attrs["grid_mapping"] = 'spatial_ref'

        prog_bar = ptime.progressBar(maxValue=self.numSlc)

        for i in range(self.numSlc):
            box = self.rdr_bbox
            slcObj = self.pairsDict[self.dates[i]]
            dsSlc, metadata = slcObj.read(dsName, box=box)
            ds[i, :, :] = dsSlc[:, :]

            self.bperp[i] = slcObj.get_perp_baseline()
            prog_bar.update(i + 1, suffix='{}'.format(self.dates[i][0]))

        prog_bar.close()
        ds.attrs['MODIFICATION_TIME'] = str(time.time())

        ###############################
        # 1D dataset containing dates of all images
        dsName = 'date'
        dsDataType = np.bytes_
        dsShape = (self.numSlc, 1)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))

        data = np.array(self.dates, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)

        ###############################
        # 1D dataset containing perpendicular baseline of all pairs
        dsName = 'bperp'
        dsDataType = np.float32
        dsShape = (self.numSlc,)
        print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                          w=maxDigit,
                                                                          t=str(dsDataType),
                                                                          s=dsShape))
        data = np.array(self.bperp, dtype=dsDataType)
        if not dsName in f.keys():
            f.create_dataset(dsName, data=data)

        ###############################
        # Attributes
        self.get_metadata()
        if extra_metadata:
            self.metadata.update(extra_metadata)
            # print('add extra metadata: {}'.format(extra_metadata))
        self.metadata = attr.update_attribute4subset(self.metadata, self.rdr_bbox)

        self.metadata['FILE_TYPE'] = 'timeseries'  # 'slc'
        for key, value in self.metadata.items():
            f.attrs[key] = value

        f.close()

        print('Finished writing to {}'.format(self.outputFile))
        return self.outputFile
