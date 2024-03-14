#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
import os
# from libc.stdio cimport printf
from miaplpy.objects.slcStack import slcStack
import h5py
import time
from datetime import datetime
from osgeo import gdal, gdal_array
from pyproj import CRS
from miaplpy.objects.crop_geo import create_grid_mapping, create_tyx_dsets
from . cimport utils
from.utils import process_patch_c

DEFAULT_TILE_SIZE = [128, 128]
DEFAULT_TIFF_OPTIONS = (
    "COMPRESS=DEFLATE",
    "ZLEVEL=4",
    "TILED=YES",
    f"BLOCKXSIZE={DEFAULT_TILE_SIZE[1]}",
    f"BLOCKYSIZE={DEFAULT_TILE_SIZE[0]}",
)
DEFAULT_ENVI_OPTIONS = (
    "INTERLEAVE=BIL",
    "SUFFIX=ADD"
)

cdef void write_hdf5_block_3D(object fhandle, float[:, :, ::1] data, bytes datasetName, list block):

    fhandle[datasetName.decode('UTF-8')][block[0]:block[1], block[2]:block[3], block[4]:block[5]] = data
    return

cdef void write_hdf5_block_2D_int(object fhandle, int[:, ::1] data, bytes datasetName, list block):

    fhandle[datasetName.decode('UTF-8')][block[0]:block[1], block[2]:block[3]] = data
    return

cdef void write_hdf5_block_2D_float(object fhandle, float[:, ::1] data, bytes datasetName, list block):

    fhandle[datasetName.decode('UTF-8')][block[0]:block[1], block[2]:block[3]] = data
    return


cdef class CPhaseLink:

    def __init__(self, object inps):
        cdef float alpha
        self.inps = inps
        self.work_dir = inps.work_dir.encode('UTF-8')
        self.mask_file = inps.mask_file.encode('UTF-8')
        self.phase_linking_method = inps.inversion_method.encode('UTF-8')
        self.shp_test = inps.shp_test.encode('UTF-8')
        self.slc_stack = inps.slc_stack.encode('UTF-8')
        self.range_window = np.int32(inps.range_window)
        self.azimuth_window = np.int32(inps.azimuth_window)
        self.patch_size = np.int32(inps.patch_size)
        self.ps_shp = np.int32(inps.ps_shp)
        self.out_dir = self.work_dir + b'/inverted'
        #self.num_archived = np.int32(inps.num_archived)
        os.makedirs(self.out_dir.decode('UTF-8'), exist_ok='True')
        self.slcStackObj = slcStack(inps.slc_stack)
        self.metadata = self.slcStackObj.get_metadata()
        #self.all_date_list = self.slcStackObj.get_date_list()
        self.all_date_list = self.get_dates(inps.slc_stack)
        with h5py.File(inps.slc_stack, 'r') as f:
            self.prep_baselines = f['bperp'][:]
        self.n_image, self.length, self.width = self.slcStackObj.get_size()
        self.time_lag = inps.time_lag


        # total number of neighbouring pixels
        self.shp_size = self.range_window * self.azimuth_window

        # threshold for shp test based on number of images to test
        alpha = 0.01
        if self.shp_test == b'ks':
            self.distance_thresh = utils.ks_lut_cy(self.n_image, self.n_image, alpha)
        else:
            self.distance_thresh = alpha

        # split the area in to patches of size 'self.patch_size'
        self.box_list, self.num_box = self.patch_slice()

        # default number of images in each ministack
        self.mini_stack_default_size = inps.ministack_size

        if b'sequential' == self.phase_linking_method[0:10]:
            self.total_num_mini_stacks = self.n_image // self.mini_stack_default_size
        else:
            self.total_num_mini_stacks = 1

        self.window_for_shp()

        #self.RSLCfile = self.out_dir + b'/phase_series.h5'
        self.RSLCfile = self.out_dir + b'/phase_series.h5'

        if b'sequential' == self.phase_linking_method[0:10]:
            self.sequential = True
        else:
            self.sequential = False
        return

    def get_projection(self, slc_stack):
        cdef object ds, gt
        cdef str projection
        cdef dict attrs
        cdef list geotransform #, extent
        with h5py.File(slc_stack, 'r') as ds:
            attrs = dict(ds.attrs)
            if 'spatial_ref' in attrs.keys():
                projection = attrs['spatial_ref'][3:-1]
                geotransform = [attrs['X_FIRST'], attrs['X_STEP'], 0, attrs['Y_FIRST'], 0, attrs['Y_STEP']]
                geotransform = [float(x) for x in geotransform]
            else:
                geotransform = [0, 1, 0, 0, 0, -1]
                projection = CRS.from_epsg(4326).to_wkt()

        return projection, geotransform

    def get_dates(self, slc_stack):
        cdef object ds, ff, ft
        cdef list dates
        cdef double[::1] tt
        cdef double st
        with h5py.File(slc_stack) as ds:
            if 'date' in ds.keys():
                dates = list(ds['date'][()])
                return dates
            else:
                tt = ds['time'][()]
                ff = datetime.strptime(ds['time'].attrs['units'].split('seconds since ')[1], '%Y-%m-%d %H:%M:%S.%f')
                ft = datetime.strptime('19691231-16', '%Y%m%d-%H')
                st = (ff - ft).total_seconds()
                dates = [datetime.fromtimestamp(t+st) for t in tt]
                return [t.strftime('%Y%m%d') for t in dates]


    def patch_slice(self):
        """
        Slice the image into patches of size patch_size
        box = (x0 y0 x1 y1) = (col0, row0, col1, row1) for each patch with respect to the whole image
        Returns box list, number of boxes
        -------

        """
        cdef int[::1] patch_row_1 = np.arange(0, self.length - self.azimuth_window, self.patch_size, dtype=np.int32)
        cdef int[::1] patch_row_2 = np.arange(0, self.length - self.azimuth_window, self.patch_size, dtype=np.int32) + self.patch_size
        cdef int[::1] patch_col_1 = np.arange(0, self.width - self.range_window, self.patch_size, dtype=np.int32)
        cdef int[::1] patch_col_2 = np.arange(0, self.width - self.range_window, self.patch_size, dtype=np.int32) + self.patch_size
        cdef int i, t, index, nr = patch_row_1.shape[0]
        cdef int num_box, nc = patch_col_1.shape[0]
        cdef list box_list = []
        cdef cnp.ndarray[int, ndim=1] box #= np.arange(4, dtype=np.int32)
        patch_row_2[nr-1] = self.length
        patch_col_2[nc-1] = self.width

        num_box = nr * nc
        index = 0
        for i in range(nr):
            for t in range(nc):
                box = np.arange(5, dtype=np.int32)
                box[0] = patch_col_1[t]
                box[1] = patch_row_1[i]
                box[2] = patch_col_2[t]
                box[3] = patch_row_2[i]
                box[4] = index
                box_list.append(box)
                index += 1

        return box_list, num_box

    def window_for_shp(self):
        """
        Shp window to be placed on each pixel
        Returns rows, cols, reference pixel row index, reference pixel col index
        -------

        """
        self.sample_rows = np.arange(-((self.azimuth_window - 1) // 2), ((self.azimuth_window - 1) // 2) + 1, dtype=np.int32)
        self.reference_row = np.array([(self.azimuth_window - 1) // 2], dtype=np.int32)

        self.sample_cols = np.arange(-((self.range_window - 1) // 2), ((self.range_window - 1) // 2) + 1, dtype=np.int32)
        self.reference_col = np.array([(self.range_window - 1) // 2], dtype=np.int32)

        return

    def initiate_output(self):
        cdef object RSLC, psf

        with h5py.File(self.RSLCfile.decode('UTF-8'), 'a') as RSLC:

            if 'phase' in RSLC.keys():
                RSLC['phase'].resize(self.n_image, 0)
            else:
                self.metadata['FILE_TYPE'] = 'timeseries' #'phase'
                self.metadata['DATA_TYPE'] = 'float32'
                self.metadata['data_type'] = 'FLOAT'
                self.metadata['description'] = 'Inverted wrapped phase time series'
                self.metadata['file_name'] = self.RSLCfile.decode('UTF-8')
                self.metadata['family'] = 'wrappedphase'

                for key, value in self.metadata.items():
                    RSLC.attrs[key] = value

                RSLC.create_dataset('phase',
                                    shape=(self.n_image, self.length, self.width),
                                    maxshape=(None, self.length, self.width),
                                    chunks=True,
                                    dtype=np.float32)

                RSLC.create_dataset('amplitude',
                                    shape=(self.n_image, self.length, self.width),
                                    maxshape=(None, self.length, self.width),
                                    chunks=True,
                                    dtype=np.float32)
                '''
                if b'real_time' == self.phase_linking_method[0:9]:
                    RSLC.create_dataset('phase_seq',
                                        shape=(self.n_image, self.length, self.width),
                                        maxshape=(None, self.length, self.width),
                                        chunks=True,
                                        dtype=np.float32)

                    RSLC.create_dataset('amplitude_seq',
                                        shape=(self.n_image, self.length, self.width),
                                        maxshape=(None, self.length, self.width),
                                        chunks=True,
                                        dtype=np.float32)
                '''
                RSLC.create_dataset('shp',
                                    shape=(self.length, self.width),
                                    maxshape=(self.length, self.width),
                                    chunks=True,
                                    dtype=np.int32)

                RSLC['shp'][:, :] = 1

                RSLC.create_dataset('temporalCoherence',
                                    shape=(2, self.length, self.width),
                                    maxshape=(2, self.length, self.width),
                                    chunks=True,
                                    dtype=np.float32)

                RSLC['temporalCoherence'][:, :, :] = -1

                # 1D dataset containing dates of all images
                data = np.array(self.all_date_list, dtype=np.string_)
                RSLC.create_dataset('date', data=data)

                # 1D dataset containing perpendicular baselines of all images
                data = np.array(self.prep_baselines, dtype=np.float32)
                RSLC.create_dataset('bperp', data=data)

        mask_ps_file = self.work_dir + b'/maskPS.h5'

        with h5py.File(mask_ps_file.decode('UTF-8'), 'a') as psf:
            if not 'mask' in psf.keys():
                self.metadata['FILE_TYPE'] = 'mask' #'phase'
                self.metadata['DATA_TYPE'] = 'bool'
                self.metadata['data_type'] = 'BYTE'
                self.metadata['description'] = 'PS mask'
                self.metadata['file_name'] = mask_ps_file.decode('UTF-8')
                self.metadata['family'] = 'PS mask'

                for key, value in self.metadata.items():
                    psf.attrs[key] = value

                psf.create_dataset('mask',
                                    shape=(self.length, self.width),
                                    maxshape=(self.length, self.width),
                                    chunks=True,
                                    dtype=np.bool_)
                psf['mask'][:, :] = 0

        return

    def get_datakwargs(self):

        cdef dict data_kwargs = dict(range_window=self.range_window, azimuth_window=self.azimuth_window,
                                     width=self.width, length=self.length, n_image=self.n_image,
                                     slcStackObj=self.slcStackObj, distance_threshold=self.distance_thresh,
                                     def_sample_rows=np.array(self.sample_rows),
                                     def_sample_cols=np.array(self.sample_cols), reference_row=self.reference_row,
                                     reference_col=self.reference_col, phase_linking_method=self.phase_linking_method,
                                     total_num_mini_stacks=self.total_num_mini_stacks,
                                     default_mini_stack_size=self.mini_stack_default_size, ps_shp=self.ps_shp,
                                     shp_test=self.shp_test, out_dir=self.out_dir, time_lag=self.time_lag,
                                     mask_file=self.mask_file)
        #"num_archived": self.num_archived,
        return data_kwargs


    def loop_patches(self, list box_list):

        cdef float m, s, start_time = time.time()
        cdef cnp.ndarray[int, ndim=1] box
        cdef dict data_kwargs = self.get_datakwargs()

        for box in box_list:

            data_kwargs['box'] = box
            os.makedirs(self.out_dir.decode('UTF-8') + '/PATCHES', exist_ok=True)
            process_patch_c(**data_kwargs)

        m, s = divmod(time.time() - start_time, 60)
        print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
        return

    def set_projection_hdf(self, object fhandle, object projection, list geotransform):
        # cdef object grp

        create_grid_mapping(group=fhandle, crs=projection, gt=list(geotransform))

        #if "georeference" in fhandle:
        #    grp = fhandle["georeference"]
        #else:
        #    grp = fhandle.create_group("georeference")

        #if not "transform" in grp.keys():
        #    grp.create_dataset("transform", data=geotransform, dtype=np.float32)
        #grp.attrs["crs"] = projection
        #grp.attrs["extent"] = crop_extent
        #grp.attrs["pixel_size_x"] = self.pixel_width
        #grp.attrs["pixel_size_y"] = self.pixel_height
        return

    def set_projection_gdal1_int(self, cnp.ndarray[int, ndim=2] data, int bands, bytes output, str description,
                             str projection, list geotransform):
        cdef object driver, dataset, band1, target_crs
        driver = gdal.GetDriverByName('ENVI')
        dataset = driver.Create(output, self.width, self.length, 1, gdal.GDT_Int16, DEFAULT_ENVI_OPTIONS)
        dataset.SetGeoTransform(list(geotransform))
        dataset.SetProjection(projection) #.to_wkt()) #target_crs.ExportToWkt())
        band1 = dataset.GetRasterBand(1)
        band1.SetDescription(description)
        gdal_array.BandWriteArray(band1, data, xoff=0, yoff=0)
        # band1.WriteArray(data, xoff=0, yoff=0)
        band1.SetNoDataValue(np.nan)

        dataset.FlushCache()
        dataset = None

        return

    def set_projection_gdal1(self, cnp.ndarray[float, ndim=2] data, int bands, bytes output, str description,
                             str projection, list geotransform):
        cdef object driver, dataset, band1, target_crs
        driver = gdal.GetDriverByName('ENVI')
        dataset = driver.Create(output, self.width, self.length, 1, gdal.GDT_Float32, DEFAULT_ENVI_OPTIONS)
        dataset.SetGeoTransform(list(geotransform))
        dataset.SetProjection(projection) #.to_wkt()) #target_crs.ExportToWkt())
        band1 = dataset.GetRasterBand(1)
        band1.SetDescription(description)
        gdal_array.BandWriteArray(band1, data, xoff=0, yoff=0)
        # band1.WriteArray(data, xoff=0, yoff=0)
        band1.SetNoDataValue(np.nan)

        dataset.FlushCache()
        dataset = None

        return

    def set_projection_gdalm(self, cnp.ndarray[float, ndim=3] data, int bands, bytes output, list description,
                             str projection, list geotransform):
        cdef int i
        cdef object driver, dataset, band
        driver = gdal.GetDriverByName('ENVI')
        dataset = driver.Create(output, self.width, self.length, bands, gdal.GDT_Float32, DEFAULT_ENVI_OPTIONS)

        for i in range(bands):
            band = dataset.GetRasterBand(i+1)
            band.SetDescription(description[i])
            gdal_array.BandWriteArray(band, data[i, :, :], xoff=0, yoff=0)
            #band.WriteArray(data[i, :, :], xoff=0, yoff=0)
            band.SetNoDataValue(np.nan)
            del band

        dataset.SetGeoTransform(list(geotransform))
        dataset.SetProjection(projection) #.to_wkt())
        dataset.FlushCache()
        dataset = None

        return

    def unpatch(self):
        cdef list block
        cdef object fhandle, psf
        cdef int index, box_length, box_width
        cdef cnp.ndarray[int, ndim=1] box
        cdef bytes patch_dir
        cdef str projection
        cdef float complex[:, :, ::1] rslc_ref, rslc_ref_seq
        cdef cnp.ndarray[float, ndim=3] temp_coh, ps_prod, eig_values = np.zeros((3, self.length, self.width), dtype=np.float32)
        cdef cnp.ndarray[float, ndim=2] amp_disp = np.zeros((self.length, self.width), dtype=np.float32)
        #cdef cnp.ndarray[int, ndim=2] reference_index_map = np.zeros((self.length, self.width), dtype=np.int32)
        cdef list geotransform

        if os.path.exists(self.RSLCfile.decode('UTF-8')):
            print('Deleting old phase_series.h5 ...')
            os.remove(self.RSLCfile.decode('UTF-8'))

        mask_ps_file = self.work_dir + b'/maskPS.h5'
        if os.path.exists(mask_ps_file.decode('UTF-8')):
            os.remove(mask_ps_file.decode('UTF-8'))

        self.initiate_output()
        print('Concatenate and write wrapped phase time series to HDF5 file phase_series.h5 ')
        print('open  HDF5 file phase_series.h5 in a mode')

        dask_chunks = (1, 128 * 10, 128 * 10)
        projection, geotransform = self.get_projection(self.inps.slc_stack)

        with h5py.File(self.RSLCfile.decode('UTF-8'), 'a') as fhandle:
            #create_grid_mapping(group=fhandle, crs=projection, gt=list(geotransform))
            #create_tyx_dsets(group=fhandle, gt=list(geotransform), times=self.all_date_list, shape=(self.length, self.width))

            for index, box in enumerate(self.box_list):
                box_width = box[2] - box[0]
                box_length = box[3] - box[1]

                patch_dir = self.out_dir + ('/PATCHES/PATCH_{:04.0f}'.format(index)).encode('UTF-8')
                rslc_ref = np.load(patch_dir.decode('UTF-8') + '/phase_ref.npy', allow_pickle=True)
                temp_coh = np.load(patch_dir.decode('UTF-8') + '/tempCoh.npy', allow_pickle=True)
                shp = np.load(patch_dir.decode('UTF-8') + '/shp.npy', allow_pickle=True)
                mask_ps = np.load(patch_dir.decode('UTF-8') + '/mask_ps.npy', allow_pickle=True)
                ps_prod = np.load(patch_dir.decode('UTF-8') + '/ps_products.npy', allow_pickle=True)
                #reference_index = np.load(patch_dir.decode('UTF-8') + '/reference_index.npy', allow_pickle=True)
                #if b'real_time' == self.phase_linking_method[0:9]:
                #    rslc_ref_seq = np.load(patch_dir.decode('UTF-8') + '/phase_ref_seq.npy', allow_pickle=True)

                temp_coh[temp_coh<0] = 0

                print('-' * 50)
                print("Concatenate block {}/{} : {}".format(index, self.num_box, box[0:4]))

                # wrapped interferograms 3D
                block = [0, self.n_image, box[1], box[3], box[0], box[2]]
                ## write_hdf5_block_3D(fhandle, rslc_ref, b'slc', block)
                write_hdf5_block_3D(fhandle, np.angle(rslc_ref), b'phase', block)
                write_hdf5_block_3D(fhandle, np.abs(rslc_ref), b'amplitude', block)
                #if b'real_time' == self.phase_linking_method[0:9]:
                #    write_hdf5_block_3D(fhandle, np.angle(rslc_ref_seq), b'phase_seq', block)
                #    write_hdf5_block_3D(fhandle, np.abs(rslc_ref_seq), b'amplitude_seq', block)

                # SHP - 2D
                block = [box[1], box[3], box[0], box[2]]
                write_hdf5_block_2D_int(fhandle, shp, b'shp', block)
                amp_disp[block[0]:block[1], block[2]:block[3]] = ps_prod[0, :, :]
                #reference_index_map[block[0]:block[1], block[2]:block[3]] = reference_index[:, :]

                # temporal coherence - 3D
                block = [0, 2, box[1], box[3], box[0], box[2]]
                write_hdf5_block_3D(fhandle, temp_coh, b'temporalCoherence', block)
                eig_values[0:3, block[2]:block[3], block[4]:block[5]] = ps_prod[1:4, :, :]


            ###
            print('close HDF5 file phase_series.h5.')

            print('write amplitude dispersion and top eigen values')
            amp_disp_file = self.out_dir + b'/amp_dipersion_index'
            self.set_projection_gdal1(amp_disp, 1,
                                      amp_disp_file, 'Amplitude dispersion index', projection, geotransform)
            #self.set_projection_gdal1(amp_disp, 1, amp_disp_file, 'Amplitude dispersion index', projection, geotransform)
            top_eig_files = self.out_dir + b'/top_eigenvalues'
            self.set_projection_gdalm(np.array(eig_values), 3, top_eig_files,
                                      ['Top eigenvalue 1', 'Top eigenvalue 2', 'Top eigenvalue 3'], projection, geotransform)
            print('write averaged temporal coherence file from mini stacks')
            temp_coh_file = self.out_dir + b'/tempCoh_average'
            self.set_projection_gdal1(fhandle['temporalCoherence'][0, :, :], 1,
                                      temp_coh_file, 'Temporal coherence average', projection, geotransform)
            temp_coh_file = self.out_dir + b'/tempCoh_full'
            self.set_projection_gdal1(fhandle['temporalCoherence'][1, :, :], 1,
                                      temp_coh_file, 'Temporal coherence full', projection, geotransform)
            #ref_index_file = self.out_dir + b'/reference_index'
            #self.set_projection_gdal1_int(reference_index_map, 1,
            #                          ref_index_file, 'Reference index map', projection, geotransform)



        print('write PS mask file')

        with h5py.File(mask_ps_file.decode('UTF-8'), 'a') as psf:
           for index, box in enumerate(self.box_list):
               patch_dir = self.out_dir + ('/PATCHES/PATCH_{:04.0f}'.format(index)).encode('UTF-8')
               mask_ps = np.load(patch_dir.decode('UTF-8') + '/mask_ps.npy', allow_pickle=True)
               block = [box[1], box[3], box[0], box[2]]
               psf['mask'][block[0]:block[1], block[2]:block[3]] = mask_ps

        return
