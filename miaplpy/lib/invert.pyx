#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
cimport utils as iut
import utils as iut
import os
from libc.stdio cimport printf
from miaplpy.objects.slcStack import slcStack
import h5py
import time
from isceobj.Util.ImageUtil import ImageLib as IML



cdef void write_wrapped(list date_list, bytes out_dir, int width, int length, bytes RSLCfile, bytes date):

    cdef int d = date_list.index(date.decode('UTF-8'))
    cdef bytes out_name, wrap_date
    cdef object fhandle
    cdef float complex[:, :, ::1] out_rslc

    printf('write wrapped_phase {}'.format(date.decode('UTF-8')))
    wrap_date = os.path.join(out_dir, b'wrapped_phase', date)
    os.makedirs(wrap_date.decode('UTF-8'), exist_ok=True)
    out_name = os.path.join(wrap_date, date + b'.slc')

    if not os.path.exists(out_name.decode('UTF-8')):
        fhandle = h5py.File(RSLCfile.decode('UTF-8'), 'r')
        out_rslc = np.memmap(out_name, dtype='complex64', mode='w+', shape=(length, width))
        out_rslc[:, :] = fhandle['slc'][d, :, :]
        fhandle.close()
        IML.renderISCEXML(out_name.decode('UTF-8'), bands=1, nyy=length, nxx=width, datatype='complex64',
                          scheme='BSQ')
    else:
        IML.renderISCEXML(out_name.decode('UTF-8'), bands=1, nyy=length, nxx=width, datatype='complex64',
                          scheme='BSQ')
    return


cdef void write_hdf5_block_3D(object fhandle, float[:, :, ::1] data, bytes datasetName, list block):

    fhandle[datasetName.decode('UTF-8')][block[0]:block[1], block[2]:block[3], block[4]:block[5]] = data
    return

cdef void write_hdf5_block_2D_int(object fhandle, int[:, ::1] data, bytes datasetName, list block):

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
        os.makedirs(self.out_dir.decode('UTF-8'), exist_ok='True')

        self.slcStackObj = slcStack(inps.slc_stack)
        self.metadata = self.slcStackObj.get_metadata()
        self.all_date_list = self.slcStackObj.get_date_list()
        with h5py.File(inps.slc_stack, 'r') as f:
            self.prep_baselines = f['bperp'][:]
        self.n_image, self.length, self.width = self.slcStackObj.get_size()
        self.time_lag = inps.time_lag


        # total number of neighbouring pixels
        self.shp_size = self.range_window * self.azimuth_window

        # threshold for shp test based on number of images to test
        alpha = 0.01
        if self.shp_test == b'ks':
            self.distance_thresh = iut.ks_lut_cy(self.n_image, self.n_image, alpha)
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

        self.RSLCfile = self.out_dir + b'/phase_series.h5'


        if b'sequential' == self.phase_linking_method[0:10]:
            self.sequential = True
        else:
            self.sequential = False
        return

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
                self.metadata['DATA_TYPE'] = 'int32'
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
                                    dtype=np.int32)
                psf['mask'][:, :] = 0

        return

    def get_datakwargs(self):

        cdef dict data_kwargs = {
            "range_window" : self.range_window,
            "azimuth_window" : self.azimuth_window,
            "width" : self.width,
            "length" : self.length,
            "n_image" : self.n_image,
            "slcStackObj" : self.slcStackObj,
            "distance_threshold" : self.distance_thresh,
            "def_sample_rows" : np.array(self.sample_rows),
            "def_sample_cols" : np.array(self.sample_cols),
            "reference_row" : self.reference_row,
            "reference_col" : self.reference_col,
            "phase_linking_method" : self.phase_linking_method,
            "total_num_mini_stacks" : self.total_num_mini_stacks,
            "default_mini_stack_size" : self.mini_stack_default_size,
            'ps_shp': self.ps_shp,
            "shp_test": self.shp_test,
            "out_dir": self.out_dir,
            "time_lag": self.time_lag,
            "mask_file": self.mask_file,
        }
        return data_kwargs


    def loop_patches(self, list box_list):

        cdef float m, s, start_time = time.time()
        cdef cnp.ndarray[int, ndim=1] box
        cdef dict data_kwargs = self.get_datakwargs()

        for box in box_list:

            data_kwargs['box'] = box
            os.makedirs(self.out_dir.decode('UTF-8') + '/PATCHES', exist_ok=True)
            iut.process_patch_c(**data_kwargs)

        m, s = divmod(time.time() - start_time, 60)
        print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
        return

    def unpatch(self):
        cdef list block
        cdef object fhandle, psf
        cdef int index, box_length, box_width
        cdef cnp.ndarray[int, ndim=1] box
        cdef bytes patch_dir
        cdef float complex[:, :, ::1] rslc_ref
        cdef cnp.ndarray[float, ndim=3] temp_coh, ps_prod, eig_values = np.zeros((3, self.length, self.width), dtype=np.float32)
        cdef cnp.ndarray[float, ndim=2] amp_disp = np.zeros((self.length, self.width), dtype=np.float32)

        if os.path.exists(self.RSLCfile.decode('UTF-8')):
            print('Deleting old phase_series.h5 ...')
            os.remove(self.RSLCfile.decode('UTF-8'))

        mask_ps_file = self.work_dir + b'/maskPS.h5'
        if os.path.exists(mask_ps_file.decode('UTF-8')):
            os.remove(mask_ps_file.decode('UTF-8'))

        self.initiate_output()
        print('Concatenate and write wrapped phase time series to HDF5 file phase_series.h5 ')
        print('open  HDF5 file phase_series.h5 in a mode')

        with h5py.File(self.RSLCfile.decode('UTF-8'), 'a') as fhandle:
            
            for index, box in enumerate(self.box_list):
                box_width = box[2] - box[0]
                box_length = box[3] - box[1]

                patch_dir = self.out_dir + ('/PATCHES/PATCH_{:04.0f}'.format(index)).encode('UTF-8')
                rslc_ref = np.load(patch_dir.decode('UTF-8') + '/phase_ref.npy', allow_pickle=True)
                temp_coh = np.load(patch_dir.decode('UTF-8') + '/tempCoh.npy', allow_pickle=True)
                shp = np.load(patch_dir.decode('UTF-8') + '/shp.npy', allow_pickle=True)
                mask_ps = np.load(patch_dir.decode('UTF-8') + '/mask_ps.npy', allow_pickle=True)
                ps_prod = np.load(patch_dir.decode('UTF-8') + '/ps_products.npy', allow_pickle=True)

                temp_coh[temp_coh<0] = 0

                print('-' * 50)
                print("Concatenate block {}/{} : {}".format(index, self.num_box, box[0:4]))

                # wrapped interferograms 3D
                block = [0, self.n_image, box[1], box[3], box[0], box[2]]
                #write_hdf5_block_3D(fhandle, rslc_ref, b'slc', block)
                write_hdf5_block_3D(fhandle, np.angle(rslc_ref), b'phase', block)
                write_hdf5_block_3D(fhandle, np.abs(rslc_ref), b'amplitude', block)

                # SHP - 2D
                block = [box[1], box[3], box[0], box[2]]
                write_hdf5_block_2D_int(fhandle, shp, b'shp', block)
                amp_disp[block[0]:block[1], block[2]:block[3]] = ps_prod[0, :, :]

                # temporal coherence - 3D
                block = [0, 2, box[1], box[3], box[0], box[2]]
                write_hdf5_block_3D(fhandle, temp_coh, b'temporalCoherence', block)
                eig_values[0:3, block[2]:block[3], block[4]:block[5]] = ps_prod[1:4, :, :]

            print('write amplitude dispersion and top eigen values')
            amp_disp_file = self.out_dir + b'/amp_dipersion_index'
            if not os.path.exists(amp_disp_file.decode('UTF-8')):
                amp_disp_memmap = np.memmap(amp_disp_file.decode('UTF-8'), mode='write', dtype='float32',
                                           shape=(self.length, self.width))
                IML.renderISCEXML(amp_disp_file.decode('UTF-8'), bands=1, nyy=self.length, nxx=self.width,
                                  datatype='float32', scheme='BIL')
            else:
                amp_disp_memmap = np.memmap(amp_disp_file.decode('UTF-8'), mode='r+', dtype='float32',
                                           shape=(self.length, self.width))

            amp_disp_memmap[:, :] = amp_disp[:, :]
            amp_disp_memmap = None

            top_eig_files = self.out_dir + b'/top_eigenvalues'
            if not os.path.exists(top_eig_files.decode('UTF-8')):
                top_eig_memmap = np.memmap(top_eig_files.decode('UTF-8'), mode='write', dtype='float32',
                                           shape=(3, self.length, self.width))
                IML.renderISCEXML(top_eig_files.decode('UTF-8'), bands=3, nyy=self.length, nxx=self.width,
                                  datatype='float32', scheme='BSQ')
            else:
                top_eig_memmap = np.memmap(top_eig_files.decode('UTF-8'), mode='r+', dtype='float32',
                                           shape=(3, self.length, self.width))

            print(top_eig_memmap.shape, np.array(eig_values).shape)
            top_eig_memmap[0:3, :, :] = eig_values[:, :, :]
            #top_eig_memmap[2, :, :] = eig_values[1, :, :]/eig_values[0, :, :]
            rr, cc, kk = np.where(np.isnan(top_eig_memmap))
            top_eig_memmap[:, rr, cc] = np.nan
            top_eig_memmap = None

            print('write averaged temporal coherence file from mini stacks')
            temp_coh_file = self.out_dir + b'/tempCoh_average'

            if not os.path.exists(temp_coh_file.decode('UTF-8')):
                temp_coh_memmap = np.memmap(temp_coh_file.decode('UTF-8'), mode='write', dtype='float32',
                                           shape=(self.length, self.width))
                IML.renderISCEXML(temp_coh_file.decode('UTF-8'), bands=1, nyy=self.length, nxx=self.width,
                                  datatype='float32', scheme='BIL')
            else:
                temp_coh_memmap = np.memmap(temp_coh_file.decode('UTF-8'), mode='r+', dtype='float32',
                                           shape=(self.length, self.width))

            temp_coh_memmap[:, :] = fhandle['temporalCoherence'][0, :, :]
            temp_coh_memmap = None

            print('write temporal coherence file from full stack')
            temp_coh_file = self.out_dir + b'/tempCoh_full'

            if not os.path.exists(temp_coh_file.decode('UTF-8')):
                temp_coh_memmap = np.memmap(temp_coh_file.decode('UTF-8'), mode='write', dtype='float32',
                                           shape=(self.length, self.width))
                IML.renderISCEXML(temp_coh_file.decode('UTF-8'), bands=1, nyy=self.length, nxx=self.width,
                                  datatype='float32', scheme='BIL')
            else:
                temp_coh_memmap = np.memmap(temp_coh_file.decode('UTF-8'), mode='r+', dtype='float32',
                                           shape=(self.length, self.width))

            temp_coh_memmap[:, :] = fhandle['temporalCoherence'][1, :, :]
            temp_coh_memmap = None

            print('close HDF5 file phase_series.h5.')

        print('write PS mask file')

        with h5py.File(mask_ps_file.decode('UTF-8'), 'a') as psf:
           for index, box in enumerate(self.box_list):
               patch_dir = self.out_dir + ('/PATCHES/PATCH_{:04.0f}'.format(index)).encode('UTF-8')
               mask_ps = np.load(patch_dir.decode('UTF-8') + '/mask_ps.npy', allow_pickle=True)
               block = [box[1], box[3], box[0], box[2]]
               write_hdf5_block_2D_int(psf, mask_ps, b'mask', block)

        return



