#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:   Sara Mirzaee                                   #
############################################################

import h5py
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mintpy.utils import readfile, ptime, plot as pp, utils as ut
from mintpy.multilook import multilook_data
from mintpy import subset, view
import minopy_utilities as mnp
from skimage.measure import label


###########################################################################################
EXAMPLE = """example:
  tsview.py timeseries.h5
  tsview.py timeseries.h5  --wrap
  tsview.py timeseries.h5  --yx 300 400 --zero-first  --nodisplay
  tsview.py geo_timeseries.h5  --lalo 33.250 131.665  --nodisplay
  tsview.py timeseries_ECMWF_ramp_demErr.h5  --sub-x 900 1400 --sub-y 0 500

  # press left / right key to slide images

  # multiple time-series files
  tsview.py timeseries_ECMWF_ramp_demErr.h5 timeseries_ECMWF_ramp.h5 timeseries_ECMWF.h5 timeseries.h5 --off 5
  tsview.py timeseries_ECMWF_ramp_demErr.h5 ../GIANT/Stack/LS-PARAMS.h5 --off 5 --label mintpy giant
"""

def create_parser():
    parser = argparse.ArgumentParser(description='Interactive time-series viewer',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('--velocity', dest='velocity_file', nargs='+',
                        help='velocity file to display\n'
                             'i.e.: velocity.h5 (MintPy)\n')
    parser.add_argument('--slc', dest='slc_file', nargs='+',
                        help='slc stack file to get pixel shp\n'
                             'i.e.: slcStack.h5 (MintPy)\n')
    parser.add_argument('-rw', '--rangeWindow', dest='range_win', type=str, default='11'
                        , help='SHP searching window size in range direction. -- Default : 11')
    parser.add_argument('-aw', '--azimuthWindow', dest='azimuth_win', type=str, default='15'
                        , help='SHP searching window size in azimuth direction. -- Default : 15')

    parser.add_argument('--ylim', dest='ylim', nargs=2, metavar=('YMIN', 'YMAX'), type=float,
                        help='Y limits for point plotting.')
    parser.add_argument('--tick-right', dest='tick_right', action='store_true',
                        help='set tick and tick label to the right')

    parser.add_argument('-l', '--lookup', dest='lookup_file', type=str,
                        help='lookup table file')

    pixel = parser.add_argument_group('Pixel Input')
    pixel.add_argument('--yx', type=int, metavar=('Y', 'X'), nargs=2,
                       help='initial pixel to plot in Y/X coord')
    pixel.add_argument('--lalo', type=float, metavar=('LAT', 'LON'), nargs=2,
                       help='initial pixel to plot in lat/lon coord')

    parser.add_argument('--noverbose', dest='print_msg', action='store_false',
                        help='Disable the verbose message printing.')

    parser = pp.add_data_disp_argument(parser)
    parser = pp.add_dem_argument(parser)
    parser = pp.add_figure_argument(parser)
    parser = pp.add_gps_argument(parser)
    parser = pp.add_mask_argument(parser)
    parser = pp.add_map_argument(parser)
    parser = pp.add_point_argument(parser)
    parser = pp.add_reference_argument(parser)
    parser = pp.add_save_argument(parser)
    parser = pp.add_subset_argument(parser)

    return parser

def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if (not inps.disp_fig or inps.outfile) and not inps.save_fig:
        inps.save_fig = True
    if inps.ylim:
        inps.ylim = sorted(inps.ylim)
    if inps.zero_mask:
        inps.mask_file = 'no'

    # default value
    if not inps.disp_unit:
        inps.disp_unit = 'cm'
    if not inps.colormap:
        inps.colormap = 'jet'
    if not inps.fig_size:
        inps.fig_size = [8.0, 4.5]

    # verbose print using --noverbose option
    global vprint
    vprint = print if inps.print_msg else lambda *args, **kwargs: None

    if not inps.disp_fig:
        plt.switch_backend('Agg')

    return inps


def read_init_info(inps):
    # Time Series Info
    vel_file0 = inps.velocity_file[0]
    atr = readfile.read_attribute(vel_file0)
    inps.key = atr['FILE_TYPE']

    inps.file_label = ['velocity']

    # default mask file
    if not inps.mask_file and 'masked' not in vel_file0:
        dir_name = os.path.dirname(vel_file0)
        if 'Y_FIRST' in atr.keys():
            inps.mask_file = os.path.join(dir_name, 'geo_maskTempCoh.h5')
        else:
            inps.mask_file = os.path.join(dir_name, 'maskTempCoh.h5')
        if not os.path.isfile(inps.mask_file):
            inps.mask_file = None


    # Display Unit
    (inps.disp_unit,
     inps.unit_fac) = pp.scale_data2disp_unit(metadata=atr, disp_unit=inps.disp_unit)[1:3]

    # default lookup table file
    if not inps.lookup_file:
        inps.lookup_file = ut.get_lookup_file('./inputs/geometryRadar.h5')
    inps.coord = ut.coordinate(atr, inps.lookup_file)

    # size and lalo info
    inps.pix_box, inps.geo_box = subset.subset_input_dict2box(vars(inps), atr)
    inps.pix_box = inps.coord.check_box_within_data_coverage(inps.pix_box)
    inps.geo_box = inps.coord.box_pixel2geo(inps.pix_box)
    data_box = (0, 0, int(atr['WIDTH']), int(atr['LENGTH']))
    vprint('data   coverage in y/x: '+str(data_box))
    vprint('subset coverage in y/x: '+str(inps.pix_box))
    vprint('data   coverage in lat/lon: '+str(inps.coord.box_pixel2geo(data_box)))
    vprint('subset coverage in lat/lon: '+str(inps.geo_box))
    vprint('------------------------------------------------------------------------')

    # reference pixel
    if not inps.ref_lalo and 'REF_LAT' in atr.keys():
        inps.ref_lalo = (float(atr['REF_LAT']), float(atr['REF_LON']))
    if inps.ref_lalo:
        if inps.ref_lalo[1] > 180.:
            inps.ref_lalo[1] -= 360.
        inps.ref_yx = inps.coord.geo2radar(inps.ref_lalo[0], inps.ref_lalo[1], print_msg=False)[0:2]
    if not inps.ref_yx:
        inps.ref_yx = [int(atr['REF_Y']), int(atr['REF_X'])]

    # Initial Pixel Coord
    if inps.lalo:
        inps.yx = inps.coord.geo2radar(inps.lalo[0], inps.lalo[1], print_msg=False)[0:2]
    try:
        inps.lalo = inps.coord.radar2geo(inps.yx[0], inps.yx[1], print_msg=False)[0:2]
    except:
        inps.lalo = None

    # Flip up-down / left-right
    if inps.auto_flip:
        inps.flip_lr, inps.flip_ud = pp.auto_flip_direction(atr, print_msg=inps.print_msg)

    # Transparency - Alpha
    if not inps.transparency:
        # Auto adjust transparency value when showing shaded relief DEM
        if inps.dem_file and inps.disp_dem_shade:
            inps.transparency = 0.7
        else:
            inps.transparency = 1.0

    # display unit ans wrap
    # if wrap_step == 2*np.pi (default value), set disp_unit_img = radian;
    # otherwise set disp_unit_img = disp_unit
    inps.disp_unit_img = inps.disp_unit
    if inps.wrap:
        inps.range2phase = -4. * np.pi / float(atr['WAVELENGTH'])
        if   'cm' == inps.disp_unit.split('/')[0]:   inps.range2phase /= 100.
        elif 'mm' == inps.disp_unit.split('/')[0]:   inps.range2phase /= 1000.
        elif 'm'  == inps.disp_unit.split('/')[0]:   inps.range2phase /= 1.
        else:
            raise ValueError('un-recognized display unit: {}'.format(inps.disp_unit))

        if (inps.wrap_range[1] - inps.wrap_range[0]) == 2*np.pi:
            inps.disp_unit_img = 'radian'
        inps.vlim = inps.wrap_range
    inps.cbar_label = 'Displacement [{}]'.format(inps.disp_unit_img)
    return inps, atr


def _adjust_ts_axis(ax, inps):
    ax.tick_params(which='both', direction='in', labelsize=inps.font_size, bottom=True, top=True, left=True, right=True)
    ax = pp.auto_adjust_xaxis_date(ax, inps.yearList, fontsize=inps.font_size)[0]
    ax.set_xlabel('Time [years]', fontsize=inps.font_size)
    ax.set_ylabel('Displacement [{}]'.format(inps.disp_unit), fontsize=inps.font_size)
    ax.set_ylim(inps.ylim)
    return ax


def _get_ts_title(y, x, coord):
    title = 'Y/X = {}, {}'.format(y, x)
    try:
        lat, lon = coord.radar2geo(y, x, print_msg=False)[0:2]
        title += ', lat/lon = {:.4f}, {:.4f}'.format(lat, lon)
    except:
        pass
    return title


def save_coh_plot(yx, fig_img, fig_pts, inps):
    vprint('save info on pixel ({}, {})'.format(yx[0], yx[1]))
    # output file name
    if inps.outfile:
        inps.outfile_base, ext = os.path.splitext(inps.outfile[0])
        if ext != '.pdf':
            vprint(('Output file extension is fixed to .pdf,'
                    ' input extension {} is ignored.').format(ext))
    else:
        inps.outfile_base = 'y{}_x{}'.format(yx[0], yx[1])

    # Figure - point time-series
    outName = '{}_coh_matrix.pdf'.format(inps.outfile_base)
    fig_pts.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save time-series plot to '+outName)

    # Figure - map
    outName = '{}_{}.png'.format(inps.outfile_base, inps.date_list[inps.idx])
    fig_img.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save map plot to '+outName)
    return


def read_velocity_data(inps):
    """Read data of time-series files
    Parameters: inps : Namespace of input arguments
    Returns:    ts_data : list of 3D np.array in size of (num_date, length, width)
                mask : 2D np.array in size of (length, width)
                inps : Namespace of input arguments
    """

    # read velocity file
    fname = 'velocity'
    vprint('reading velocity from file {} ...'.format(fname))
    velocity_data, atr = readfile.read(fname, datasetName=fname, box=inps.pix_box)

    # Display Unit
    (data,
     inps.disp_unit,
     inps.unit_fac) = pp.scale_data2disp_unit(velocity_data,
                                              metadata=atr,
                                              disp_unit=inps.disp_unit)

    # Mask file: input mask file + non-zero ts pixels - ref_point
    mask = np.ones(data[0].shape[-2:], np.bool_)
    msk = pp.read_mask(inps.timeseries_file[0],
                       mask_file=inps.mask_file,
                       datasetName='displacement',
                       box=inps.pix_box,
                       print_msg=inps.print_msg)[0]
    mask[msk == 0.] = False
    del msk

    # default vlim
    inps.dlim = [np.nanmin(velocity_data), np.nanmax(velocity_data)]
    vel_data_mli = multilook_data(np.squeeze(velocity_data), 10, 10)
    if not inps.vlim:
        inps.vlim = [np.nanmin(vel_data_mli[inps.ex_flag != 0]),
                     np.nanmax(vel_data_mli[inps.ex_flag != 0])]
    vprint('data    range: {} {}'.format(inps.dlim, inps.disp_unit))
    vprint('display range: {} {}'.format(inps.vlim, inps.disp_unit))

    # default ylim
    num_file = len(inps.timeseries_file)
    if not inps.ylim:
        vel_data_mli = multilook_data(np.squeeze(velocity_data), 4, 4)
        ymin, ymax = (np.nanmin(vel_data_mli[inps.ex_flag != 0]),
                      np.nanmax(vel_data_mli[inps.ex_flag != 0]))
        ybuffer = (ymax - ymin) * 0.05
        inps.ylim = [ymin - ybuffer, ymax + ybuffer]
        if inps.offset:
            inps.ylim[1] += inps.offset * (num_file - 1)
    del vel_data_mli

    return velocity_data, mask, inps



class CoherenceViewer():
    """Class for tsview.py

    Example:
        cmd = 'tsview.py timeseries_ECMWF_ramp_demErr.h5'
        obj = timeseriesViewer(cmd)
        obj.configure()
        obj.plot()
    """

    def __init__(self, cmd=None, iargs=None):
        if cmd:
            iargs = cmd.split()[1:]
        self.cmd = cmd
        self.iargs = iargs
        # print command line
        cmd = '{} '.format(os.path.basename(__file__))
        cmd += ' '.join(iargs)
        print(cmd)

        # figure variables
        self.figname_img = 'Velocity map'
        self.figsize_img = None
        self.fig_img = None
        self.ax_img = None
        self.cbar_img = None
        self.img = None

        self.ax_tslider = None
        self.tslider = None

        self.figname_pts = 'Point Temporal Coherence'
        self.figsize_pts = None
        self.fig_pts = None
        self.ax_pts = None
        return

    def configure(self):
        inps = cmd_line_parse(self.iargs)
        inps, self.atr = read_init_info(inps)
        # copy inps to self object
        for key, value in inps.__dict__.items():
            setattr(self, key, value)
        # input figsize for the point time-series plot
        self.figsize_pts = self.fig_size
        self.pts_marker = 'r^'

        self.slcStack = inps.slc_file
        self.range_window = int(inps.range_win)
        self.azimuth_window = int(inps.azimuth_win)
        return


    def plot(self):
        # read 3D time-series
        self.vel_data, self.mask = read_velocity_data(self)[0:2]

        # Figure 1 - Velocity Map
        self.fig_img = plt.figure(self.figname_img, figsize=self.figsize_img)

        # Figure 1 - Axes 1 - Velocity Map
        self.ax_img = self.fig_img.add_axes([0.125, 0.25, 0.75, 0.65])
        img_data = np.array(self.vel_data[:, :])  ####################
        img_data[self.mask == 0] = np.nan
        self.plot_init_image(img_data)

        # Figure 2 - Temporal Coherence - Point
        self.fig_pts, self.ax_pts = plt.subplots(nrows=1, ncols=2, figsize=self.figsize_pts)
        if self.yx:
            d_coh = self.plot_point_coh_matrix(self.yx)

        # Output
        if self.save_fig:
            save_coh_plot(self.yx, self.fig_img, self.fig_pts, self)

        # Final linking of the canvas to the plots.
        self.fig_img.canvas.mpl_connect('button_press_event', self.update_plot_timeseries)
        if self.disp_fig:
            vprint('showing ...')
            plt.show()
        return

    def plot_init_image(self, img_data):
        # prepare data
        if self.wrap:
            if self.disp_unit_img == 'radian':
                img_data *= self.range2phase
            img_data = ut.wrap(img_data, wrap_range=self.wrap_range)

        # Title and Axis Label
        self.fig_title = 'Velocity Map'

        # Initial Pixel
        if self.yx and self.yx != self.ref_yx:
            self.pts_yx = np.array(self.yx).reshape(-1, 2)
            if self.lalo:
                self.pts_lalo = np.array(self.lalo).reshape(-1, 2)
            else:
                self.pts_lalo = None

        # call view.py to plot
        self.img, self.cbar_img = view.plot_slice(self.ax_img, img_data, self.atr, self)[2:4]
        return self.img, self.cbar_img

    def plot_point_coh_matrix(self, yx):
        """Plot point displacement time-series at pixel [y, x]
        Parameters: yx : list of 2 int
        Returns:    d_ts : 2D np.array in size of (num_date, num_date)
        """
        self.ax_pts.cla()

        slc_file = h5py.File(self.slcStack, 'r')

        length = slc_file['slc'][:, :].shape[1]
        width = slc_file['slc'][:, :].shape[2]
        n_image = slc_file['slc'][:, :].shape[0]
        num_slc = 20

        row = yx[0] - self.pix_box[1]
        col = yx[1] - self.pix_box[0]

        sample_rows = np.ogrid[-((self.azimuth_window - 1) / 2):((self.azimuth_window - 1) / 2) + 1]
        sample_rows = sample_rows.astype(int)
        reference_row = np.array([(self.azimuth_window - 1) / 2]).astype(int)
        reference_row = reference_row - (self.azimuth_window - len(sample_rows))

        sample_cols = np.ogrid[-((self.range_window - 1) / 2):((self.range_window - 1) / 2) + 1]
        sample_cols = sample_cols.astype(int)
        reference_col = np.array([(self.range_window - 1) / 2]).astype(int)
        reference_col = reference_col - (self.range_window - len(sample_cols))

        sample_rows = row + sample_rows
        sample_rows[sample_rows < 0] = -1
        sample_rows[sample_rows >= length] = -1

        sample_cols = col + sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= width] = -1

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        win = np.abs(slc_file['slc'][n_image - num_slc::, y, x])

        testvec = np.sort(win.reshape(num_slc, self.azimuth_window * self.range_window), axis=0)
        ksres = np.zeros(self.azimuth_window * self.range_window).astype(int)

        S1 = np.abs(slc_file['slc'][n_image - num_slc::, row, col]).reshape(num_slc, 1)
        S1 = np.sort(S1.flatten())

        x = x.flatten()
        y = y.flatten()

        distance_thresh = mnp.ks_lut(num_slc, num_slc, alpha=0.05)

        for m in range(testvec.shape[1]):
            if x[m] >= 0 and y[m] >= 0:
                S2 = testvec[:, m]
                S2 = np.sort(S2.flatten())
                ksres[m] = mnp.ks2smapletest(S1, S2, threshold=distance_thresh)

        ks_label = label(ksres.reshape(self.azimuth_window, self.range_window), background=False, connectivity=2)
        shp = 1 * (ks_label == ks_label[reference_row, reference_col])

        shp_rows, shp_cols = np.where(shp == 1)
        shp_rows = np.array(shp_rows + row - (self.azimuth_window - 1) / 2).astype(int)
        shp_cols = np.array(shp_cols + col - (self.range_window - 1) / 2).astype(int)

        CCG = np.matrix(1.0 * np.arange(n_image * len(shp_rows)).reshape(n_image, len(shp_rows)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(slc_file['slc'][:, shp_rows, shp_cols])

        coh_mat = mnp.est_corr(CCG)

        slc_file.close()

        cmap, norm = mnp.custom_cmap()
        im1 = self.ax_pts.imshow(np.abs(coh_mat), cmap='jet', norm=norm)
        cbar = plt.colorbar(im1, ticks=[0, 1], orientation='vertical')

        im2 = self.ax_pts.imshow(np.angle(coh_mat), cmap='jet', norm=norm)
        cbar = plt.colorbar(im2, ticks=[-np.pi, np.pi], orientation='vertical')

        plt.show()

        return coh_mat

    def update_plot_timeseries(self, event):
        """Event function to get y/x from button press"""
        if event.inaxes == self.ax_img:
            # get row/col number
            if self.fig_coord == 'geo':
                y, x = self.coord.geo2radar(event.ydata, event.xdata, print_msg=False)[0:2]
            else:
                y, x = int(event.ydata + 0.5), int(event.xdata + 0.5)

            # plot time-series displacement
            self.plot_point_coh_matrix((y, x))
        return

###########################################################################################
def main(iargs=None):
    obj = CoherenceViewer(iargs=iargs)
    obj.configure()
    obj.plot()
    return


#########################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
