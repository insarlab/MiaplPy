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
import minopy_utilities as mnp
from skimage.measure import label
from mintpy.tsview import *   


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

def tcoh_create_parser():
    parser = create_parser()
    parser.add_argument('--slc', dest='slc_file', nargs='+',
                        help='slc stack file to get pixel shp\n'
                             'i.e.: slcStack.h5 (MintPy)\n')
    parser.add_argument('-rw', '--rangeWindow', dest='range_win', type=str, default='11'
                        , help='SHP searching window size in range direction. -- Default : 11')
    parser.add_argument('-aw', '--azimuthWindow', dest='azimuth_win', type=str, default='15'
                        , help='SHP searching window size in azimuth direction. -- Default : 15')

    return parser

def cmd_line_parse(iargs=None):
    parser = tcoh_create_parser()
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

def save_ts_plot(yx, fig_img, fig_pts, d_ts, fig_coh, inps):
    vprint('save info on pixel ({}, {})'.format(yx[0], yx[1]))
    # output file name
    if inps.outfile:
        inps.outfile_base, ext = os.path.splitext(inps.outfile[0])
        if ext != '.pdf':
            vprint(('Output file extension is fixed to .pdf,'
                    ' input extension {} is ignored.').format(ext))
    else:
        inps.outfile_base = 'y{}_x{}'.format(yx[0], yx[1])

    # get aux info
    vel, std = estimate_slope(d_ts[0], inps.yearList,
                              ex_flag=inps.ex_flag,
                              disp_unit=inps.disp_unit)

    # TXT - point time-series
    outName = '{}_ts.txt'.format(inps.outfile_base)
    header_info = 'timeseries_file={}\n'.format(inps.timeseries_file)
    header_info += '{}\n'.format(_get_ts_title(yx[0], yx[1], inps.coord))
    header_info += 'reference pixel: y={}, x={}\n'.format(inps.ref_yx[0], inps.ref_yx[1])
    header_info += 'reference date: {}\n'.format(inps.date_list[inps.ref_idx])
    header_info += 'unit: {}\n'.format(inps.disp_unit)
    header_info += 'slope: {:.2f} +/- {:.2f} [{}/yr]'.format(vel, std, inps.disp_unit)

    # prepare data
    data = np.array(inps.date_list).reshape(-1, 1)
    for i in range(len(d_ts)):
        data = np.hstack((data, d_ts[i].reshape(-1, 1)))
    # write
    np.savetxt(outName,
               data,
               fmt='%s',
               delimiter='\t',
               header=header_info)
    vprint('save displacement time-series in meter to '+outName)

    # Figure - point time-series
    outName = '{}_ts.pdf'.format(inps.outfile_base)
    fig_pts.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save time-series plot to '+outName)

    # Figure - map
    outName = '{}_{}.png'.format(inps.outfile_base, inps.date_list[inps.idx])
    fig_img.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save map plot to '+outName)

    # Figure - map
    outName = '{}_{}_tcoh.png'.format(inps.outfile_base, inps.date_list[inps.idx])
    fig_coh.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save map plot to '+outName)

    return


class CoherenceViewer(timeseriesViewer):
    """Class for tsview.py

    Example:
        cmd = 'tsview.py timeseries_ECMWF_ramp_demErr.h5'
        obj = timeseriesViewer(cmd)
        obj.configure()
        obj.plot()
    """

    def __init__(self, cmd=None, iargs=None):
        super().__init__(cmd, iargs)
        if cmd:
            iargs = cmd.split()[1:]
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
        self.ts_data, self.mask = read_timeseries_data(self)[0:2]

        # Figure 1 - Cumulative Displacement Map
        self.fig_img = plt.figure(self.figname_img, figsize=self.figsize_img)

        # Figure 1 - Axes 1 - Displacement Map
        self.ax_img = self.fig_img.add_axes([0.125, 0.25, 0.75, 0.65])
        img_data = np.array(self.ts_data[0][self.idx, :, :])  ####################
        img_data[self.mask == 0] = np.nan
        self.plot_init_image(img_data)

        # Figure 1 - Axes 2 - Time Slider
        self.ax_tslider = self.fig_img.add_axes([0.2, 0.1, 0.6, 0.07])
        self.plot_init_time_slider(init_idx=self.idx, ref_idx=self.ref_idx)
        self.tslider.on_changed(self.update_time_slider)

        # Figure 2 - Time Series Displacement - Point
        self.fig_pts, self.ax_pts = plt.subplots(num=self.figname_pts, figsize=self.figsize_pts)
        if self.yx:
            d_ts = self.plot_point_timeseries(self.yx)

        # Figure 3 - Temporal Coherence - Point
        self.fig_coh, self.ax_coh = plt.subplots(nrows=1, ncols=2)
        if self.yx:
            d_coh = self.plot_point_coh_matrix(self.yx)

        # Output
        if self.save_fig:
            save_ts_plot(self.yx, self.fig_img, self.fig_pts, d_ts, self.fig_coh, self)

        # Final linking of the canvas to the plots.
        self.fig_img.canvas.mpl_connect('button_press_event', self.update_plot_timeseries)
        if self.disp_fig:
            vprint('showing ...')
            plt.show()
        return

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

        #plt.show()

        return coh_mat


###########################################################################################
def main(iargs=None):
    obj = CoherenceViewer(iargs=iargs)
    obj.configure()
    obj.plot()
    return

#########################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
