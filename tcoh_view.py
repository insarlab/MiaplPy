#!/usr/bin/env python3
############################################################
# Program is part of MiNoPy                                #
# Author:   Sara Mirzaee                                   #
############################################################

import h5py
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import minopy_utilities as mnp
from skimage.measure import label
from mintpy import subset, view
#from mintpy.tsview import *
import mintpy.tsview as tsview

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
    parser = tsview.create_parser()
    parser.add_argument('--slc', dest='slc_file', nargs='+',
                        help='slc stack file to get pixel shp\n'
                             'i.e.: slcStack.h5 (MintPy)\n')
    parser.add_argument('-rw', '--rangeWindow', dest='range_win', type=str, default='19'
                        , help='SHP searching window size in range direction. -- Default : 19')
    parser.add_argument('-aw', '--azimuthWindow', dest='azimuth_win', type=str, default='21'
                        , help='SHP searching window size in azimuth direction. -- Default : 21')

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
    tsview.vprint = print if inps.print_msg else lambda *args, **kwargs: None

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
    vel, std = tsview.estimate_slope(d_ts[0], inps.yearList,
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


class CoherenceViewer(tsview.timeseriesViewer):
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
        inps, self.atr = tsview.read_init_info(inps)
        # copy inps to self object
        for key, value in inps.__dict__.items():
            setattr(self, key, value)
        # input figsize for the point time-series plot
        self.figsize_pts = self.fig_size
        self.pts_marker = 'r^'
        self.slcStack = inps.slc_file[0]
        self.range_window = int(inps.range_win)
        self.azimuth_window = int(inps.azimuth_win)

        slc_file = h5py.File(self.slcStack, 'r')
        self.rslc = slc_file['slc'][:, :, :]
        slc_file.close()
        self.length = self.rslc.shape[1]
        self.width = self.rslc.shape[2]
        self.n_image = self.rslc.shape[0]
        self.num_slc = self.n_image

        return


    def plot(self):
        # read 3D time-series
        self.ts_data, self.mask = tsview.read_timeseries_data(self)[0:2]

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
        self.fig_coh, self.ax_coh = plt.subplots(nrows=2, ncols=2)
        if self.yx:
            d_coh = self.plot_point_coh_matrix(self.yx)

        # Output
        if self.save_fig:
            save_ts_plot(self.yx, self.fig_img, self.fig_pts, d_ts, self.fig_coh, self)

        # Final linking of the canvas to the plots.
        self.fig_img.canvas.mpl_connect('button_press_event', self.update_plot_timeseries)
        self.fig_img.canvas.mpl_connect('button_press_event', self.update_plot_coh)
        self.fig_img.canvas.mpl_connect('key_press_event', self.on_key_event)
        if self.disp_fig:
            vprint('showing ...')
            msg = '\n------------------------------------------------------------------------'
            msg += '\nTo scroll through the image sequence:'
            msg += '\n1) Move the slider, OR'
            msg += '\n2) Press left or right arrow key (if not responding, click the image and try again).'
            msg += '\n------------------------------------------------------------------------'
            vprint(msg)
            plt.show()
        return

    def plot_point_coh_matrix(self, yx):
        """Plot point displacement time-series at pixel [y, x]
        Parameters: yx : list of 2 int
        Returns:    d_ts : 2D np.array in size of (num_date, num_date)
        """
        self.ax_coh[0, 0].cla()
        self.ax_coh[0, 1].cla()
        self.ax_coh[1, 0].cla()
        self.ax_coh[1, 1].cla()

        row = yx[0] - self.pix_box[1]
        col = yx[1] - self.pix_box[0]

        time0 = time.time()

        distance_thresh = mnp.ks_lut(self.num_slc, self.num_slc, alpha=0.05)

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
        sample_rows[sample_rows >= self.length] = -1

        sample_cols = col + sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= self.width] = -1

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        win = np.abs(self.rslc[0:self.n_image, y, x])

        mask = 1 * (x >= 0) * (y >= 0)
        indx = np.where(mask == 1)
        x = x[indx[0], indx[1]]
        y = y[indx[0], indx[1]]

        testvec = np.sort(np.abs(self.rslc[:, y, x]), axis=0)
        S1 = np.sort(np.abs(self.rslc[:, row, col])).reshape(self.n_image, 1)

        data1 = np.repeat(S1, testvec.shape[1], axis=1)
        data_all = np.concatenate((data1, testvec), axis=0)

        res = np.zeros([self.azimuth_window, self.range_window])
        res[indx[0], indx[1]] = 1 * (np.apply_along_axis(mnp.ecdf_distance, 0, data_all) <= distance_thresh)

        ks_label = label(res, background=0, connectivity=2)
        shp = 1 * (ks_label == ks_label[reference_row, reference_col]) * mask

        shp_rows, shp_cols = np.where(shp == 1)
        shp_rows = np.array(shp_rows + row - (self.azimuth_window - 1) / 2).astype(int)
        shp_cols = np.array(shp_cols + col - (self.range_window - 1) / 2).astype(int)

        CCG = np.matrix(1.0 * np.arange(self.n_image * len(shp_rows)).reshape(self.n_image, len(shp_rows)))
        CCG = np.exp(1j * CCG)
        CCG[:, :] = np.matrix(self.rslc[:, shp_rows, shp_cols])

        coh_mat = mnp.est_corr(CCG)

        vec_refined = mnp.phase_linking_process(coh_mat, 0, 'EMI', squeez=False)
        tcoh = mnp.gam_pta(np.angle(coh_mat), vec_refined)

        cmap, norm = mnp.custom_cmap()
        im1 = self.ax_coh[0, 0].imshow(np.abs(coh_mat), cmap='jet', norm=norm)

        cmap, norm = mnp.custom_cmap(-np.pi, np.pi)
        im2 = self.ax_coh[0, 1].imshow(np.angle(coh_mat), cmap='jet', norm=norm)

        im3 = self.ax_coh[1, 0].imshow(np.mean(win[:, :, :], axis=0), cmap='jet')

        im4 = self.ax_coh[1, 1].imshow(shp, cmap='jet')

        self.fig_coh.canvas.draw()

        vprint('minimum coherence: {}\n'
               'mean coherence: {}\n'
               'num shp: {}\n'
               'temporal coherence: {}'
               .format(np.nanmin(np.abs(coh_mat)),
                       np.mean(np.abs(coh_mat)),
                       len(shp_cols),
                       tcoh))

        vprint('showing Coherence matrix for ({})'.format(yx))

        return coh_mat

    def update_plot_coh(self, event):
        """Event function to get y/x from button press"""
        if event.inaxes == self.ax_img:

            # get row/col number
            if self.fig_coord == 'geo':
                y, x = self.coord.geo2radar(event.ydata, event.xdata, print_msg=False)[0:2]
            else:
                y, x = int(event.ydata+0.5), int(event.xdata+0.5)

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
