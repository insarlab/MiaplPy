#!/usr/bin/env python3
############################################################
# Program is part of MiaplPy                                #
# Author:   Sara Mirzaee                                   #
############################################################

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from miaplpy.objects.invert_pixel import ks_lut_cy, get_shp_row_col_c, custom_cmap, gam_pta
from mintpy.utils import arg_group, ptime, time_func, readfile, plot as pp
from miaplpy.objects.slcStack import slcStack
import miaplpy.lib.utils as iut
from mintpy import timeseries2velocity as ts2vel
import mintpy.tsview as tsview
import argparse

###########################################################################################
EXAMPLE = """example:
  tcoh_view.py timeseries.h5 --slc ../inputs/slcStack.h5
  tcoh_view.py timeseries.h5 --slc ../inputs/slcStack.h5 -rw 19 -aw 9 
  tcoh_view.py timeseries.h5 --slc ../inputs/slcStack.h5 --wrap
  tcoh_view.py timeseries.h5 --slc ../inputs/slcStack.h5 --yx 300 400 --zero-first  --nodisplay
  tcoh_view.py timeseries_ECMWF_ramp_demErr.h5 --slc ../inputs/slcStack.h5  --sub-x 900 1400 --sub-y 0 500

  # press left / right key to slide images

"""

def create_parser():
    parser = argparse.ArgumentParser(description='Interactive time-series viewer',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)
    parser.add_argument('file', nargs='+',
                        help='time-series file to display\n'
                             'i.e.: timeseries_ERA5_ramp_demErr.h5 (MintPy)\n'
                             '      LS-PARAMS.h5 (GIAnT)\n'
                             '      S1_IW12_128_0593_0597_20141213_20180619.he5 (HDF-EOS5)')
    parser.add_argument('--label', dest='file_label', nargs='*', help='labels to display for multiple input files')
    parser.add_argument('--ylim', dest='ylim', nargs=2, metavar=('YMIN', 'YMAX'), type=float, help='Y limits for point plotting.')
    parser.add_argument('--tick-right', dest='tick_right', action='store_true', help='set tick and tick label to the right')
    parser.add_argument('-l','--lookup', dest='lookup_file', type=str, help='lookup table file')

    parser.add_argument('-n', dest='idx', metavar='NUM', type=int, help='Epoch/slice number for initial display.')
    parser.add_argument('--error', dest='error_file', help='txt file with error for each date.')

    # time info
    parser.add_argument('--start-date', dest='start_date', type=str, help='start date of displacement to display')
    parser.add_argument('--end-date', dest='end_date', type=str, help='end date of displacement to display')
    parser.add_argument('--exclude', '--ex', dest='ex_date_list', nargs='*', default=['exclude_date.txt'], help='Exclude date shown as gray.')
    parser.add_argument('--zf', '--zero-first', dest='zero_first', action='store_true', help='Set displacement at first acquisition to zero.')
    parser.add_argument('--off','--offset', dest='offset', type=float, help='Offset for each timeseries file.')

    parser.add_argument('--noverbose', dest='print_msg', action='store_false', help='Disable the verbose message printing.')

    # temporal model fitting
    parser.add_argument('--nomodel', '--nofit', dest='plot_model', action='store_false',
                        help='Do not plot the prediction of the time function (deformation model) fitting.')
    parser.add_argument('--plot-model-conf-int', '--plot-fit-conf-int', dest='plot_model_conf_int', action='store_true',
                        help='Plot the time function prediction confidence intervals.\n'
                             '[!-- Preliminary feature alert! --!]\n'
                             '[!-- This feature is NOT throughly checked. Read the code before use. Interpret at your own risk! --!]')

    parser = arg_group.add_timefunc_argument(parser)

    # pixel of interest
    pixel = parser.add_argument_group('Pixel Input')
    pixel.add_argument('--yx', type=int, metavar=('Y', 'X'), nargs=2, help='initial pixel to plot in Y/X coord')
    pixel.add_argument('--lalo', type=float, metavar=('LAT', 'LON'), nargs=2, help='initial pixel to plot in lat/lon coord')

    pixel.add_argument('--marker', type=str, default='o', help='marker style (default: %(default)s).')
    pixel.add_argument('--ms', '--markersize', dest='marker_size', type=float, default=6.0, help='marker size (default: %(default)s).')
    pixel.add_argument('--lw', '--linewidth', dest='linewidth', type=float, default=0, help='line width (default: %(default)s).')
    pixel.add_argument('--ew', '--edgewidth', dest='edge_width', type=float, default=1.0, help='Edge width for the error bar (default: %(default)s)')

    # other groups
    parser = arg_group.add_data_disp_argument(parser)
    parser = arg_group.add_dem_argument(parser)
    parser = arg_group.add_figure_argument(parser)
    parser = arg_group.add_gps_argument(parser)
    parser = arg_group.add_mask_argument(parser)
    parser = arg_group.add_map_argument(parser)
    parser = arg_group.add_memory_argument(parser)
    parser = arg_group.add_reference_argument(parser)
    parser = arg_group.add_save_argument(parser)
    parser = arg_group.add_subset_argument(parser)

    return parser


def tcoh_create_parser():
    parser = create_parser()
    parser.add_argument('--slc', dest='slc_file', nargs='+',
                        help='slc stack file to get pixel shp\n'
                             'i.e.: slcStack.h5 (MintPy)\n')
    parser.add_argument('-rw', '--rangeWindow', dest='range_win', type=str, default='19'
                        , help='SHP searching window size in range direction. -- Default : 19')
    parser.add_argument('-aw', '--azimuthWindow', dest='azimuth_win', type=str, default='9'
                        , help='SHP searching window size in azimuth direction. -- Default : 9')

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

    inps = ts2vel.init_exp_log_dicts(inps)

    return inps

def save_ts_data_and_plot(yx, d_ts, fig_coh, m_strs, inps):
    """Save TS data and plots into files."""
    y, x = yx
    vprint('save info on pixel ({}, {})'.format(y, x))

    # output file name
    if inps.outfile:
        inps.outfile_base, ext = os.path.splitext(inps.outfile[0])
        if ext != '.pdf':
            vprint(('Output file extension is fixed to .pdf,'
                    ' input extension {} is ignored.').format(ext))
    else:
        inps.outfile_base = 'y{}x{}'.format(y, x)

    # TXT - point time-series and time func param
    outName = '{}_ts.txt'.format(inps.outfile_base)
    header = 'time-series file = {}\n'.format(inps.file[0])
    header += '{}\n'.format(tsview.get_ts_title(y, x, inps.coord))
    header += 'reference pixel: y={}, x={}\n'.format(inps.ref_yx[0], inps.ref_yx[1]) if inps.ref_yx else ''
    header += 'reference date: {}\n'.format(inps.date_list[inps.ref_idx]) if inps.ref_idx else ''
    header += 'estimated time function parameters:\n'
    for m_str in m_strs:
        header += f'    {m_str}\n'
    header += 'unit: {}'.format(inps.disp_unit)

    # prepare data
    data = np.hstack((np.array(inps.date_list).reshape(-1, 1), d_ts.reshape(-1, 1)))

    # write
    np.savetxt(outName, data, fmt='%s', delimiter='\t', header=header)
    vprint('save displacement time-series to file: ' + outName)

    # Figure - point time-series
    outName = '{}_ts.pdf'.format(inps.outfile_base)
    inps.fig_pts.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save time-series plot to file: ' + outName)

    # Figure - map
    outName = '{}_{}.png'.format(inps.outfile_base, inps.date_list[inps.idx])
    inps.fig_img.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    vprint('save map plot to file: ' + outName)

    # Figure - coh
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

        self.StackObj = slcStack(self.slcStack)
        self.n_image, self.length, self.width = self.StackObj.get_size()
        #slc_file = h5py.File(self.slcStack, 'r')
        #self.rslc = slc_file['slc'][:, :, :]
        #slc_file.close()
        #self.length = self.rslc.shape[1]
        #self.width = self.rslc.shape[2]
        #self.n_image = self.rslc.shape[0]
        #self.num_slc = self.n_image

        return


    def plot(self):
        # read 3D time-series
        self.ts_data, self.mask = tsview.read_timeseries_data(self)[0:2]

        # Figure 1 - Cumulative Displacement Map
        if not self.figsize_img:
            self.figsize_img = pp.auto_figure_size(
                ds_shape=self.ts_data[0].shape[-2:],
                disp_cbar=True,
                disp_slider=True,
                print_msg=self.print_msg)
        self.fig_img = plt.figure(self.figname_img, figsize=self.figsize_img)

        # Figure 1 - Axes 1 - Displacement Map
        self.ax_img = self.fig_img.add_axes([0.125, 0.25, 0.75, 0.65])
        img_data = np.array(self.ts_data[0][self.idx, :, :])
        img_data[self.mask == 0] = np.nan
        self.plot_init_image(img_data)

        # Figure 1 - Axes 2 - Time Slider
        self.ax_tslider = self.fig_img.add_axes([0.125, 0.1, 0.75, 0.07])
        self.plot_init_time_slider(init_idx=self.idx, ref_idx=self.ref_idx)
        self.tslider.on_changed(self.update_time_slider)

        # Figure 2 - Time Series Displacement - Point
        self.fig_pts, self.ax_pts = plt.subplots(num=self.figname_pts, figsize=self.figsize_pts)
        if self.yx:
            d_ts, m_strs = self.plot_point_timeseries(self.yx)

        # Figure 3 - Temporal Coherence - Point
        self.fig_coh, self.ax_coh = plt.subplots(nrows=1, ncols=3)
        if self.yx:
            d_coh = self.plot_point_coh_matrix(self.yx)

        # save figures and data to files
        if self.save_fig:
            save_ts_data_and_plot(self.yx, d_ts, self.fig_coh, m_strs, self)

        # Output
        #if self.save_fig:
        #    save_ts_plot(self.yx, self.fig_img, self.fig_pts, d_ts, self.fig_coh, self)

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
        self.ax_coh[0].cla()
        self.ax_coh[1].cla()
        self.ax_coh[2].cla()
        #self.ax_coh[1, 1].cla()

        #import pdb; pdb.set_trace()
        sample_rows = np.arange(-((self.azimuth_window - 1) // 2), ((self.azimuth_window - 1) // 2) + 1, dtype=np.int32)
        reference_row = np.array([(self.azimuth_window - 1) // 2], dtype=np.int32)
        sample_cols = np.arange(-((self.range_window - 1) // 2), ((self.range_window - 1) // 2) + 1, dtype=np.int32)
        reference_col = np.array([(self.range_window - 1) // 2], dtype=np.int32)

        distance_threshold = ks_lut_cy(self.n_image, self.n_image, 0.01)
        box = [yx[1] - 50, yx[0] - 50, yx[1] + 50, yx[0] + 50]
        row1 = box[1]
        row2 = box[3]
        col1 = box[0]
        col2 = box[2]

        data = (yx[0] - row1, yx[1] - col1)

        patch_slc_images = self.StackObj.read(datasetName='slc', box=box, print_msg=False)

        default_mini_stack_size = 10
        total_num_mini_stacks = self.n_image // default_mini_stack_size

        shp = get_shp_row_col_c(data, patch_slc_images, sample_rows, sample_cols,
                                reference_row, reference_col, distance_threshold)

        num_shp = shp.shape[0]
        CCG = np.zeros((self.n_image, num_shp), dtype=np.complex64)
        for t in range(num_shp):
            CCG[:, t] = patch_slc_images[:, shp[t, 0], shp[t, 1]]

        coh_mat = iut.est_corr_py(CCG)

        vec_refined, squeezed_images, temp_quality = iut.sequential_phase_linking_py(CCG, b'sequential_EMI',
                                                                                 default_mini_stack_size,
                                                                                 total_num_mini_stacks)
        vec_refined = iut.datum_connect_py(squeezed_images, vec_refined, default_mini_stack_size)

        amp_refined = np.mean(np.abs(CCG), axis=1)

        vec_refined = amp_refined * np.exp(1j * np.angle(vec_refined))
        vec_refined[0] = amp_refined[0] + 0j

        temp_quality_full = gam_pta(np.angle(coh_mat), vec_refined)

        sample_rows = data[0] + sample_rows
        sample_rows[sample_rows < 0] = -1
        sample_rows[sample_rows >= self.length] = -1

        sample_cols = data[1] + sample_cols
        sample_cols[sample_cols < 0] = -1
        sample_cols[sample_cols >= self.width] = -1

        x, y = np.meshgrid(sample_cols.astype(int), sample_rows.astype(int), sparse=False)

        win = np.abs(patch_slc_images[:, y, x])

        cmap, norm = custom_cmap()
        im1 = self.ax_coh[0].imshow(np.abs(coh_mat), cmap='jet', norm=norm)
        self.ax_coh[0].set_title('Coherence')

        cmap, norm = custom_cmap(-np.pi, np.pi)
        im2 = self.ax_coh[1].imshow(np.angle(coh_mat), cmap='jet', norm=norm)
        self.ax_coh[1].set_title('Interferograms')

        im3 = self.ax_coh[2].imshow(np.mean(win[:, :, :], axis=0), cmap='summer')
        self.ax_coh[2].set_title('SHPs')

        self.ax_coh[2].plot(shp[:, 1]-sample_cols[0], shp[:, 0]-sample_rows[0], 'x', color='r')

        self.fig_coh.canvas.draw()

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
