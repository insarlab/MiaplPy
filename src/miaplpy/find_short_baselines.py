#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
from datetime import datetime
from scipy.spatial import Delaunay

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='Baselines directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='Output text file')
    parser.add_argument('-r', '--baseline_ratio', dest='baseline_ratio', default=1, type=float,
                        help='Ratio between temporal and perpendicular baseline (default = 1)')
    parser.add_argument('-t', '--temporalBaseline', dest='t_threshold', default=120, type=int,
                        help='Temporal baseline threshold')
    parser.add_argument('-p', '--perpBaseline', dest='p_threshold', default=200, type=int,
                        help='Perpendicular baseline threshold')
    parser.add_argument('-d', '--date_list', dest='date_list', default=None, type=str,
                        help='Text file having existing SLC dates')
    #parser.add_argument('--MinSpanTree', dest='min_span_tree', action='store_true',
    #                      help='Keep minimum spanning tree pairs')

    inps = parser.parse_args(args=iargs)
    return inps


def find_baselines(iargs=None):
    inps = cmd_line_parse(iargs)

    baselines, dates0 = get_baselines_dict(inps.baseline_dir)
    with open(inps.date_list, 'r') as f:
        date_list = f.readlines()
        date_list = [dd.split('\n')[0] for dd in date_list]

    min_baselines = min(baselines.values())
    max_baselines = max(baselines.values())

    dates = []
    for date in dates0:
        if date in date_list:
            dates.append(date)

    dates = np.sort(dates)

    days = [(datetime.strptime(date, '%Y%m%d') - datetime.strptime(dates[0], '%Y%m%d')).days for date in dates]

    temp2perp_scale = np.abs((max_baselines - min_baselines) / (np.nanmin(np.array(days)) - np.nanmax(np.array(days))))
    days = [tbase * temp2perp_scale for tbase in days]

    inps.t_threshold *= temp2perp_scale
    multplier = np.sqrt(inps.baseline_ratio)
    days = [x / multplier for x in days]

    pairtr = []
    for i, date in enumerate(dates):
        pairtr.append([days[i], baselines[date] * multplier])

    pairtr = np.array(pairtr)
    tri = Delaunay(pairtr, incremental=False)

    qm = np.zeros([len(dates), len(dates)])

    for trp in pairtr[tri.simplices]:
        x1 = trp[0][0]
        x2 = trp[1][0]
        x3 = trp[2][0]
        b1 = trp[0][1]
        b2 = trp[1][1]
        b3 = trp[2][1]
        if np.abs(x1 - x2) <= inps.t_threshold:
            qm[days.index(x1), days.index(x2)] = np.abs(b1 - b2)
            qm[days.index(x2), days.index(x1)] = np.abs(b1 - b2)
        if np.abs(x2 - x3) <= inps.t_threshold:
            qm[days.index(x2), days.index(x3)] = np.abs(b2 - b3)
            qm[days.index(x3), days.index(x2)] = np.abs(b2 - b3)
        if np.abs(x1 - x3) <= inps.t_threshold:
            qm[days.index(x1), days.index(x3)] = np.abs(b1 - b3)
            qm[days.index(x3), days.index(x1)] = np.abs(b1 - b3)

    qm[qm > inps.p_threshold] = 0

    #if inps.min_span_tree:
    #    X = csr_matrix(qm)
    #    Tcsr = minimum_spanning_tree(X)
    #    A = Tcsr.toarray()
    #else:
    #    for i in range(len(dates)):
    #        if len(np.nonzero(qm[i, :])[0]) <= 1:
    #            qm[i, :] = 0
    #    A = np.triu(qm)

    for i in range(len(dates)):
        if len(np.nonzero(qm[i, :])[0]) <= 1:
            qm[i, :] = 0
    A = np.triu(qm)

    ind1, ind2 = np.where(A > 0)
    ifgdates = ['{}_{}\n'.format(dates[g], dates[h]) for g, h in zip(ind1, ind2)]

    with open(inps.out_file, 'w') as f:
        f.writelines(ifgdates)

    plot_baselines(ind1=ind1, ind2=ind2, dates=dates, baselines=baselines,
                   out_dir=os.path.dirname(inps.out_file))

    return


def get_baselines_dict(baseline_dir):

    bf = os.listdir(baseline_dir)
    if not bf[0].endswith('txt'):
        bf2 = ['{}/{}.txt'.format(d, d) for d in bf]
    else:
        bf2 = bf

    baselines = {}
    reference = bf[0].split('_')[0]
    baselines[reference] = 0
    dates = [x.split('_')[1].split('.')[0] for x in bf]
    dates.append(reference)

    for d in bf2:
        secondary = d.split('.txt')[0].split('_')[-1]
        with open(os.path.join(baseline_dir, d), 'r') as f:
            lines = f.readlines()
            if len(lines) != 0:
                if 'Bperp (average):' in lines[1]:
                    baseline = float(lines[1].split('Bperp (average):')[1])
                else:
                    baseline = float(lines[1].split('PERP_BASELINE_TOP')[1])
                # baselines.append(baseline)
                baselines[secondary] = baseline
    return baselines, dates


def plot_baselines(ind1, ind2, dates=None, baselines=None, out_dir=None, baseline_dir=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    years_fmt = mdates.DateFormatter('%Y')

    if not baseline_dir is None and baselines is None:
        baselines = get_baselines_dict(baseline_dir)[0]

    dates = np.sort(dates)

    ifgdates = ['{}_{}, {}, {}, {}\n'.format(dates[g], dates[h], str(baselines[dates[g]]),
                                             str(baselines[dates[h]]), str(baselines[dates[g]] - baselines[dates[h]]))
                    for g, h in zip(ind1, ind2)]

    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(8, 4))

    for d in ifgdates:
        X = d.split(',')[0].split('_')
        x1 = datetime.strptime(X[0], '%Y%m%d')
        x2 = datetime.strptime(X[1], '%Y%m%d')

        Y = d.split('\n')[0].split(',')[1:3]
        y1 = float(Y[0])
        y2 = float(Y[1])
        plt.plot([x1, x2], [y1, y2], 'ko-', markersize=10)

    plt.xlabel('Time [years]')
    plt.ylabel('Perp Baseline [m]')

    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    fig.savefig(out_dir + '/network.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)

    return

def find_short_pbaseline_pair(baselines, date_list, ministack_size, last_index):

    second_index = np.arange(last_index - ministack_size + 1, last_index)
    diff_bselines = [np.abs(baselines[date_list[last_index - ministack_size - 2]] - baselines[date_list[i]]) for i in second_index]
    min_ind = np.min(diff_bselines)
    pair = (date_list[last_index - ministack_size - 2], date_list[second_index[diff_bselines.index(min_ind)]])
    return pair


def find_mini_stacks(date_list, baseline_dir, month=6):
    pairs = []
    dates = [datetime.strptime(date_str, '%Y%m%d') for date_str in date_list]
    bperp = get_baselines_dict(baseline_dir)[0]
    years = np.array([x.year for x in dates])
    u, indices_first = np.unique(years, return_index=True)
    f_ind = indices_first
    l_ind = np.zeros(indices_first.shape, dtype=int)
    l_ind[0:-1] = np.array(f_ind[1::]).astype(int)
    l_ind[-1] = len(dates)
    ref_inds = []
    for i in range(len(f_ind)):
        months = np.array([x.month for x in dates[f_ind[i]:l_ind[i]]])
        u, indices = np.unique(months, return_index=True)
        ind = np.where(u == month)[0]
        if len(ind) == 0:
            ind = int(len(u)//2)
        else:
            ind = ind[0]
        ref_ind = indices[ind] + f_ind[i]
        ref_inds.append(ref_ind)

        for k in range(f_ind[i], l_ind[i]):
            if not ref_ind == k:
                pairs.append((date_list[ref_ind], date_list[k]))
        ministack_size = l_ind[i] - f_ind[i]
        if i > 0:
            pairs.append(find_short_pbaseline_pair(bperp, date_list, ministack_size, l_ind[i]))

    for i in range(len(ref_inds)-1):
        pairs.append((date_list[ref_inds[i]], date_list[ref_inds[i+1]]))
        pairs.append((date_list[l_ind[i]-1], date_list[l_ind[i]]))
    return pairs

def find_one_year_interferograms(date_list):
    dates = np.array([datetime.datetime.strptime(date, '%Y%m%d') for date in date_list])

    ifg_ind = []
    for i, date in enumerate(dates):
        range_1 = date + datetime.timedelta(days=365) - datetime.timedelta(days=5)
        range_2 = date + datetime.timedelta(days=365) + datetime.timedelta(days=5)
        index = np.where((dates >= range_1) * (dates <= range_2))[0]
        if len(index) >= 1:
            date_diff = list(dates[index] - (date + datetime.timedelta(days=365)))
            ind = date_diff.index(np.nanmin(date_diff))
            ind_date = index[ind]
            date2 = date_list[ind_date]
            ifg_ind.append((date_list[i], date2))

    return ifg_ind

if __name__ == '__main__':
    find_baselines()


