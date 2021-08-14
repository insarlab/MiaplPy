#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='Baselines directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='Output text file')
    parser.add_argument('-t', '--temporalBaseline', dest='t_threshold', default=2, type=int,
                        help='Number of sequential interferograms to consider')
    parser.add_argument('-p', '--perpBaseline', dest='p_threshold', default=200, type=int,
                        help='Perpendicular baseline threshold')
    parser.add_argument('-d', '--date_list', dest='date_list', default=None, type=str,
                        help='Text file having existing SLC dates')
    parser.add_argument('--MinSpanTree', dest='min_span_tree', action='store_true',
                          help='Keep minimum spanning tree pairs')

    inps = parser.parse_args(args=iargs)
    return inps


def find_baselines(iargs=None):
    inps = cmd_line_parse(iargs)

    baselines, dates = get_baselines_dict(inps.baseline_dir)
    with open(inps.date_list, 'r') as f:
        date_list = f.readlines()
        date_list = [dd.split('\n')[0] for dd in date_list]

    for i, date in enumerate(dates):
        if not date in date_list:
            del dates[i]

    dates = np.sort(dates)

    q = np.zeros([len(dates), len(dates)])

    for i, ds in enumerate(dates):
        t = np.arange(i - inps.t_threshold, i + inps.t_threshold)
        t = t[t >= 0]
        t = t[t < len(dates)]
        if len(t) > 0:
            for m in range(t[0], t[-1] + 1):
                q[i, m] = np.abs(baselines[dates[i]] - baselines[dates[m]])
                q[m, i] = q[i, m]

        #if len(t) > 3:
        #    ss = np.where(q[i, :] == np.max(q[i, :]))[0]
        #    q[i,ss]=0

    q[q > inps.p_threshold] = 0

    if inps.min_span_tree:
        X = csr_matrix(q)
        Tcsr = minimum_spanning_tree(X)
        A = Tcsr.toarray()
    else:
        for i in range(len(dates)):
            if len(np.nonzero(q[i, :])[0]) <= 1:
                q[i, :] = 0
        A = np.triu(q)

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
    from datetime import datetime
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

    fig = plt.figure(figsize=(8, 4))

    for d in ifgdates:
        X = d.split(',')[0].split('_')
        x1 = datetime.strptime(X[0], '%Y%m%d')
        x2 = datetime.strptime(X[1], '%Y%m%d')

        Y = d.split('\n')[0].split(',')[1:3]
        y1 = float(Y[0])
        y2 = float(Y[1])
        plt.plot([x1, x2], [y1, y2], 'k*-')

    plt.xlabel('Time [years]')
    plt.ylabel('Perp Baseline [m]')

    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)
    ax.autoscale_view()

    fig.savefig(out_dir + '/unwrap_network.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    return


if __name__ == '__main__':
    find_baselines()


