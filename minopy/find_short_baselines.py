#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
#from scipy.sparse import csr_matrix
#from scipy.sparse.csgraph import minimum_spanning_tree
from datetime import datetime
from scipy.spatial import Delaunay

def cmd_line_parse(iargs=None):
    parser = argparse.ArgumentParser(description='find the minimum number of connected good interferograms')
    parser.add_argument('-b', '--baselineDir', dest='baseline_dir', type=str, help='Baselines directory')
    parser.add_argument('-o', '--outFile', dest='out_file', type=str, default='./bestints.txt', help='Output text file')
    parser.add_argument('-t', '--temporalBaseline', dest='t_threshold', default=60, type=int,
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

    dates = []
    for date in dates0:
        if date in date_list:
            dates.append(date)

    dates = np.sort(dates)

    days = [(datetime.strptime(date, '%Y%m%d') - datetime.strptime(dates[0], '%Y%m%d')).days for date in dates]

    pairtr = []
    for i, date in enumerate(dates):
        pairtr.append([days[i], baselines[date]])

    pairtr = np.array(pairtr)
    tri = Delaunay(pairtr, incremental=False)

    q = np.zeros([len(dates), len(dates)])

    for trp in pairtr[tri.simplices]:
        x1 = int(trp[0][0])
        x2 = int(trp[1][0])
        x3 = int(trp[2][0])
        b1 = trp[0][1]
        b2 = trp[1][1]
        b3 = trp[2][1]
        if np.abs(x1 - x2) <= inps.t_threshold:
            q[days.index(x1), days.index(x2)] = np.abs(b1 - b2)
            q[days.index(x2), days.index(x1)] = np.abs(b1 - b2)
        if np.abs(x2 - x3) <= inps.t_threshold:
            q[days.index(x2), days.index(x3)] = np.abs(b2 - b3)
            q[days.index(x3), days.index(x2)] = np.abs(b2 - b3)
        if np.abs(x1 - x3) <= inps.t_threshold:
            q[days.index(x1), days.index(x3)] = np.abs(b1 - b3)
            q[days.index(x3), days.index(x1)] = np.abs(b1 - b3)

    q[q > inps.p_threshold] = 0

    #if inps.min_span_tree:
    #    X = csr_matrix(q)
    #    Tcsr = minimum_spanning_tree(X)
    #    A = Tcsr.toarray()
    #else:
    #    for i in range(len(dates)):
    #        if len(np.nonzero(q[i, :])[0]) <= 1:
    #            q[i, :] = 0
    #    A = np.triu(q)

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

    fig.savefig(out_dir + '/network.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)

    return


if __name__ == '__main__':
    find_baselines()


