#!/usr/bin/env python3
# Author: Sara Mirzaee

import os
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
#import matplotlib.pyplot as plt

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

    bf = os.listdir(inps.baseline_dir)
    if not bf[0].endswith('txt'):
        bf2 = ['{}/{}.txt'.format(d, d) for d in bf]
    else:
        bf2 = bf
        
    baselines = []
    for d in bf2:
        with open(os.path.join(inps.baseline_dir, d), 'r') as f:
            lines = f.readlines()
            if len(lines) != 0:
                if 'Bperp (average):' in lines[1]:
                    baseline = lines[1].split('Bperp (average):')[1]
                else:
                    baseline = lines[1].split('PERP_BASELINE_TOP')[1]
                baselines.append(baseline)

    baselines = [float(x.split('\n')[0].strip()) for x in baselines]
    reference = bf[0].split('_')[0]
    dates = [x.split('_')[1].split('.')[0] for x in bf]
    dates.append(reference)
    baselines.append(0)
    if not inps.date_list is None:
        with open(inps.date_list, 'r') as fr:
            date_list = fr.readlines()
        date_list = [x.split('\n')[0] for x in date_list]
        for i, date in enumerate(dates):
            if date not in date_list:
                del dates[i]
                del baselines[i]

    db_tuples = [(x, y) for x, y in zip(dates, baselines)]
    db_tuples.sort()

    dates = [x[0] for x in db_tuples]
    baselines = [x[1] for x in db_tuples]

    q = np.zeros([len(dates), len(dates)])

    for i, ds in enumerate(dates):
        t = np.arange(i - inps.t_threshold, i + inps.t_threshold)
        t = t[t >= 0]
        t = t[t < len(dates)]
        if len(t) > 0:
            for m in range(t[0], t[-1] + 1):
                q[i, m] = np.abs(baselines[i] - baselines[m])
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
    intdates = ['{}_{}\n'.format(dates[g], dates[h]) for g, h in zip(ind1, ind2)]
    intdates_test = ['{}_{}, {}, {}, {}\n'.format(dates[g], dates[h], str(baselines[g]),
                                             str(baselines[h]), str(baselines[g] - baselines[h]))
                for g, h in zip(ind1, ind2)]

    with open(inps.out_file, 'w') as f:
        f.writelines(intdates)

    plot_baselines(intdates_test, os.path.join(os.path.dirname(inps.out_file)))

    return


def plot_baselines(ifgdates, out_dir):
    import matplotlib.pyplot as plt
    from datetime import datetime

    fig = plt.figure(figsize=(8, 4))

    dates = [x.split(',')[0].split('_') for x in ifgdates]
    baselines = [x.split('\n')[0].split(',')[1:3] for x in ifgdates]

    for d, b in zip(dates, baselines):
        x1 = datetime.strptime(d[0], '%Y%m%d')
        x2 = datetime.strptime(d[1], '%Y%m%d')
        y1 = float(b[0])
        y2 = float(b[1])
        plt.plot([x1, x2], [y1, y2], '*-')

    fig.savefig(out_dir + '/unwrap_network.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

    return

if __name__ == '__main__':
    find_baselines()


