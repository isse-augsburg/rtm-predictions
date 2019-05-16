import os
from functools import partial

from multiprocessing.pool import Pool

import h5py
import time
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_filling_from_erfh5(filename):
    f = h5py.File(filename, 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _coords = coord_as_np_array[:, :-1]
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = [f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()] for state in all_states]
    states_as_list = [x[-5:] for x in list(all_states.keys())]
    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    states_and_fillings = [(i, j) for i, j in zip(states_as_list, flat_fillings)]

    t0 = time.time()
    with Pool(6) as p:
        res = p.map(partial(plot_wrapper, coords= _coords), states_and_fillings)
    print('Done after', time.time() -t0)


def plot_wrapper(states_and_filling, coords):
    filename = r'C:\Users\stiebesi\code\datamuddler\plots\lautern_flawless_hd\%s.png' % str(states_and_filling[0])
    if os.path.exists(filename):
        return False
    fig = plt.gcf()
    fig.set_size_inches(18.5, 18.5)
    areas = len(states_and_filling[1]) * [3]
    norm = cm.colors.Normalize(vmax=states_and_filling[1].max(), vmin=states_and_filling[1].min())
    plt.scatter(coords[:, 0], coords[:, 1], c=states_and_filling[1], s=areas, norm=norm)
    fig.savefig(filename)
    return True


def plot_wrapper_simple(coords):
    print('Start test')
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.savefig(r'C:\Users\stiebesi\code\datamuddler\plots\lautern\test.png')
    print('Done plotting')
    return True