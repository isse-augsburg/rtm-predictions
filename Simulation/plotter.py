import os
from functools import partial

from multiprocessing.pool import Pool
from pathlib import Path

import h5py
import time
# import matplotlib.pyplot as plt
from PIL import Image
# from matplotlib import cm
import numpy as np

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


def plot_image_and_label(label_path, img_path, i):
    print(i)
    label = Image.open(label_path)
    im = Image.open(img_path).convert('RGBA')
    label = Image.fromarray((np.asarray(label) / 3) * 255).convert('RGBA')
    Path('1on1').mkdir(exist_ok=True, parents=True)
    Image.blend(im, label, .6).save('1on1/im%05d.png' % i)


if __name__ == "__main__":
    for i in range(100, 812):
        plot_image_and_label(r'X:\s\t\stiebesi\data\RTM\Leoben\03_MessdatenOptPermeameter\03_MessdatenOptPermeameter\V01_20150313_085049\im%05d_label.png' % i,
            r'X:\s\t\stiebesi\data\RTM\Leoben\03_MessdatenOptPermeameter\03_MessdatenOptPermeameter\V01_20150313_085049\im%05d.png' % i, i)