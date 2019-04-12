import os
from pathlib import Path

import h5py
import numpy as np


def analize_subdirs(path, print_options='all'):
    for root, dirs, files in os.walk(path, topdown=True):
        for name in dirs:
            abspath = os.path.join(root, name)
            analize_finished_run(abspath, print_options)
        break

def analize_finished_run(path, print_options='all'):
    filename = ''
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            abspath = os.path.join(root, name)
            if abspath.split('.')[-1:][0] == 'erfh5':
                filename = abspath
                break
        if filename == '':
            # print('ERFH5 file not found!')
            return
    f = h5py.File(filename, 'r')
    all_states = f['post']['singlestate']

    # How long did it run, how many steps
    last_state_str = list(all_states.keys())[-1:][0]
    last_state_int = int(last_state_str.split('state')[-1:][0])

    # How much did it finish?
    # Sometimes, the last state does not have a FILLING_FACTOR, so we step reversed to the last state that has it
    filled = 0
    for i in reversed(range(len(all_states.keys()))):
        if f['post']['singlestate'][list(all_states.keys())[i]]['entityresults']['NODE'].keys().__contains__('FILLING_FACTOR'):
            filling_factors_at_certain_times = f['post']['singlestate'][list(all_states.keys())[i]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()].flatten()
            all_nodes = filling_factors_at_certain_times.shape[0]
            filled = np.sum(filling_factors_at_certain_times) / all_nodes
            break

    success = False
    if filled == 1:
        success = True
    if print_options == 'all':
        print_result_line(path, success, filled, last_state_int)
    if print_options == 'success_only' and success:
        print_result_line(path, success, filled, last_state_int)
    if print_options == 'fails_only' and not success:
        print_result_line(path, success, filled, last_state_int)


def print_result_line(path, success, filled, last_state_int):
    sigma = 0
    mu = 0

    for e in path.split('_'):
        if 'sigma' in e:
            _, ssigma = e.split('sigma')
            sigma = ssigma.replace('.',',')
        if 'mu' in e:
            _, smu = e.split('mu')
            mu = smu.replace('.',',')
    sfilled = f'{filled:.5f}'.replace('.', ',')

    p = Path(path)
    p1 = str(p / p.stem) + '.0_RESULT.erfh5'
    print(f'{p1}\t{sigma}\t{success}\t{sfilled}\t{last_state_int} steps')