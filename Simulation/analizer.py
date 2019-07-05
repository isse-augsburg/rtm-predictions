import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def analize_subdirs(path, print_options='all', suffix='erfh5'):
    files = Path(path).glob(f'**/*.{suffix}')
    good = 0
    bad = 0
    for f in files:
        if analize_finished_run(f, print_options, suffix):
            good += 1
        else:
            bad += 1
        print(good, bad)


def analize_finished_run(path, print_options='all', suffix='erfh5'):
    f = h5py.File(path, 'r')
    success = False
    if suffix == 'erfh5':
        all_states = f['post']['singlestate']

        # How long did it run, how many steps
        last_state_str = list(all_states.keys())[-1:][0]
        last_state_int = int(last_state_str.split('state')[-1:][0])

        # How much did it finish?
        # Sometimes, the last state does not have a FILLING_FACTOR, so we step reversed to the last state that has it
        filled = 0
        for i in reversed(range(len(all_states.keys()))):
            if f['post']['singlestate'][list(all_states.keys())[i]]['entityresults']['NODE'
            ].keys().__contains__('FILLING_FACTOR'):
                if not f['post']['singlestate'][list(all_states.keys())[i]]['entityresults']['NODE']['FILLING_FACTOR'][
                                        'ZONE1_set1']['erfblock'].__contains__('res'):
                    filled = 0
                    break
                filling_factors_at_certain_times = \
                f['post']['singlestate'][list(all_states.keys())[i]]['entityresults']['NODE']['FILLING_FACTOR'][
                    'ZONE1_set1']['erfblock']['res'][()].flatten()
                all_nodes = filling_factors_at_certain_times.shape[0]
                filled = np.sum(filling_factors_at_certain_times) / all_nodes
                break

        if filled == 1:
            success = True
        if print_options == 'all':
            print_result_line(path, success, filled, last_state_int)
        elif print_options == 'success_only' and success:
            print_result_line(path, success, filled, last_state_int)
        elif print_options == 'fails_only' and not success:
            print_result_line(path, success, filled, last_state_int)
        elif print_options == 'none':
            pass
    elif suffix == 'hdf5':
        shape_types = f['perturbation_factors']['Shapes'].keys()
        shape_num = 0
        for s in shape_types:
            shape_num += f['perturbation_factors']['Shapes'][s]['Num'][()]
        if not (path.parent / 'img_cache').exists():
            return False
        im = Image.open(path.parent / 'img_cache' / 'fiber_fraction.png')
        print(path.parent, shape_num)
        im.show()
        im.close()
    return success


def print_result_line(path, success, filled, last_state_int):
    sigma = 0
    mu = 0

    # for e in str(path).split('_'):
    #     if 'sigma' in e:
    #         _, ssigma = e.split('sigma')
    #         sigma = ssigma.replace('.',',')
    #     if 'mu' in e:
    #         _, smu = e.split('mu')
    #         mu = smu.replace('.',',')
    sfilled = f'{filled:.5f}'.replace('.', ',')

    p = Path(path)
    if (p.parent / 'img_cache').exists():
        i = 0
        for c in (p.parent / 'img_cache').iterdir():
            i += 1
        # p1 = [x for x in p.glob('**/*.ERFH5')][0]
        print(f'{path}\t{success}\t{sfilled}\t{last_state_int} steps\t{i} imgs')
    else:
        print(f'{path}\t{success}\t{sfilled}\t{last_state_int} steps')

if __name__ == "__main__":
    path = r'Y:\data\RTM\Lautern\output\with_shapes\2019-06-07_13-48-28_3500p'
    print_options = ['all', 'fails_only', 'success_only']
    analize_subdirs(path, print_options='none', suffix='erfh5')