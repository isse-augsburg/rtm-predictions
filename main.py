from functools import partial

import h5py
import shlex
import time
import pandas
import datetime
import os
import numpy as np
import subprocess
from shutil import copy2
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm

import resources


def read_lperm_and_perturbate_k1(in_filename, out_folder, sigma=5e-15, mu=0, iteration=-1):
    t0 = time.time()
    df = pandas.read_csv(in_filename, sep=' ')
    # For starters, let's just perturbate the K1 permeability; K3 is always 0
    # The use of the vectors for the principal directions are currently unknown
    keys = list(df.keys())
    iK1 = keys.index('K1')
    df.iloc[:, iK1] = df.iloc[:, iK1] + np.random.normal(mu, sigma, len(df))

    root_dir = Path(in_filename).parent
    append = ''
    if iteration != -1:
        append = f'_{iteration}'
    new_folder = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_k1_pertubated_sigma{sigma:.3e}_mu{mu:.1f}{append}'
    filename = new_folder + '.lperm'
    if not (root_dir / out_folder).exists():
        (root_dir / out_folder).mkdir()
    fn = root_dir / out_folder / filename
    if fn.exists():
        return fn
    lperm_str = df.to_string(formatters = {'\#Element_ID':'{:,i}'.format,
                                         # 'Thickness': '{:,.6f}'.format,
                                         # 'Fiber_Content': '{:,.6f}'.format,
                                         'K1': '{:,.4e}'.format,
                                         'K2': '{:,.4e}'.format,
                                         'K3': '{:,.4e}'.format
                                         # 'Perm_Vec1_x': '{,.6f}'.format,
                                         # 'Perm_Vec1_y': '{,.6f}'.format,
                                         # 'Perm_Vec1_z': '{,.6f}'.format,
                                         # 'Perm_Vec2_x': '{,.6f}'.format,
                                         # 'Perm_Vec2_y': '{,.6f}'.format,
                                         # 'Perm_Vec2_z': '{,.6f}'.format
                                         }, index=False)
    with open(fn, 'w') as f:
        f.write(lperm_str)
    print(f'Adding noise and writing file took {time.time()-t0} seconds.')
    return fn


def write_solver_input(pertubated_lperms_folder, solver_input_folder, vdb_input, new_perturbated_lperms=''):
    vebatch_exec = r'C:\Program Files\ESI Group\Visual-Environment\14.5\Windows-x64\VEBatch.bat'

    t0 = time.time()
    paths = []
    if new_perturbated_lperms == '':
        for root, dirs, files in os.walk(pertubated_lperms_folder, topdown=False):
            for name in files:
                p = Path(name)
                if p.suffix == '.lperm':
                    paths.append(Path(root) / p)
            break
    else:
        paths = new_perturbated_lperms

    solver_input_folder = Path(solver_input_folder)
    if not solver_input_folder.exists():
        solver_input_folder.mkdir()

    py_script_str = resources.python2_script_X_vdbs % vdb_input
    used_var_nums_in_script = 3 + 1 # starts with 1
    py_script_as_list = [py_script_str]

    fns_vdb = []
    for i, p in enumerate(paths):
        dest_dir = solver_input_folder / p.stem
        if not dest_dir.exists():
            dest_dir.mkdir()
        copy2(p, dest_dir)
        var_count = used_var_nums_in_script + i
        py_script_as_list.append(resources.import_lperm % (var_count, var_count, p, var_count))

        fn_vdb = f"{solver_input_folder / p.stem / p.stem}.vdb"
        fns_vdb.append(fn_vdb)
        py_script_as_list.append(resources.export_file % fn_vdb)

    fn_vdb_writer = solver_input_folder / 'vdb_writerpy2.py'
    with open(fn_vdb_writer, 'w') as f:
        f.write(''.join(py_script_as_list))

    t0 = time.time()
    print('Creating .vdb files ...')
    call_make_vdb = fr''' "{vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -sessionrun "{fn_vdb_writer}" -nodisplay -exit'''
    args = shlex.split(call_make_vdb)
    subprocess.call(args, shell=True, stdout=subprocess.PIPE)
    print(f'Took {time.time() - t0} seconds.')

    t2 = time.time()
    print('Creating .unf files ...')
    for e in fns_vdb:
        call_make_rest = fr'''"{vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -nodisplay -imp "{e}" -datacast -exit'''
        args2 = shlex.split(call_make_rest)
        subprocess.call(args2, shell=True, stdout=subprocess.PIPE)
    print(f'Took {time.time() - t2} seconds.')
    return fns_vdb


def solve_simulations(path):
    print('Unsave function: solve_simulations()\n Use slurm.')
    exit(-1)
    unfs = []
    finished = set()
    for root, dirs, files in os.walk(path, topdown=True):
        for fn in files:
            if fn[-5:] == "g.unf":
                unfs.append(os.path.join(root, fn))
            if fn[-6:] == '.erfh5':
                finished.add(root)
    solver = r'C:\Program Files\ESI Group\PAM-COMPOSITES\2019.0\RTMSolver\bin\pamcmxdmp.bat'
    for e in unfs:
        if os.path.split(e)[:-1][0] in finished:
            print('Already finished/Started:', os.path.split(e)[:-1][0])
            continue
        print('Starting:', os.path.split(e)[:-1][0])
        call_solve = rf'''"{solver}" -np 12 "{e}"'''
        args = shlex.split(call_solve)
        process = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)


def write_slurm_script(input_dir, output_dir):
    calls = []
    line_before_unf = 'singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np 16'
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for fn in files:
            if fn[-5:] == "g.unf":
                p = Path(root)
                unf_path = (p / (p.stem + '.0')).as_posix().replace('Y:', '/cfs/share') + 'g.unf'
                call = '%s %s' % (line_before_unf, unf_path)
                calls.append(call)
    script_str = '%s\n%s'%(resources.sbatch_script_header, '\n'.join(calls))

    with open((Path(output_dir) / '4_solve_pam_rtm_auto.sh'), 'w') as f:
        f.write(script_str)

    filename = Path(output_dir) / '4_solve_pam_rtm_auto.sh'
    fileContents = open(filename,"r").read()
    f = open(filename,"w", newline="\n")
    f.write(fileContents)
    f.close()

def analyse_finished_run(path, print_options='all'):
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


def analyse_subdirs(path, print_options='all'):
    for root, dirs, files in os.walk(path, topdown=True):
        for name in dirs:
            abspath = os.path.join(root, name)
            analyse_finished_run(abspath, print_options)
        break

def run_until_slurm_script():
    perturbated_lperms = r'Y:\data\RTM\Lautern\100_perturbated_lperms_sigma1.11e-11'
    print('Perturbating k1 ...')
    for i in range(20):
        p = read_lperm_and_perturbate_k1(
            r'C:\Users\stiebesi\Documents\0_RTM_data\Data\Lautern\flawless_one_layer\k1_k2_equal_one_layer.lperm',
                perturbated_lperms,
                sigma=1.11e-11)

    solver_input_folder = r'Y:\data\RTM\Lautern\2000_auto_solver_inputs'
    vdb_origin = r"C:\Users\stiebesi\Documents\0_RTM_data\Data\Lautern\flawless_one_layer\flawless_one_layer.vdb"
    print('Writing .vdbs ...')
    vdbs = write_solver_input(perturbated_lperms, solver_input_folder, vdb_origin)
    print('Writing slurm script.')
    write_slurm_script(input_dir=solver_input_folder, output_dir=r'X:\s\t\stiebesi\slurm_scripts')


if __name__== "__main__":
    # Write simulations
    run_until_slurm_script()
    # Execute slurm ...

    # Run an analysis over the new simulations
    path = r'Y:\data\RTM\Lautern\2000_auto_solver_inputs'
    print_options = ['all', 'fails_only', 'success_only']
    analyse_subdirs(path, print_options='all')