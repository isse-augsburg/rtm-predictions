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

import resources

# class Perturbator:
#     def __init__(self):
#         self.in_filename = in_filename
#         self.outfolder

def read_lperm_and_perturbate_kN(in_filename, out_folder, sigma=5e-15, mu=0, iteration=-1, count=1):
    t0 = time.time()
    root_dir = Path(in_filename).parent
    if not (root_dir / out_folder).exists():
        (root_dir / out_folder).mkdir()
    df = pandas.read_csv(in_filename, sep=' ')
    keys = list(df.keys())
    with Pool() as p:
        res = p.map(partial(perturbate_wrapper, root_dir, out_folder, iteration, sigma, mu, keys, df), range(count))
    print(f'Adding noise and writing all files took {time.time() - t0} seconds.')


def perturbate_wrapper(root_dir, out_folder, iteration, sigma, mu, keys, df, count):
    append = ''
    if iteration != -1:
        append = f'_{iteration}'

    new_stem = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_k1_pertubated_sigma{sigma:.3e}_mu{mu:.1f}{append}_{count}'
    filename = new_stem + '.lperm'
    fn = root_dir / out_folder / filename
    if fn.exists():
        return fn

    iK1 = keys.index('K1')
    # iK2 = keys.index('K2')
    df.iloc[:, iK1] = df.iloc[:, iK1] + np.random.normal(mu, sigma, len(df))
    # df.iloc[:, iK2] = df.iloc[:, iK2] + np.random.normal(mu, sigma, len(df))

    lperm_str = df.to_string(formatters={'\#Element_ID': '{:,i}'.format,
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
    # print(count)

def get_perturbated_lperms_paths(perturbated_lperms_folder):
    paths = []
    for root, dirs, files in os.walk(perturbated_lperms_folder, topdown=False):
        for name in files:
            p = Path(name)
            if p.suffix == '.lperm':
                paths.append(Path(root) / p)
        break
    return paths


def get_vdb_blocks_and_vdb_filenames_for_each_lperm_file(paths, solver_input_folder, var_count, copy_lperm):
    py_vdb_blocks = []
    fns_vdb = []
    for i, p in enumerate(paths):
        py_vdb_lines = []
        dest_dir = solver_input_folder / p.stem
        if not dest_dir.exists():
            dest_dir.mkdir()

        if copy_lperm and not (dest_dir / (p.stem + p.suffix)).exists():
            copy2(p, dest_dir)

        py_vdb_lines.append(resources.import_lperm % (var_count, var_count, p, var_count))

        fn_vdb = f"{solver_input_folder / p.stem / p.stem}.vdb"
        fns_vdb.append(fn_vdb)
        py_vdb_lines.append(resources.export_file % fn_vdb)

        py_vdb_blocks.append(''.join(py_vdb_lines))

    return py_vdb_blocks, fns_vdb


def write_solver_input(perturbated_lperms_folder, solver_input_folder, vdb_input, new_perturbated_lperms='',
                       step_size=2, max_injection_time=800000, copy_lperm=False):
    if new_perturbated_lperms == '':
        paths = get_perturbated_lperms_paths(perturbated_lperms_folder)
    else:
        paths = new_perturbated_lperms

    solver_input_folder = Path(solver_input_folder)
    if not solver_input_folder.exists():
        solver_input_folder.mkdir()

    py_script_str = resources.python2_script_X_vdbs % (vdb_input, max_injection_time, step_size)
    used_var_nums_in_script = 3 + 1 # starts with 1

    var_count = used_var_nums_in_script
    py_vdb_blocks, fns_vdb = get_vdb_blocks_and_vdb_filenames_for_each_lperm_file(paths, solver_input_folder, var_count, copy_lperm)

    fn_vdb_writer = solver_input_folder / f'vdb_writerpy2.py'
    py_vdb_for_one_call = ''.join(py_vdb_blocks)
    _str = py_script_str + py_vdb_for_one_call
    with open(fn_vdb_writer, 'w') as f:
        f.write(''.join(_str))

    return fns_vdb, fn_vdb_writer


def create_unfs_et_al(fns_vdb, vebatch_exec):
    print('Creating .unf files ...')
    t0 = time.time()
    for i, e in enumerate(fns_vdb):
        if Path(e[:-4] + 'g.unf').exists():
            continue

        call_make_rest = fr'''"{vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -nodisplay -imp "{e}" -datacast -exit'''
        args2 = shlex.split(call_make_rest)
        subprocess.call(args2, shell=True, stdout=subprocess.PIPE)
    print(f'Writing .unf files took {time.time()-t0}.')


def create_vdbs(fn_vdb_writer, vebatch_exec):
    print(f'Writing .vdb files ...')
    t0 = time.time()
    call_make_vdb = fr''' "{vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -sessionrun "{fn_vdb_writer}" -nodisplay -exit'''
    args = shlex.split(call_make_vdb)
    subprocess.call(args, shell=True, stdout=subprocess.PIPE)
    print(f'Writing .vdbs took {time.time() - t0} seconds.')


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


def write_slurm_script(input_dir, output_dir, num_hosts=1, num_cpus=16, timeout=12):
    calls = []
    line_before_unf = 'srun -t %s singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np %d' % (timeout, num_cpus)
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for fn in files:
            if fn[-5:] == "g.unf":
                p = Path(root)
                unf_path = (p / (p.stem + p.suffix)).as_posix().replace('Y:', '/cfs/share') + 'g.unf'
                call = '%s %s' % (line_before_unf, unf_path)
                calls.append(call)

    delim = int(np.round(len(calls) / num_hosts))
    sublist_calls = [calls[x:x + delim] for x in range(0, len(calls), delim)]
    for i in range(num_hosts):
        script_str = '%s\n%s' % (f'''#!/bin/sh
#SBATCH --partition=small-cpu
#SBATCH --mem=24000
#SBATCH --time=100:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%x.out
''', '\n'.join(sublist_calls[i]))
        filename = Path(output_dir) / f'0{i}_solve_pam_rtm_auto.sh'
        with open(filename, 'w') as f:
            f.write(script_str)

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

def analyse_subdirs(path, print_options='all'):
    for root, dirs, files in os.walk(path, topdown=True):
        for name in dirs:
            abspath = os.path.join(root, name)
            analyse_finished_run(abspath, print_options)
        break

def run_until_slurm_script():
    sigma = 1.01e-11
    count = 2
    initial_timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    solved_sims = r'Y:\data\RTM\Lautern\1_solved_simulations\output'

    original_lperm = r'C:\Data\0_RTM_data\Data\Lautern\flawless_one_layer\k1_k2_equal_one_layer.lperm'
    perturbated_lperm_folder = r'C:\Data\0_RTM_data\Data\Lautern\0_lperm_files\%s_%dp_k1_sig%s' % (initial_timestamp, count, str(sigma))
    solver_input_folder = r'C:\Data\0_RTM_data\Data\output'
    vdb_origin = r"C:\Data\0_RTM_data\Data\Lautern\flawless_one_layer\flawless_one_layer.vdb"
    vebatch_exec = r'C:\Program Files\ESI Group\Visual-Environment\14.5\Windows-x64\VEBatch.bat'
    sim_step = 4

    print('Perturbating k1 ...')
    read_lperm_and_perturbate_kN(
        original_lperm,
        perturbated_lperm_folder,
        sigma=sigma,
        count=count)

    fns_vdb, fn_vdb_writer = write_solver_input(perturbated_lperm_folder, solver_input_folder, vdb_origin, step_size=sim_step)

    create_vdbs(fn_vdb_writer, vebatch_exec)

    create_unfs_et_al(fns_vdb, vebatch_exec)

    copy_simfiles_to_cluster(fns_vdb, solved_sims)

    print('Writing slurm script ...')
    write_slurm_script(input_dir=solved_sims, output_dir=r'X:\s\t\stiebesi\slurm_scripts', num_hosts=2, num_cpus=8)


def copy_simfiles_to_cluster(fns_vdb, solved_sims):
    f_endings = ['g.unf', 'ff.unf', '.elrnm', '.ndrnm', '.ptrnm', 'd.out', 'p.dat']
    for e in fns_vdb:
        path = e[:-4]
        filename = path.split('\\')[-1:][0]
        if not os.path.exists(os.path.join(solved_sims, filename)):
            os.makedirs(os.path.join(solved_sims, filename))
        [copy2(path + x, os.path.join(solved_sims, filename, filename + x)) for x in f_endings]


if __name__== "__main__":
    # if os.environ['Write_Simulation'] == '1':
        # Write simulations
    run_until_slurm_script()
    # Execute slurm ...

    # Run an analysis over the new simulations
    # elif os.environ['Analysis'] == '1':
    #     path = r'Y:\data\RTM\Lautern\2000_auto_solver_inputs'
    #     print_options = ['all', 'fails_only', 'success_only']
    #     analyse_subdirs(path, print_options='all')