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
import analizer


class SimCreator:
    def __init__(self):
        self.vebatch_exec = Path(r'C:\Program Files\ESI Group\Visual-Environment\14.5\Windows-x64\VEBatch.bat')
        self.initial_timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.sigma = 8e-12
        self.count = 1
        self.num_hosts = 2
        self.num_cpus = 8

        self.sim_step = 4
        self.max_sim_step = 1500
        self.max_injection_time = 800000

        self.solved_sims            = r'Y:\data\RTM\Lautern\1_solved_simulations\output'
        self.original_lperm         = Path(r'C:\Data\0_RTM_data\Data\Lautern\flawless_one_layer\k1_k2_equal_one_layer.lperm')
        self.vdb_origin             = Path(r'C:\Data\0_RTM_data\Data\Lautern\flawless_one_layer\flawless_one_layer.vdb')
        self.solver_input_folder    = Path(r'C:\Data\0_RTM_data\Data\output11\k1_sig%s\%s_%dp' % (str(self.sigma), self.initial_timestamp, self.count))
        self.slurm_scripts_folder   = Path(r'X:\s\t\stiebesi\slurm_scripts')

        self.slurm_scripts = []
        self.dirs_with_stems = []
        self.fn_vdb_writer = []
        self.unf_files_on_storage = []

    def create_folder_structure_and_perturbate_kN(self):
        print('Perturbating k1 ...')
        t0 = time.time()
        self.solver_input_folder.mkdir(parents=True, exist_ok=True)
        df = pandas.read_csv(self.original_lperm, sep=' ')
        keys = list(df.keys())
        with Pool() as p:
            self.dirs_with_stems = p.map(partial(self.perturbate_wrapper, keys, df), range(self.count))
        # self.dirs_with_stems =self.perturbate_wrapper(keys, df, 1)
        print(f'Adding noise and writing all files took {time.time() - t0} seconds.')

    def perturbate_wrapper(self, keys, df, count, mu=0):
        new_stem = str(count)
        dir = self.solver_input_folder / new_stem
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = f'{self.initial_timestamp}_{count}.lperm'
        fn = dir / filename
        if fn.exists():
            return fn

        iK1 = keys.index('K1')
        # iK2 = keys.index('K2')
        df.iloc[:, iK1] = df.iloc[:, iK1] + np.random.normal(mu, self.sigma, len(df))
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
        return Path(dir / f'{self.initial_timestamp}_{count}')

    def write_solver_input(self):
        py_script_str = resources.python2_script_X_vdbs % (self.vdb_origin, self.max_injection_time, self.sim_step)
        used_var_nums_in_script = 3 + 1 # starts with 1

        var_count = used_var_nums_in_script
        py_vdb_blocks = []

        for i, dir_stem in enumerate(self.dirs_with_stems):
            py_vdb_lines = [resources.import_lperm % (var_count, var_count, f'{dir_stem}.lperm', var_count),
                            resources.export_file % f'{dir_stem}.vdb']
            py_vdb_blocks.append(''.join(py_vdb_lines))

        self.fn_vdb_writer = os.path.join(self.solver_input_folder, 'vdb_writerpy2.py')
        py_vdb_for_one_call = ''.join(py_vdb_blocks)
        _str = py_script_str + py_vdb_for_one_call
        with open(self.fn_vdb_writer, 'w') as f:
            f.write(''.join(_str))

    def create_unfs_et_al(self):
        print('Writing .unf files ...')
        t0 = time.time()
        for i, e in enumerate(self.dirs_with_stems):
            fn = str(e) + 'g.unf'
            if os.path.exists(fn):
                continue
            fn2 = str(e) + '.vdb'
            call_make_rest = fr'''"{self.vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -nodisplay -imp "{fn2}" -datacast -exit'''
            args2 = shlex.split(call_make_rest)
            subprocess.call(args2, shell=True, stdout=subprocess.PIPE)
        print(f'Writing .unf files took {time.time()-t0}.')

    def create_vdbs(self):
        print(f'Writing .vdb files ...')
        t0 = time.time()
        call_make_vdb = fr''' "{self.vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -sessionrun "{self.fn_vdb_writer}" -nodisplay -exit'''
        args = shlex.split(call_make_vdb)
        subprocess.call(args, shell=True, stdout=subprocess.PIPE)
        print(f'Writing .vdbs took {time.time() - t0} seconds.')

    # Does not work
    def solve_simulations(self, selfpath):
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

    def write_slurm_scripts(self, timeout=15):
        print('Writing slurm script ...')
        calls = []
        line_before_unf = 'srun -t %s singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np %d' % (timeout, self.num_cpus)
        for e in self.unf_files_on_storage:
            call = '%s %s' % (line_before_unf, e)
            calls.append(call)

        if len(calls) > 1:
            delim = int(np.round(len(calls) / self.num_hosts))
        else:
            delim = 1
        sublist_calls = [calls[x:x + delim] for x in range(0, len(calls), delim)]
        for i in range(self.num_hosts):
            script_str = '%s\n%s' % (f'''#!/bin/sh
#SBATCH --partition=small-cpu
#SBATCH --mem=24000
#SBATCH --time=100:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task={self.num_cpus}
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%x.out
''', '\n'.join(sublist_calls[i]))
            n = f'0{i}_solve_pam_rtm_auto.sh'
            self.slurm_scripts.append(n)
            filename = self.slurm_scripts_folder / n
            with open(filename, 'w') as f:
                f.write(script_str)
            fileContents = open(filename,"r").read()
            f = open(filename,"w", newline="\n")
            f.write(fileContents)
            f.close()
            if len(calls) == 1:
                break

    def alter_dat_files(self):
        for i, e in enumerate(self.dirs_with_stems):
            fn = e.parent / (e.stem + 'p.dat')
            if fn.exists():
                with open(fn, 'r') as f:
                    lines = f.readlines()
                    lines = [f'NSTEP {self.max_sim_step}\n' if 'NSTEP' in x else x for x in lines]
                with open(fn, 'w') as f:
                    f.writelines(lines)

    def run(self):
        self.create_folder_structure_and_perturbate_kN()
        self.write_solver_input()
        self.create_vdbs()
        self.create_unfs_et_al()
        self.alter_dat_files()
        self.copy_simfiles_to_cluster()
        self.write_slurm_scripts()

    def copy_simfiles_to_cluster(self):
        f_endings = ['g.unf', 'ff.unf', '.elrnm', '.ndrnm', '.ptrnm', 'd.out', 'p.dat']
        for e in self.dirs_with_stems:
            p = Path(self.solved_sims + '/' + '/'.join(e.parts[-4:]))
            p.mkdir(parents=True, exist_ok=True)
            for end in f_endings:
                copy2(str(e) + end, str(p / p.stem) + end)
                if end == 'g.unf':
                    self.unf_files_on_storage.append(Path(str(p / p.stem) + end).as_posix().replace('Y:', '/cfs/share'))

    # Does not work
    def run_slurm(self):
        for e in self.slurm_scripts:
            call = f'ssh swt-clustermanager sbatch ~/slurm_scripts/{e}'
            c = shlex.split(call)
            p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE)
            std, err = p.communicate()
            print(std)
            print(err)


if __name__== "__main__":
    if os.environ['Write_Simulation'] == '1':
        sc = SimCreator()
        sc.run()
    # Execute slurm ...

    # Run an analysis over the new simulations
    elif os.environ['Analysis'] == '1':
        path = r'Y:\data\RTM\Lautern\1_solved_simulations\output'
        print_options = ['all', 'fails_only', 'success_only']
        analizer.analize_subdirs(path, print_options='all')