import logging
import shlex
import socket
import subprocess
from enum import Enum
from functools import partial
from multiprocessing import Pool
from shutil import copy2

import h5py
import pandas

from Pipeline.data_loaders_IMG import draw_polygon_map
from Simulation import resources
from Simulation.Shapes import Shaper, TargetSimulation
from Simulation.h5writer import create_h5, write_dict_to_Hdf5
from Simulation.helper_functions import *


class OutputFrequencyType(Enum):
    Step = "Step", 0
    FillPercentage = "% Fill", 1
    Time = "Time", 2


def create_vdbs_from_parent_folder(parent_folder):
    s = SimCreator()
    p = Path(parent_folder)
    s.dirs_with_stems = [x.parent / x.stem for x in p.glob('**/*.lperm')]
    s.create_vdbs()


def create_unfs_from_parent_folder(parent_folder, folder_on_storage=r'Y:\data\RTM\Lautern\output'):
    s = SimCreator()
    p = Path(parent_folder)
    s.dirs_with_stems = [x.parent / x.stem for x in p.glob('**/*.lperm')]
    s.create_unfs_et_al()
    s.alter_dat_files()
    s.solved_sims = folder_on_storage
    s.copy_simfiles_to_cluster()
    s.write_slurm_scripts()


class SimCreator:
    def __init__(self, perturbation_factors=None, initial_timestamp='', n_in_batch=10,
                 batch_num=0, run_on_cluster=True, overall_count=10,
                 data_path=Path(r'X:\s\t\stiebesi\data\RTM\Lautern')):
        self.overall_count = overall_count
        self.batch_num = batch_num
        self.initial_timestamp = initial_timestamp
        self.start_index = batch_num * n_in_batch
        self.run_on_cluster = run_on_cluster
        free_space = shutil.disk_usage(r'C:\\').free // (1024 ** 3)
        estimated_used_space = n_in_batch * 0.55 + 8

        self.init_logger()

        self.logger.info(f'Going to use {estimated_used_space} of {free_space} GB.')
        if not run_on_cluster and estimated_used_space > free_space:
            print('Not enough free space on device. Finishing ...')
            self.run_on_cluster = True
            exit()
        self.output_frequency_type = OutputFrequencyType.Time
        self.save_to_h5_data = {'output_frequency_type': self.output_frequency_type.value[1],
                                'perturbation_factors': perturbation_factors}
        self.perturbation_factors_str = 'with_shapes'
        # f"with_{perturbation_factors['Shapes']['Rectangles']['Num']}_Rect_{perturbation_factors['Shapes']['Circles']['Num']}_Circ"
        # + re.sub('[{\',:}]', '', str(perturbation_factors)).replace(' ', '_')

        self.n_in_batch = n_in_batch
        self.visual_version = '15.0'
        self.logger.info(f'Using Visual Env {self.visual_version}.')
        if os.name == 'nt':
            self.vebatch_exec = Path(r'C:\Program Files\ESI Group\Visual-Environment\%s\Windows-x64\VEBatch.bat' % self.visual_version)
            str_sip = str(data_path / Path('output/%s/%s_%dp'))
            if self.run_on_cluster:
                disk = r'Y:'
            elif socket.gethostname() == 'PC610-74-virtuos':
                disk = r'D:\Data'
            else:
                disk = r'C:\Data'
            str_sip = str_sip.replace(r'X:\s\t\stiebesi\data', disk)
            self.slurm_scripts_folder = Path("")
            # self.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\slurm_scripts\%d_batch' % batch_num)
        else:
            # FIXME no support for Leoben data
            print('ERROR: Linux currently not working.')
            exit()
            self.vebatch_exec = '/usr/local/esi/Visual-Environment/14.5/Linux_x86_64_2.27/VEBatch.sh'
            data_path = Path('/run/user/1000/gvfs/smb-share:server=swt-clusterstorage,share=share/data/RTM/Lautern')
            str_sip = r'/home/stieber/data/output/%s/%s_%dp'
            self.slurm_scripts_folder = Path(r'/run/user/1000/gvfs/smb-share:server=swt-clusterstorage,share=home/s/t/stiebesi/slurm_scripts/%d_batch' % batch_num)

        self.solver_input_folder = Path(str_sip % (self.perturbation_factors_str, self.initial_timestamp, self.overall_count))
        self.slurm_scripts_folder.mkdir(parents=True, exist_ok=True)
        self.solved_sims            = data_path / 'output'
        self.sim_files_data_heap    = data_path / 'simulation_files'
        if perturbation_factors is None:
            self.perturbation_factors = {}
        else:
            self.perturbation_factors = perturbation_factors

        sources_path = Path('\\'.join(['Y:'] + list(data_path.parts)[4:])) / 'sources'
        self.original_lperm         = sources_path / 'origin.lperm'
        self.vdb_origin             = sources_path / 'origin.vdb'
        self.reference_erfh5        = sources_path / 'origin.erfh5'

        if 'Lautern' in str(sources_path):
            target = TargetSimulation.Lautern
        elif 'Leoben' in str(sources_path):
            target = TargetSimulation.Leoben
        self.Shaper = Shaper(self.reference_erfh5, self.perturbation_factors, target=target)

        self.num_big_hosts      = 9
        self.num_small_hosts    = 1

        self.output_frequency   = 0.5
        self.max_sim_step       = 3000
        self.max_runtime_slurm  = 15
        self.max_injection_time_pam_rtm = 800000

        self.slurm_scripts = []
        self.dirs_with_stems = []
        self.fn_vdb_writer = []
        self.unf_files_on_storage = []

    def init_logger(self):
        self.logger = logging.getLogger('SimCreator')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('debug.log')
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def create_folder_structure_and_perturbate_kN(self):
        self.logger.info(f'Perturbating {self.perturbation_factors_str} ...')
        t0 = time.time()
        self.solver_input_folder.mkdir(parents=True, exist_ok=True)
        df = pandas.read_csv(self.original_lperm, sep=' ')
        with Pool(1) as p:
            self.dirs_with_stems = p.map(partial(self.perturbate_wrapper, df),
                                         range(self.start_index, self.start_index + self.n_in_batch))
        self.logger.info(f'Adding noise and writing all files took {(time.time() - t0)/60:.1f} minutes.')

    def perturbate_wrapper(self, df, count, mu=0):
        df = df.copy()
        new_stem = str(count)
        dir = self.solver_input_folder / new_stem
        dir.mkdir(parents=True, exist_ok=True)
        filename = f'{self.initial_timestamp}_{count}.lperm'
        fn = dir / filename
        if fn.exists():
            return fn

        iK = df.keys().get_loc('Fiber_Content')
        # Perturbate FVC everywhere
        df.iloc[:, iK] = df.iloc[:, iK] + np.random.normal(mu, self.perturbation_factors['General_Sigma'], len(df))
        self.save_to_h5_data = self.Shaper.apply_shapes(df, self.save_to_h5_data)
        # Apply function that gets k1 and k2 from FVC
        # FIXME should be different values for K1 and K", currently just the same
        df['K1'] = fvc_to_k1(df['Fiber_Content'])
        df['K2'] = fvc_to_k1(df['Fiber_Content'])

        formatters = \
        {'\#Element_ID': '{:,i}'.format,
         # 'Thickness': '{:,.6f}'.format,
         'Fiber_Content': '{:,.6f}'.format,
         'K1': '{:,.4e}'.format,
         'K2': '{:,.4e}'.format,
         'K3': '{:,.4e}'.format
         # 'Perm_Vec1_x': '{,.6f}'.format,
         # 'Perm_Vec1_y': '{,.6f}'.format,
         # 'Perm_Vec1_z': '{,.6f}'.format,
         # 'Perm_Vec2_x': '{,.6f}'.format,
         # 'Perm_Vec2_y': '{,.6f}'.format,
         # 'Perm_Vec2_z': '{,.6f}'.format
         }
        debug = False
        if debug:
            f = h5py.File('Debugging/2019-05-17_16-45-57_0_RESULT.erfh5', 'r')
            coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
            _all_coords = coord_as_np_array[:, :-1]
            scaled_coords = (_all_coords + 23.25) * 10

            triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
            triangle_coords = triangle_coords[:, :-1] - 1
            values_for_triangles = df['Fiber_Content']
            im = draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords)
            im.show()
            print('--')

        lperm_str = df.to_string(formatters=formatters, index=False)
        with open(fn, 'w') as f:
            f.write(lperm_str)
        fn2 = f'{self.initial_timestamp}_{count}_meta_data'
        f = create_h5(str(dir / fn2))
        write_dict_to_Hdf5(f, self.save_to_h5_data)
        return Path(dir / f'{self.initial_timestamp}_{count}')

    def write_solver_input(self):
        freq_str = \
            f'''
VCmd.SetStringValue( var3, r"OutputFrequencyType", r"{self.output_frequency_type.value[0]}" )
VCmd.SetDoubleValue( var3, r"OutputFrequency", {self.output_frequency}  )'''
        py_script_str = resources.python2_script_X_vdbs % (self.vdb_origin, self.max_injection_time_pam_rtm, freq_str)
        used_var_nums_in_script = 3 + 1  # starts with 1

        var_count = used_var_nums_in_script
        py_vdb_blocks = []

        for i, dir_stem in enumerate(self.dirs_with_stems):
            py_vdb_lines = [resources.import_lperm % (var_count, var_count, f'{dir_stem}.lperm', var_count),
                            resources.export_file % f'{dir_stem}.vdb']
            py_vdb_blocks.append(''.join(py_vdb_lines))
            var_count += 1

        self.fn_vdb_writer = os.path.join(self.solver_input_folder, 'vdb_writerpy2.py')
        py_vdb_for_one_call = ''.join(py_vdb_blocks)
        _str = py_script_str + py_vdb_for_one_call
        with open(self.fn_vdb_writer, 'w') as f:
            f.write(''.join(_str))

    def create_vdbs(self):
        print(f'Writing .vdb files ...')
        t0 = time.time()
        call_make_vdb = fr''' "{self.vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -sessionrun "{self.fn_vdb_writer}" -nodisplay -exit'''
        args = shlex.split(call_make_vdb)
        subprocess.call(args, shell=True, stdout=subprocess.PIPE)
        print(f'Writing .vdbs took {(time.time() - t0) / 60:.1f} minutes.')

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
        print(f'Writing .unf files took {(time.time()-t0)/60:.1f} minutes.')

    def write_slurm_scripts(self):
        self.logger.info('Writing slurm script ...')

        line_before_unf = 'srun -t %d singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np %d'
        path_to_unf = self.solved_sims / self.perturbation_factors_str / \
                      f'{self.initial_timestamp}_{self.overall_count}p' / \
                      f'${{SLURM_ARRAY_TASK_ID}}/{self.initial_timestamp}_${{SLURM_ARRAY_TASK_ID}}g.unf'
        path_to_unf = path_to_unf.as_posix().replace('Y:', '/cfs/share')

        # if self.n_in_batch < 10:
        #     calls_on_small_partition = 1
        # else:
        #     calls_on_small_partition = np.round(self.n_in_batch * 0.08).astype(int)
        #
        # array = f'#SBATCH --array={calls_on_small_partition}-{self.n_in_batch - 1}%{self.num_big_hosts}'
        # self.insert_vars_into_script(array, line_before_unf, 'solve_pam_rtm_auto_big.sh', 32, path_to_unf, 'big-cpu')
        #
        # array = f'#SBATCH --array=0-{calls_on_small_partition - 1}%{self.num_small_hosts}'
        # self.insert_vars_into_script(array, line_before_unf, 'solve_pam_rtm_auto_small.sh', 8, path_to_unf, 'small-cpu')

        if self.n_in_batch >= 10:
            calls_on_small_partition = np.round(self.overall_count * 0.08).astype(int)
        else:
            calls_on_small_partition = 1

        array = f'#SBATCH --array={calls_on_small_partition}-{self.overall_count - 1}%{self.num_big_hosts}'
        self.insert_vars_into_script(array, line_before_unf, 'complete_solve_pam_rtm_auto_big.sh', 32, path_to_unf, 'big-cpu')

        array = f'#SBATCH --array=0-{calls_on_small_partition - 1}%{self.num_small_hosts}'
        self.insert_vars_into_script(array, line_before_unf, 'complete_solve_pam_rtm_auto_small.sh', 8, path_to_unf, 'small-cpu')

    def insert_vars_into_script(self, array, line_before_unf, filename, num_cpus, path_to_unf, slurm_partition):
        script_str = f'''#!/bin/sh
#SBATCH --partition={slurm_partition}
#SBATCH --mem=24000
#SBATCH --time=1000:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%a.out
{array}

{line_before_unf % (self.max_runtime_slurm, num_cpus) + ' ' + str(path_to_unf)}
'''
        self.slurm_scripts.append(filename)
        full_path = self.slurm_scripts_folder / filename
        with open(full_path, 'w') as f_win_line_endings:
            f_win_line_endings.write(script_str)
        convert_win_to_unix_lineendings(full_path)

    def alter_dat_files(self):
        for i, e in enumerate(self.dirs_with_stems):
            fn = e.parent / (e.stem + 'p.dat')
            if fn.exists():
                with open(fn, 'r') as f:
                    lines = f.readlines()
                    lines = [f'NSTEP {self.max_sim_step}\n' if 'NSTEP' in x else x for x in lines]
                with open(fn, 'w') as f:
                    f.writelines(lines)

    def copy_simfiles_to_cluster(self, move=True):
        print('Copying simfiles to cluster ...')
        t0 = time.time()
        f_endings = ['g.unf', 'ff.unf', '.elrnm', '.ndrnm', '.ptrnm', 'd.out', 'p.dat', '_meta_data.hdf5', '.lperm', '.vdb.zip']
        for e in self.dirs_with_stems:
            p = Path(str(self.solved_sims) + '/' + '/'.join(e.parts[-4:-1]))
            p.mkdir(parents=True, exist_ok=True)
            stem = Path(str(self.solved_sims) + '/' + '/'.join(e.parts[-4:]))
            for end in f_endings:
                if not move:
                    copy2(str(e) + end, str(stem) + end)
                else:
                    shutil.move(str(e) + end, str(stem) + end)
                if end == 'g.unf':
                    self.unf_files_on_storage.append(Path(str(stem) + 'g.unf').as_posix().replace('Y:', '/cfs/share'))
                    #TODO fix for posix
        print(f'Copying took {(time.time() - t0) / 60:.1f} minutes.')

    def zip_vdbs(self):
        print('Zipping .vdbs ...')
        t0 = time.time()
        vdbs = [str(x) + '.vdb' for x in self.dirs_with_stems]
        with Pool() as p:
            p.map(partial(zip_file, True), vdbs)
        print(f'Zipping took {(time.time() - t0) / 60:.1f} minutes.')

    def run(self):
        self.create_folder_structure_and_perturbate_kN()
        self.write_solver_input()
        self.create_vdbs()
        self.create_unfs_et_al()
        self.zip_vdbs()
        self.alter_dat_files()
        if self.run_on_cluster:
            self.unf_files_on_storage = [Path(str(x) + 'g.unf').as_posix().replace('Y:', '/cfs/share') for x in self.dirs_with_stems]
        if not self.run_on_cluster:
            self.copy_simfiles_to_cluster()
        self.write_slurm_scripts()
        # if not self.run_on_cluster:
        #     zip_fn = zip_folder(self.solver_input_folder, self.batch_num, delete_after=True)
        #     shutil.move(zip_fn, self.sim_files_data_heap)

