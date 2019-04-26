import random
import re
import shutil
import zipfile
from functools import partial
import os
import h5py
import shlex
import pandas
import datetime
import numpy as np
import subprocess
from shutil import copy2
from pathlib import Path
from multiprocessing import Pool
import time

import resources
from enum import Enum

from shapes import Rectangle, Circle


class OutputFrequencyType(Enum):
    Step = "Step"
    FillPercentage = "% Fill"
    Time = "Time"


def derive_k1k2_from_fvc(fvcs):
    points = np.array([(0, 1e-6), (0.4669, 1.62609e-10), (0.5318, 5.14386e-11), (0.5518, 4.28494e-11)])
    x = points[:, 0]
    y = points[:, 1]
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)


def fvc_to_k1(fvc):
    return 1.00000000e-06 * np.exp(-1.86478393e+01*fvc)


def rounded_random(value, minClip):
    return round(float(value)/ minClip) * minClip


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


def zip_folder(path, delete_after=True):
    print(f'Zipping {path} ...')
    t0 = time.time()
    ps = [x for x in Path(path).glob('**/*') if x.is_file()]
    zip_file = zipfile.ZipFile(path + '.zip', 'w')
    with zip_file:
        for file in ps:
            zip_file.write(file, compress_type=zipfile.ZIP_DEFLATED)
    print(f'Zipping took {time.time() - t0:.0f} seconds.')
    if delete_after:
        print(f'Deleting {path} ...')
        shutil.rmtree(path)


class SimCreator:
    def __init__(self, perturbation_factors=None, count=20):
        self.save_to_h5_data = {}
        self.vebatch_exec = Path(r'C:\Program Files\ESI Group\Visual-Environment\14.5\Windows-x64\VEBatch.bat')
        self.initial_timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if perturbation_factors is None:
            self.perturbation_factors = {}
        else:
            self.perturbation_factors = perturbation_factors
        self.perturbation_factors_str = 'with_shapes'# re.sub('[{\',:}]', '', str(perturbation_factors)).replace(' ', '_')
        self.count = count
        self.num_hosts = 10
        self.num_cpus = 32

        self.output_frequency_type = OutputFrequencyType.Time
        self.output_frequency = 0.5
        self.max_sim_step = 1500
        self.max_injection_time = 800000

        self.solved_sims            = Path(r'Y:\data\RTM\Lautern\output')
        sources_path                = Path(r'Y:\data\RTM\Lautern\sources')
        self.original_lperm         = sources_path / 'k1_k2_equal_one_layer.lperm'
        self.vdb_origin             = sources_path / 'flawless_one_layer.vdb'
        self.reference_erfh5        = sources_path / 'flawless_RESULT.erfh5'

        f = h5py.File(self.reference_erfh5, 'r')
        self.all_coords= f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()][:, :-1]
        self.triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()][:, :-1]
        self.x_bounds = (-20, 20)
        self.y_bounds = (-20, 20)
        self.circ_radius_bounds = (1, 3)
        self.rect_width_bounds = self.rect_height_bounds = (1, 8)
        self.grid_step = 0.125

        self.shapes = []
        if len(self.perturbation_factors.keys()) > 0:
            [self.shapes.append(Rectangle) for x in range(self.perturbation_factors['Shapes']['Rectangles']['Num'])]
            [self.shapes.append(Circle) for x in range(self.perturbation_factors['Shapes']['Circles']['Num'])]
            self.rect_fvc_bounds = self.perturbation_factors['Shapes']['Rectangles']['Fiber_Content']
            self.circ_fvc_bounds = self.perturbation_factors['Shapes']['Circles']['Fiber_Content']

        self.solver_input_folder    = Path(r'C:\Data\0_RTM_data\Data\output\%s\%s_%dp' % (self.perturbation_factors_str, self.initial_timestamp, self.count))
        self.slurm_scripts_folder   = Path(r'X:\s\t\stiebesi\slurm_scripts')
        self.slurm_partition        = "big-cpu"

        self.slurm_scripts = []
        self.dirs_with_stems = []
        self.fn_vdb_writer = []
        self.unf_files_on_storage = []

    def create_folder_structure_and_perturbate_kN(self):
        print(f'Perturbating {self.perturbation_factors_str} ...')
        t0 = time.time()
        self.solver_input_folder.mkdir(parents=True, exist_ok=True)
        df = pandas.read_csv(self.original_lperm, sep=' ')
        keys = list(df.keys())

        with Pool() as p:
            self.dirs_with_stems = p.map(partial(self.perturbate_wrapper, keys, df), range(self.count))
        # self.dirs_with_stems =self.perturbate_wrapper(keys, df, 1)
        print(f'Adding noise and writing all files took {time.time() - t0:.0f} seconds.')

    def perturbate_wrapper(self, keys, df, count, mu=0):
        new_stem = str(count)
        dir = self.solver_input_folder / new_stem
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = f'{self.initial_timestamp}_{count}.lperm'
        fn = dir / filename
        if fn.exists():
            return fn

        # factor = self.perturbation_factors['Fiber_Content']
        iK = df.keys().get_loc('Fiber_Content')
        # Perturbate FVC everywhere
        df.iloc[:, iK] = df.iloc[:, iK] + np.random.normal(mu, self.perturbation_factors['General_Sigma'], len(df))
        # Apply shapes
        all_indices_of_elements = []
        list_of_indices_of_shape = []
        for shape in self.shapes:
            y = rounded_random(random.random() * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0], self.grid_step)
            x = rounded_random(random.random() * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0], self.grid_step)
            if 'shapes' not in self.save_to_h5_data.keys():
                self.save_to_h5_data['shapes'] = []
            if shape.__name__ == 'Rectangle':
                fvc =    random.random() * (self.rect_fvc_bounds[1] - self.rect_fvc_bounds[0]) + self.rect_fvc_bounds[0]
                height = rounded_random(random.random() * (self.rect_height_bounds[1] - self.rect_height_bounds[0]) + self.rect_height_bounds[0], self.grid_step)
                width =  rounded_random(random.random() * (self.rect_width_bounds[1] - self.rect_width_bounds[0]) + self.rect_width_bounds[0], self.grid_step)
                list_of_indices_of_shape = self.get_coordinates_of_rectangle((x, y), height, width)
                _dict = {"Rectangle":
                         {"fvc": fvc,
                              "height": height,
                              "width": width
                      }
                }

            elif shape.__name__  == 'Circle':
                fvc =       random.random() * (self.circ_fvc_bounds[1] - self.circ_fvc_bounds[0]) + self.circ_fvc_bounds[0]
                radius =    rounded_random(random.random() * (self.circ_radius_bounds[1] - self.circ_radius_bounds[0]) + self.circ_radius_bounds[0], self.grid_step)

                list_of_indices_of_shape = self.get_coordinates_of_circle((x, y), radius)
                _dict = {"Circle":
                            {"fvc": fvc,
                            "radius": radius,
                      }
                }
            self.save_to_h5_data['shapes'].append(_dict)

            indices_of_elements = self.get_elements_in_shape(list_of_indices_of_shape)
            df.update(df.iloc[indices_of_elements]['Fiber_Content'] * (1 + fvc))

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
        lperm_str = df.to_string(formatters=formatters, index=False)
        with open(fn, 'w') as f:
            f.write(lperm_str)
        return Path(dir / f'{self.initial_timestamp}_{count}')

    def get_coordinates_of_rectangle(self, lower_left, height, width):
        current_rect = []

        for i in np.arange(lower_left[0], lower_left[0] + width, self.grid_step):
            for j in np.arange(lower_left[1], lower_left[1] + height, self.grid_step):
                index = np.where((self.all_coords[:, 0] == [i]) & (self.all_coords[:, 1] == [j]))[0]
                if index.size != 0:
                    current_rect.append(index[0])
        return set(current_rect)

    def get_coordinates_of_circle(self, centre, radius):
        current_indices = []
        for i in np.arange(centre[0]-radius, centre[0]+radius, self.grid_step):
            for j in np.arange(centre[1]-radius, centre[1]+radius, self.grid_step):
                distance = (i - centre[0])**2 + (j-centre[1])**2
                if distance <= radius**2:
                    index = np.where((self.all_coords[:,0] == [i]) & (self.all_coords[:,1] == [j]))
                    index = index[0]
                    if index.size != 0:
                        current_indices.append(index[0])

        #list that contains lists of the indices of circles
        return set(current_indices)

    def get_elements_in_shape(self, indeces_nodes):
        current_elements = list()
        for index, t in enumerate(self.triangle_coords):
            if t[0] in indeces_nodes and t[1] in indeces_nodes and t[2] in indeces_nodes:
                current_elements.append(index)
        return current_elements

    def write_solver_input(self):
        str = \
f'''
VCmd.SetStringValue( var3, r"OutputFrequencyType", r"{self.output_frequency_type.name}" )
VCmd.SetDoubleValue( var3, r"OutputFrequency", {self.output_frequency}  )'''
        py_script_str = resources.python2_script_X_vdbs % (self.vdb_origin, self.max_injection_time, str)
        used_var_nums_in_script = 3 + 1 # starts with 1

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
        print(f'Writing .unf files took {time.time()-t0:.0f} seconds.')

    def create_vdbs(self):
        print(f'Writing .vdb files ...')
        t0 = time.time()
        call_make_vdb = fr''' "{self.vebatch_exec}" -activeconfig Trade:CompositesandPlastics -activeapp VisualRTM -sessionrun "{self.fn_vdb_writer}" -nodisplay -exit'''
        args = shlex.split(call_make_vdb)
        subprocess.call(args, shell=True, stdout=subprocess.PIPE)
        print(f'Writing .vdbs took {time.time() - t0:.0f} seconds.')

    def write_slurm_scripts(self, timeout=15):
        print('Writing slurm script ...')
        calls = []
        line_before_unf = f'srun -t {timeout} singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np {self.num_cpus}'
        for e in self.unf_files_on_storage:
            call = '%s %s' % (line_before_unf, e)
            calls.append(call)

        if len(calls) > 1:
            delim = int(np.round(len(calls) / self.num_hosts))
        else:
            delim = 1
        sublist_calls = [calls[x:x + delim] for x in range(0, len(calls), delim)]
        for i in range(self.num_hosts):
            slurm_partition = self.slurm_partition
            num_cpus = self.num_cpus
            calls_str = '\n'.join(sublist_calls[i])
            if i == 0:
                slurm_partition = 'small-cpu'
                num_cpus = 8
                calls_str = calls_str.replace(f'-np {self.num_cpus}', f'-np {num_cpus}')

            script_str = '%s\n%s' % (f'''#!/bin/sh
#SBATCH --partition={slurm_partition}
#SBATCH --mem=24000
#SBATCH --time=100:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%x.out
''', calls_str)
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

    def copy_simfiles_to_cluster(self):
        f_endings = ['g.unf', 'ff.unf', '.elrnm', '.ndrnm', '.ptrnm', 'd.out', 'p.dat']
        for e in self.dirs_with_stems:
            p = Path(str(self.solved_sims) + '/' + '/'.join(e.parts[-4:-1]))
            p.mkdir(parents=True, exist_ok=True)
            stem = Path(str(self.solved_sims) + '/' + '/'.join(e.parts[-4:]))
            for end in f_endings:
                copy2(str(e) + end, str(stem) + end)
                if end == 'g.unf':
                    self.unf_files_on_storage.append(Path(str(stem) + 'g.unf').as_posix().replace('Y:', '/cfs/share'))

    def run(self):
        self.create_folder_structure_and_perturbate_kN()
        self.write_solver_input()
        self.create_vdbs()
        self.create_unfs_et_al()
        self.alter_dat_files()
        self.copy_simfiles_to_cluster()
        self.write_slurm_scripts()
        zip_folder(self.solver_input_folder, delete_after=True)
