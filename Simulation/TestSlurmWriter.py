import unittest
from pathlib import Path

from Simulation.SimCreator import SimCreator


class TestSlurmWriter(unittest.TestCase):
    maxDiff = None

    big_cpu_reference = '''#!/bin/sh
#SBATCH --partition=big-cpu
#SBATCH --mem=24000
#SBATCH --time=1000:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task=32
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%a.out
#SBATCH --array=2-19%9

srun -t 15 singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np 32 /cfs/share/data/RTM/Lautern/output/with_shapes/2000-01-01_00-00-00_20p/${SLURM_ARRAY_TASK_ID}/2000-01-01_00-00-00_${SLURM_ARRAY_TASK_ID}g.unf
'''

    small_cpu_reference = '''#!/bin/sh
#SBATCH --partition=small-cpu
#SBATCH --mem=24000
#SBATCH --time=1000:00:00
#SBATCH --job-name=PAM_RTM
#SBATCH --cpus-per-task=8
#SBATCH --output=/cfs/home/s/t/stiebesi/logs_slurm/slurm-%A-%a.out
#SBATCH --array=0-1%1

srun -t 15 singularity run -B /cfs:/cfs /cfs/share/singularity_images/pamrtm_2019_0.simg -np 8 /cfs/share/data/RTM/Lautern/output/with_shapes/2000-01-01_00-00-00_20p/${SLURM_ARRAY_TASK_ID}/2000-01-01_00-00-00_${SLURM_ARRAY_TASK_ID}g.unf
'''

    def setUp(self):
        self.sc = SimCreator(overall_count=20)
        self.sc.initial_timestamp = '2000-01-01_00-00-00'
        self.sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')

    def test_write_big(self):
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_big.sh') as f:
            out_lines = f.readlines()
            reference_lines = self.big_cpu_reference.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_write_small(self):
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_small.sh') as f:
            out_lines = f.readlines()
            reference_lines = self.small_cpu_reference.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_max_runtime_slurm(self):
        self.sc.max_runtime_slurm = 45
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_big.sh') as f:
            out_lines = f.readlines()
            str = self.big_cpu_reference.replace('srun -t 15', f'srun -t {self.sc.max_runtime_slurm}')
            reference_lines = str.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_max_runtime_slurm_small(self):
        self.sc.max_runtime_slurm = 45
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_small.sh') as f:
            out_lines = f.readlines()
            str = self.small_cpu_reference.replace('srun -t 15', f'srun -t {self.sc.max_runtime_slurm}')
            reference_lines = str.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_slurm_array(self):
        self.sc.n_in_batch = 100
        self.sc.overall_count = 500
        calls_on_small_partition = int(self.sc.overall_count * 0.08)
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_big.sh') as f:
            out_lines = f.readlines()
            str = self.big_cpu_reference.replace(
                '#SBATCH --array=2-19%9', f'#SBATCH --array={calls_on_small_partition}-{self.sc.overall_count - 1}%9')
            str = str.replace('2000-01-01_00-00-00_20p', f'2000-01-01_00-00-00_{self.sc.overall_count}p')
            reference_lines = str.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_slurm_array_small(self):
        self.sc.n_in_batch = 100
        self.sc.overall_count = 500
        calls_on_small_partition = int(self.sc.overall_count * 0.08)
        self.sc.write_slurm_scripts()
        with open(self.sc.slurm_scripts_folder / 'complete_solve_pam_rtm_auto_small.sh') as f:
            out_lines = f.readlines()
            str = self.small_cpu_reference.replace(
                '#SBATCH --array=0-1%1', f'#SBATCH --array=0-{calls_on_small_partition - 1}%1')
            str = str.replace('2000-01-01_00-00-00_20p', f'2000-01-01_00-00-00_{self.sc.overall_count}p')
            reference_lines = str.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()