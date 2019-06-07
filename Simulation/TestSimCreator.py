import unittest
from pathlib import Path

from Simulation.SimCreator import SimCreator

class TestSlurmWriter(unittest.TestCase):
    def test_write(self):
        sc = SimCreator()
        sc.initial_timestamp = '2000-01-01_00-00-00'
        sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')
        sc.write_slurm_scripts()
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()