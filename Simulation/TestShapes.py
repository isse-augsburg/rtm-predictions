import unittest
from pathlib import Path

import h5py
import pandas
import numpy as np
from PIL import Image

from Simulation.Shapes import Rectangle, Circle
from Simulation.SimCreator import SimCreator, OutputFrequencyType


class TestShapes(unittest.TestCase):
    def setUp(self):
        # print('setup')
        self.perturbation_factors = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 1,
                        "Fiber_Content": [.7, .8]
                    },
                "Circles":
                    {
                        "Num": 1,
                        "Fiber_Content": [.7, .8]
                    },
                "Runners":
                    {
                        "Num": 0,
                        "Fiber_Content": [-.7, -.8]
                    }
            }
        }

    def setup_leoben(self):
        data_path = Path(r'Y:\data\RTM\Leoben')
        self.sc = SimCreator(perturbation_factors=self.perturbation_factors, data_path=data_path)
        self.sc.initial_timestamp = '2000-01-01_00-00-00'
        self.sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')
        self.sc.solver_input_folder = Path(r'X:\s\t\stiebesi\code\tests\solver_input_folder')

    def setup_lautern(self):
        data_path = Path(r'Y:\data\RTM\Lautern')
        self.sc = SimCreator(perturbation_factors=self.perturbation_factors, data_path=data_path)
        self.sc.initial_timestamp = '2000-01-01_00-00-00'
        self.sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')
        self.sc.solver_input_folder = Path(r'X:\s\t\stiebesi\code\tests\solver_input_folder')
        self.reference_im_1_rect = Image.open(Path().absolute().parent / 'Test_Data' / 'reference_lautern_1_rect.png')

    def test_get_coordinates_of_rectangle_leoben(self):
        self.setup_leoben()
        lower_left = (1, 1)
        height = 5
        width = 5
        rect = self.sc.Shaper.get_coordinates_of_rectangle(lower_left, height, width)
        self.assertGreater(len(rect), 0)

        lower_left = (29, 29)
        height = 10
        width = 10
        rect = self.sc.Shaper.get_coordinates_of_rectangle(lower_left, height, width)
        self.assertGreater(len(rect), 0)

    def test_get_coordinates_of_rectangle_lautern(self):
        self.setup_lautern()
        lower_left = (1, 1)
        height = 5
        width = 5
        rect = self.sc.Shaper.get_coordinates_of_rectangle(lower_left, height, width)
        self.assertGreater(len(rect), 0)

        lower_left = (20, 20)
        height = 10
        width = 10
        rect = self.sc.Shaper.get_coordinates_of_rectangle(lower_left, height, width)
        self.assertGreater(len(rect), 0)


    def test_get_coordinates_of_circle_lautern(self):
        self.setup_lautern()
        centre = (1, 1)
        radius = 5

        circ = self.sc.Shaper.get_coordinates_of_circle(centre, radius)
        self.assertGreater(len(circ), 0)

        centre = (20, 20)
        radius = 10
        circ = self.sc.Shaper.get_coordinates_of_circle(centre, radius)
        self.assertGreater(len(circ), 0)

    def test_get_coordinates_of_circle_leoben(self):
        self.setup_leoben()
        centre = (1, 1)
        radius = 5

        circ = self.sc.Shaper.get_coordinates_of_circle(centre, radius)
        self.assertGreater(len(circ), 0)

        centre = (20, 20)
        radius = 10
        circ = self.sc.Shaper.get_coordinates_of_circle(centre, radius)
        self.assertGreater(len(circ), 0)

    def test_apply_shapes_lautern(self):
        self.setup_lautern()
        self.sc.Shaper.shapes = [
                                    Rectangle(lower_left=(0, 0), width=10, height=3, fvc=0.9),
                                ]
        df = pandas.read_csv(self.sc.original_lperm, sep=' ')
        _save_dict = {}
        self.sc.Shaper.apply_shapes(df, _save_dict, randomize=False)
        im, _, _ = self.sc.Shaper.create_img_from_lperm(df)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_im_1_rect))

    def test_apply_shapes_leoben(self):
        self.setup_leoben()
        self.sc.Shaper.shapes = [
                                    Rectangle(lower_left=(0, 0), width=10, height=3),
                                    # Circle(center=(7, 7), radius=3)
                                ]
        df = pandas.read_csv(self.sc.original_lperm, sep=' ')
        _save_dict = {}
        self.sc.Shaper.apply_shapes(df, _save_dict, randomize=False)
        im, _, _ = self.sc.Shaper.create_img_from_lperm(df)

    def tearDown(self):
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.unlink() for x in all_files if x.is_file()]
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.rmdir() for x in all_files if x.is_dir()]


if __name__ == '__main__':
    unittest.main()
