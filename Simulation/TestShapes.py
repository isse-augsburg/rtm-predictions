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

    def setup_paths(self):
        self.sc.initial_timestamp = '2000-01-01_00-00-00'
        self.sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')
        self.sc.solver_input_folder = Path(r'X:\s\t\stiebesi\code\tests\solver_input_folder')
        self.test_folder = Path(r'X:\s\t\stiebesi\code\tests\shaper')

    def setup_leoben(self):
        self.sc = SimCreator(perturbation_factors=self.perturbation_factors, data_path=Path(r'Y:\data\RTM\Leoben'))
        self.setup_paths()
        self.reference_im_1_rect            = Image.open(self.test_folder / 'reference_leoben_1_rect.png')
        self.reference_im_1_circ            = Image.open(self.test_folder / 'reference_leoben_1_circ.png')
        self.reference_1_of_rect_and_circ   = Image.open(self.test_folder / 'reference_leoben_1_of_rect_and_circ.png')

    def setup_lautern(self):
        self.sc = SimCreator(perturbation_factors=self.perturbation_factors, data_path=Path(r'Y:\data\RTM\Lautern'))
        self.setup_paths()
        self.reference_im_1_rect            = Image.open(self.test_folder / 'reference_lautern_1_rect.png')
        self.reference_im_1_circ            = Image.open(self.test_folder / 'reference_lautern_1_circ.png')
        self.reference_1_of_rect_and_circ   = Image.open(self.test_folder / 'reference_lautern_1_of_rect_and_circ.png')

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

    def test_apply_1_rectangle_lautern(self):
        im = self.apply_shape_list([Rectangle(lower_left=(0, 0), width=10, height=3, fvc=0.9)],
                                   is_lautern=True)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_im_1_rect))

    def test_apply_1_circle_lautern(self):
        im = self.apply_shape_list([Circle(center=(0, 0), radius=10, fvc=0.9)],
                                   is_lautern=True)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_im_1_circ))

    def test_apply_1_of_rect_and_circ_lautern(self):
        im = self.apply_shape_list([Rectangle(lower_left=(5, 5), width=10, height=3, fvc=0.9),
                                    Circle(center=(0, 0), radius=5, fvc=0.9)],
                                   is_lautern=True)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_1_of_rect_and_circ))

    # TODO test why there are gaps at the sensors
    # TODO add robustness at corners: fix for now: do not allow overlap
    def apply_shape_list(self, shape_list, is_lautern):
        if is_lautern:
            self.setup_lautern()
            size = (465, 465)
        else:
            self.setup_leoben()
            size = (375, 300)
        self.sc.Shaper.triangle_coords = self.sc.Shaper.triangle_coords - self.sc.Shaper.triangle_coords.min()
        self.sc.Shaper.shapes = shape_list
        df = pandas.read_csv(self.sc.original_lperm, sep=' ')
        self.sc.Shaper.apply_shapes(df, save_to_h5_data={}, randomize=False)
        im, _, _ = self.sc.Shaper.create_img_from_lperm(df, size)
        return im

    def test_apply_1_rectangle_leoben(self):
        im = self.apply_shape_list([Rectangle(lower_left=(5, 5), width=10, height=3, fvc=0.9)],
                                   is_lautern=False)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_im_1_rect))

    def test_apply_1_circle_leoben(self):
        im = self.apply_shape_list([Circle(center=(3, 3), radius=2, fvc=0.9)],
                                   is_lautern=False)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_im_1_circ))

    def test_apply_1_of_rect_and_circ_leoben(self):
        im = self.apply_shape_list([Rectangle(lower_left=(1, 1), width=10, height=3, fvc=0.9),
                                    Circle(center=(5, 5), radius=2, fvc=0.9)],
                                   is_lautern=False)
        np.testing.assert_array_equal(np.asarray(im), np.asarray(self.reference_1_of_rect_and_circ))

    def tearDown(self):
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.unlink() for x in all_files if x.is_file()]
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.rmdir() for x in all_files if x.is_dir()]


if __name__ == '__main__':
    unittest.main()
