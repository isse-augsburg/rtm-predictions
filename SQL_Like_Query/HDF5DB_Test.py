import getpass
import os
import unittest
from datetime import datetime
from pathlib import Path

import deepdish as dd
import h5py
import numpy as np

from HDF5DB import HDF5DB
from HDF5Object import HDF5Object


class HDF5DBTest(unittest.TestCase):
    def setUp(self):
        self.age = "2019-08-30_12-23-59"
        self.testfolder = Path("Tests")
        # Metadata
        self.path_meta = Path("2019-08-30_12-23-59_test_meta_data.hdf5")
        self.output_frequency_type = np.array([99])
        self.output_frequency = np.array([0.01])
        self.general_sigma = np.array([99.999])
        self.num_rect = np.array([2])
        self.fibre_content_rect = np.array([0.09, 0.99])
        self.num_circ = np.array([2])
        self.fibre_content_circ = np.array([0.08, 0.88])
        self.num_runners = np.array([2])

        self.fibre_content_runners = np.array([-0.4, -0.9])
        self.fvc_rect = np.array([0.99, 0.1])
        self.height_rect = np.array([0.09, 0.99])
        self.width_rect = np.array((1, 2.1))
        self.posx_rect = np.array((5, 30))
        self.posy_rect = np.array((5, 30))
        self.fvc_circ = np.array((0.89, 0.25))
        self.radius_circ = np.array((0.09, 0.99))
        self.posx_circ = np.array((8, 26))
        self.posy_circ = np.array((50, 33))
        self.fvc_runner = np.array((0.22, 0.73))
        self.height_runner = np.array((0.4, 1.9))
        self.width_runner = np.array((1.6, 2.7))
        self.posx_runner = np.array((30.6, 22.5))
        self.posy_runner = np.array((15, 23, 8))
        self.pos_lower_leftx_runner = np.array((14.2, 21.9))
        self.pos_lower_lefty_runner = np.array((30.4, 21.6))

        # Results
        self.path_result = Path("2019-08-30_12-23-59_test_RESULT.erfh5")
        self.ent_id = np.array((1, 2, 3, 4, 5))
        self.index_ident = np.array((1, 2, 3, 4))
        self.index_val = np.array((0.001, 55, 2354, 87653))
        self.res = np.array(
            (
                (0, 1, 2, 3, 4),
                (0, 6.3, 20.5, 36.8, 73.1),
                (0, 0, 0, 0, 3),
                (0, 9, 99, 99.5, 99.9),
            )
        )
        self.state1 = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.state99 = np.array((0, 0, 0, 0, 0, 0, 0, 0, 0.1))
        self.state999 = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1))

        self.test_meta = {
            "output_frequency_type": self.output_frequency_type,
            "output_frequency": self.output_frequency,
            "perturbation_factors": {
                "General_Sigma": self.general_sigma,
                "Shapes": {
                    "Rectangles": {
                        "Fiber_Content": self.fibre_content_rect,
                        "Num": self.num_rect,
                    },
                    "Circles": {
                        "Fiber_Content": self.fibre_content_circ,
                        "Num": self.num_circ,
                    },
                    "Runners": {
                        "Fiber_Content": self.fibre_content_runners,
                        "Num": self.num_runners,
                    },
                },
            },
            "shapes": {
                "Rectangle": {
                    "fvc": self.fvc_rect,
                    "height": self.height_rect,
                    "width": self.width_rect,
                    "posX": self.posx_rect,
                    "posY": self.posy_rect,
                },
                "Circle": {
                    "fvc": self.fvc_circ,
                    "radius": self.radius_circ,
                    "posX": self.posx_circ,
                    "posY": self.posy_circ,
                },
                "Runner": {
                    "fvc": self.fvc_runner,
                    "height": self.height_runner,
                    "width": self.width_runner,
                    "posX": self.posx_runner,
                    "posY": self.posy_runner,
                    "posLowerLeftX": self.pos_lower_leftx_runner,
                    "posLowerLeftY": self.pos_lower_lefty_runner,
                },
            },
        }

        self.test_result = {
            "post": {
                "multistate": {
                    "TIMESERIES1": {
                        "multientityresults": {
                            "GLOBAL": {},
                            "INLET_OUTLET": {},
                            "SENSOR": {
                                "PRESSURE": {
                                    "ZONE1_set1": {
                                        "erfblock": {
                                            "entid": self.ent_id,
                                            "indexident": self.index_ident,
                                            "indexval": self.index_val,
                                            "res": self.res,
                                        }
                                    }
                                }
                            },
                        }
                    }
                },
                "singlestate": {
                    "state000000000001": {
                        "entityresults": {
                            "NODE": {
                                "FILLING_FACTOR": {
                                    "ZONE1_set1": {"erfblock": {"res": self.state1}}
                                }
                            }
                        }
                    },
                    "state000000000099": {
                        "entityresults": {
                            "NODE": {
                                "FILLING_FACTOR": {
                                    "ZONE1_set1": {"erfblock": {"res": self.state99}}
                                }
                            }
                        }
                    },
                    "state000000000999": {
                        "entityresults": {
                            "NODE": {
                                "FILLING_FACTOR": {
                                    "ZONE1_set1": {"erfblock": {"res": self.state999}}
                                }
                            }
                        }
                    },
                },
            }
        }

        os.mkdir(os.getcwd() / self.testfolder)
        dd.io.save(str(self.testfolder / self.path_meta), None, compression=None)
        dd.io.save(str(self.testfolder / self.path_result), None, compression=None)
        self.test_object = HDF5Object(
            str(self.testfolder / self.path_meta),
            str(self.testfolder / self.path_result),
        )
        self.test_object.meta_path = os.getcwd() / self.testfolder / self.path_meta
        self.test_object.output_frequency_type = self.output_frequency_type
        self.test_object.output_frequency = self.output_frequency
        self.test_object.general_sigma = self.general_sigma
        self.test_object.number_of_rectangles = self.num_rect
        self.test_object.fibre_content_rectangles = self.fibre_content_rect
        self.test_object.number_of_circles = self.num_circ
        self.test_object.fibre_content_circles = self.fibre_content_circ
        self.test_object.number_of_runners = self.num_runners
        self.test_object.number_of_sensors = np.ma.size(self.res, 1)
        self.num_shapes = self.num_circ + self.num_rect + self.num_runners
        self.test_object.number_of_shapes = self.num_shapes
        self.test_object.fibre_content_runners = self.fibre_content_runners
        self.test_object.fvc_rectangle = self.fvc_rect
        self.test_object.height_rectangle = self.height_rect
        self.test_object.width_rectangle = self.width_rect
        self.test_object.posx_rectangle = self.posx_rect
        self.test_object.posy_rectangle = self.posy_rect
        self.test_object.fvc_circle = self.fvc_circ
        self.test_object.radius_circle = self.radius_circ
        self.test_object.posx_circle = self.posx_circ
        self.test_object.posy_circle = self.posy_circ
        self.test_object.fvc_runner = self.fvc_runner
        self.test_object.height_runner = self.height_runner
        self.test_object.width_runner = self.width_runner
        self.test_object.posx_runner = self.posx_runner
        self.test_object.posy_runner = self.posy_runner
        self.test_object.pos_lower_leftx_runner = self.pos_lower_leftx_runner
        self.test_object.pos_lower_lefty_runner = self.pos_lower_lefty_runner

        self.test_object.result_path = os.getcwd() / self.testfolder / self.path_meta
        self.test_object.avg_level = np.sum(self.state999) / len(self.state999)
        self.test_object.age = datetime.strptime(self.age, "%Y-%m-%d_%H-%M-%S")
        os.remove(str(self.testfolder / self.path_meta))
        os.remove(str(self.testfolder / self.path_result))
        self.setup_test_data()

    def setup_test_data(self):
        dd.io.save(
            str(self.testfolder / self.path_meta), self.test_meta, compression=None
        )
        dd.io.save(
            str(self.testfolder / self.path_result), self.test_result, compression=None
        )

    def tearDown(self):
        os.remove(os.getcwd() / self.testfolder / self.path_meta)
        os.remove(os.getcwd() / self.testfolder / self.path_result)
        os.rmdir(os.getcwd() / self.testfolder)

    def test_load_save(self):
        self.test_db = HDF5DB()
        self.test_db2 = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.test_db.save(str(self.testfolder), "HDF5TESTDB")
        self.test_db2.load(str(self.testfolder), "HDF5TESTDB")
        self.assertTrue(self.test_db.__eq__(self.test_db2))
        os.remove(self.testfolder / Path("HDF5TESTDB.h5db"))

    def test_add_objects_from_path(self):
        self.test_result_db = HDF5DB()
        self.test_db = HDF5DB()
        self.test_result_db.add_object(self.test_object)
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.assertTrue(self.test_db.__eq__(self.test_result_db))

    # Select-tests
    # Metapath and resultpath
    def test_select_paths(self):
        # Illegal operator > and <
        # Meta
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        self.temp = len(self.test_db.hdf5_object_list)
        self.test_db.select(
            "path_meta", str(os.getcwd() / self.testfolder / self.path_meta), "<"
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        self.test_db.select(
            "path_meta", str(os.getcwd() / self.testfolder / self.path_meta), ">"
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        self.test_db.select(
            "path_meta", str(os.getcwd() / self.testfolder / self.path_meta), "="
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        # Result
        self.temp = len(self.test_db.hdf5_object_list)
        self.test_db.select(
            "path_result", str(os.getcwd() / self.testfolder / self.path_result), "<"
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        self.test_db.select(
            "path_result", str(os.getcwd() / self.testfolder / self.path_result), ">"
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        self.test_db.select(
            "path_result", str(os.getcwd() / self.testfolder / self.path_result), "="
        )
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)

    def test_select_output_frequency(self):
        # Output_frequency
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.temp = self.test_object.output_frequency[0] - 0.001
        self.test_db.select("output_frequency", self.temp, ">")
        self.assertGreater(self.test_db.hdf5_object_list[0].output_frequency, self.temp)
        self.temp = self.test_object.output_frequency[0] + 0.001
        self.test_db.select("output_frequency", self.temp, "<")
        self.assertLess(self.test_db.hdf5_object_list[0].output_frequency, self.temp)
        self.temp = self.test_object.output_frequency[0]
        self.test_db.select("output_frequency", self.temp, "=")
        self.assertEqual(self.test_db.hdf5_object_list[0].output_frequency, self.temp)
        # Output_frequency_type
        self.temp = self.test_object.output_frequency_type[0] - 0.001
        self.test_db.select("output_frequency_type", self.temp, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].output_frequency_type, self.temp
        )
        self.temp = self.test_object.output_frequency_type[0] + 0.001
        self.test_db.select("output_frequency_type", self.temp, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].output_frequency_type, self.temp
        )
        self.temp = self.test_object.output_frequency_type[0]
        self.test_db.select("output_frequency_type", self.temp, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].output_frequency_type, self.temp
        )

    def test_select_general_sigma(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.test_db.select("general_sigma", self.general_sigma - 1, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].general_sigma, self.general_sigma - 1
        )
        self.test_db.select("general_sigma", self.general_sigma + 1, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].general_sigma, self.general_sigma + 1
        )
        self.test_db.select("general_sigma", self.general_sigma, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].general_sigma, self.general_sigma
        )

    # Number of circles, rectangles, runners, shapes
    def test_select_num_shapes(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        # Circles
        self.test_db.select("number_of_circles", self.num_circ - 1, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_circles, self.num_circ - 1
        )
        self.test_db.select("number_of_circles", self.num_circ + 1, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].number_of_circles, self.num_circ + 1
        )
        self.test_db.select("number_of_circles", self.num_circ, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].number_of_circles, self.num_circ
        )
        # Rectangles
        self.test_db.select("number_of_rectangles", self.num_rect - 1, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_rectangles, self.num_rect - 1
        )
        self.test_db.select("number_of_rectangles", self.num_rect + 1, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].number_of_circles, self.num_rect + 1
        )
        self.test_db.select("number_of_rectangles", self.num_circ, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].number_of_circles, self.num_rect
        )
        # Runners
        self.test_db.select("number_of_runners", self.num_rect - 1, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_runners, self.num_runners - 1
        )
        self.test_db.select("number_of_runners", self.num_rect + 1, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].number_of_runners, self.num_runners + 1
        )
        self.test_db.select("number_of_runners", self.num_circ, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].number_of_runners, self.num_runners
        )
        # Shapes
        self.test_db.select("number_of_shapes", self.num_shapes - 1, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_shapes, self.num_shapes - 1
        )
        self.test_db.select("number_of_shapes", self.num_shapes + 1, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].number_of_shapes, self.num_shapes + 1
        )
        self.test_db.select("number_of_shapes", self.num_shapes, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].number_of_shapes, self.num_shapes
        )

    def test_select_fibre_content(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        # Circles
        self.test_db.select(
            "fibre_content_circles", np.amax(self.fibre_content_circ) - 1, ">"
        )
        self.assertGreater(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_circles),
            np.amax(self.fibre_content_circ - 1),
        )
        self.test_db.select(
            "fibre_content_circles", np.amin(self.fibre_content_circ) + 1, "<"
        )
        self.assertLess(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_circles),
            np.amin(self.fibre_content_circ + 1),
        )
        self.test_db.select("fibre_content_circles", self.fibre_content_circ[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_circles),
            self.fibre_content_circ[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_circles),
            self.fibre_content_circ[1],
        )
        # Rectangles
        self.test_db.select(
            "fibre_content_rectangles", np.amax(self.fibre_content_rect) - 1, ">"
        )
        self.assertGreater(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_rectangles),
            np.amax(self.fibre_content_rect) - 1,
        )
        self.test_db.select(
            "fibre_content_rectangles", np.amin(self.fibre_content_rect) + 1, "<"
        )
        self.assertLess(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_rectangles),
            np.amin(self.fibre_content_rect) + 1,
        )
        self.test_db.select("fibre_content_rectangles", self.fibre_content_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_rectangles),
            self.fibre_content_rect[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_rectangles),
            self.fibre_content_rect[1],
        )
        # Runners
        self.test_db.select(
            "fibre_content_runners", np.amax(self.fibre_content_runners) - 1, ">"
        )
        self.assertGreater(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_runners),
            np.amax(self.fibre_content_runners - 1),
        )
        self.test_db.select(
            "fibre_content_runners", np.amin(self.fibre_content_runners + 1), "<"
        )
        self.assertLess(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_runners),
            np.amin(self.fibre_content_runners + 1),
        )
        self.test_db.select("fibre_content_runners", self.fibre_content_runners[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fibre_content_runners),
            self.fibre_content_runners[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fibre_content_runners),
            self.fibre_content_runners[1],
        )

    def test_select_circle(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        # Fvc of circle
        # <
        self.temp = np.amax(self.fvc_circ) + 0.1
        self.test_db.select("fvc_circle", self.temp, "<")
        self.assertLess(np.amax(self.test_db.hdf5_object_list[0].fvc_circle), self.temp)
        # >
        self.temp = np.amin(self.fvc_circ) - 0.1
        self.test_db.select("fvc_circle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].fvc_circle), self.temp
        )
        # =
        self.test_db.select("fvc_circle", self.fvc_circ[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fvc_circle), self.fvc_circ[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fvc_circle), self.fvc_circ[1]
        )
        # Radius of circle
        # <
        self.temp = np.amax(self.radius_circ) + 1
        self.test_db.select("radius_circle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].radius_circle), self.temp
        )
        # >
        self.temp = np.amin(self.radius_circ) - 1
        self.test_db.select("radius_circle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].radius_circle), self.temp
        )
        # =
        self.test_db.select("radius_circle", self.radius_circ[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].radius_circle), self.radius_circ[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].radius_circle), self.radius_circ[1]
        )
        # Position x of circle
        # <
        self.temp = np.amax(self.posx_circ) + 1
        self.test_db.select("posx_circle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posx_circle), self.temp
        )
        # >
        self.temp = np.amin(self.posx_circ) - 1
        self.test_db.select("posx_circle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posx_circle), self.temp
        )
        # =
        self.test_db.select("posx_circle", self.posx_circ[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posx_circle), self.posx_circ[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posx_circle), self.posx_circ[1]
        )
        # Position y of circle
        # <
        self.temp = np.amax(self.posy_circ) + 1
        self.test_db.select("posy_circle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posy_circle), self.temp
        )
        # >
        self.temp = np.amin(self.posy_circ) - 1
        self.test_db.select("posy_circle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posy_circle), self.temp
        )
        # =
        self.test_db.select("posy_circle", self.posy_circ[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posy_circle), self.posy_circ[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posy_circle), self.posy_circ[1]
        )

    def test_select_rectangle(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        # Fvc of rectangle
        # <
        self.temp = np.amax(self.fvc_rect) + 0.1
        self.test_db.select("fvc_rectangle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].fvc_rectangle), self.temp
        )
        # >
        self.temp = np.amin(self.fvc_rect) - 0.1
        self.test_db.select("fvc_rectangle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].fvc_rectangle), self.temp
        )
        # =
        self.test_db.select("fvc_rectangle", self.fvc_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fvc_rectangle), self.fvc_rect[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fvc_rectangle), self.fvc_rect[1]
        )
        # Height of rectangle
        # <
        self.temp = np.amax(self.height_rect) + 1
        self.test_db.select("height_rectangle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].height_rectangle), self.temp
        )
        # >
        self.temp = np.amin(self.height_rect) - 1
        self.test_db.select("height_rectangle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].height_rectangle), self.temp
        )
        # =
        self.test_db.select("height_rectangle", self.height_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].height_rectangle),
            self.height_rect[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].height_rectangle),
            self.height_rect[1],
        )
        # Width of rectangle
        # <
        self.temp = np.amax(self.width_rect) + 1
        self.test_db.select("width_rectangle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].width_rectangle), self.temp
        )
        # >
        self.temp = np.amin(self.width_rect) - 1
        self.test_db.select("width_rectangle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].width_rectangle), self.temp
        )
        # =
        self.test_db.select("width_rectangle", self.width_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].width_rectangle),
            self.width_rect[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].width_rectangle),
            self.width_rect[1],
        )
        # Position x of rectangle
        # <
        self.temp = np.amax(self.posx_rect) + 1
        self.test_db.select("posx_rectangle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posx_rectangle), self.temp
        )
        # >
        self.temp = np.amin(self.posx_rect) - 1
        self.test_db.select("posx_rectangle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posx_rectangle), self.temp
        )
        # =
        self.test_db.select("posx_rectangle", self.posx_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posx_rectangle), self.posx_rect[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posx_rectangle), self.posx_rect[1]
        )
        # Position y of rectangle
        # <
        self.temp = np.amax(self.posy_rect) + 1
        self.test_db.select("posy_rectangle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posy_rectangle), self.temp
        )
        # >
        self.temp = np.amin(self.posy_rect) - 1
        self.test_db.select("posy_rectangle", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posy_rectangle), self.temp
        )
        # =
        self.test_db.select("posy_rectangle", self.posy_rect[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posy_rectangle), self.posy_rect[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posy_rectangle), self.posy_rect[1]
        )

    def test_select_runner(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        # Fvc of runner
        # <
        self.temp = np.amax(self.fvc_runner) + 0.1
        self.test_db.select("fvc_runner", self.temp, "<")
        self.assertLess(np.amax(self.test_db.hdf5_object_list[0].fvc_runner), self.temp)
        # >
        self.temp = np.amin(self.fvc_runner) - 0.1
        self.test_db.select("fvc_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].fvc_runner), self.temp
        )
        # =
        self.test_db.select("fvc_runner", self.fvc_runner[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].fvc_runner), self.fvc_runner[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].fvc_runner), self.fvc_runner[1]
        )
        # Height of runner
        # <
        self.temp = np.amax(self.height_runner) + 1
        self.test_db.select("height_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].height_runner), self.temp
        )
        # >
        self.temp = np.amin(self.height_runner) - 1
        self.test_db.select("height_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].height_runner), self.temp
        )
        # =
        self.test_db.select("height_runner", self.height_runner[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].height_runner),
            self.height_runner[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].height_runner),
            self.height_runner[1],
        )
        # Width of runner
        # <
        self.temp = np.amax(self.width_runner) + 1
        self.test_db.select("width_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].width_runner), self.temp
        )
        # >
        self.temp = np.amin(self.width_runner) - 1
        self.test_db.select("width_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].width_runner), self.temp
        )
        # =
        self.test_db.select("width_runner", self.width_runner[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].width_runner), self.width_runner[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].width_runner), self.width_runner[1]
        )
        # Position x of runner
        # <
        self.temp = np.amax(self.posx_runner) + 1
        self.test_db.select("posx_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posx_runner), self.temp
        )
        # >
        self.temp = np.amin(self.posx_runner) - 1
        self.test_db.select("posx_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posx_runner), self.temp
        )
        # =
        self.test_db.select("posx_runner", self.posx_runner[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posx_runner), self.posx_runner[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posx_runner), self.posx_runner[1]
        )
        # Position y of runner
        # <
        self.temp = np.amax(self.posy_runner) + 1
        self.test_db.select("posy_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posy_runner), self.temp
        )
        # >
        self.temp = np.amin(self.posy_runner) - 1
        self.test_db.select("posy_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].posy_runner), self.temp
        )
        # =
        self.test_db.select("posy_runner", self.posy_runner[1], "=")
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].posy_runner), self.posy_runner[1]
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].posy_runner), self.posy_runner[1]
        )
        # Position lower left x of runner
        # <
        self.temp = np.amax(self.pos_lower_leftx_runner) + 1
        self.test_db.select("pos_lower_leftx_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), self.temp
        )
        # >
        self.temp = np.amin(self.pos_lower_leftx_runner) - 1
        self.test_db.select("pos_lower_leftx_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), self.temp
        )
        # =
        self.test_db.select(
            "pos_lower_leftx_runner", self.pos_lower_leftx_runner[1], "="
        )
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner),
            self.pos_lower_leftx_runner[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner),
            self.pos_lower_leftx_runner[1],
        )
        # Position lower left y of runner
        # <
        self.temp = np.amax(self.pos_lower_lefty_runner) + 1
        self.test_db.select("pos_lower_lefty_runner", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), self.temp
        )
        # >
        self.temp = np.amin(self.pos_lower_lefty_runner) - 1
        self.test_db.select("pos_lower_lefty_runner", self.temp, ">")
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), self.temp
        )
        # =
        self.test_db.select(
            "pos_lower_lefty_runner", self.pos_lower_lefty_runner[1], "="
        )
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner),
            self.pos_lower_lefty_runner[1],
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner),
            self.pos_lower_lefty_runner[1],
        )

    def test_select_fillinglevel(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path("Tests")
        self.temp = self.test_object.avg_level - 0.1
        self.test_db.select("avg_level", self.temp, ">")
        self.assertGreater(self.test_db.hdf5_object_list[0].avg_level, self.temp)
        self.temp = self.test_object.avg_level + 0.1
        self.test_db.select("avg_level", self.temp, "<")
        self.assertLess(self.test_db.hdf5_object_list[0].avg_level, self.temp)
        self.test_db.select("avg_level", self.test_object.avg_level, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].avg_level, self.test_object.avg_level
        )

    def test_select_age(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.temp = (
            "2019-08-30_12-23-58"
        )  # datetime.strptime("2019-08-30_12-23-58", "%Y-%m-%d_%H-%M-%S")
        self.test_db.select("age", self.temp, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime(self.temp, "%Y-%m-%d_%H-%M-%S"),
        )
        self.temp = (
            "2019-09-29_12-24-59"
        )  # datetime.strptime("2019-09-29_12-24-59", "%Y-%m-%d_%H-%M-%S")
        self.test_db.select("age", self.temp, "<")
        self.assertLess(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime(self.temp, "%Y-%m-%d_%H-%M-%S"),
        )
        self.temp = self.age
        self.test_db.select("age", self.temp, "=")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime(self.temp, "%Y-%m-%d_%H-%M-%S"),
        )

    def test_select_num_sensors(self):
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path(str(self.testfolder))
        self.temp = np.shape(self.res[()])[1] - 1
        self.test_db.select("number_of_sensors", self.temp, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_sensors, self.temp
        )
        self.temp = np.shape(self.res[()])[1] + 1
        self.test_db.select("number_of_sensors", self.temp, "<")
        self.assertLess(self.test_db.hdf5_object_list[0].number_of_sensors, self.temp)
        self.temp = np.shape(self.res[()])[1]
        self.test_db.select("number_of_sensors", self.temp, "=")
        self.assertEqual(self.test_db.hdf5_object_list[0].number_of_sensors, self.temp)


if __name__ == "__main__":
    unittest.main()