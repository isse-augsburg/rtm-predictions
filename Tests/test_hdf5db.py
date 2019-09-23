import os
import shutil
import unittest
from datetime import datetime
from pathlib import Path

import deepdish as dd
import numpy as np
import sys
if (os.system == "nt"):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\HDF5DB")
else:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/HDF5DB")

import hdf5db_object
import hdf5db_toolbox


class TestHDF5DB(unittest.TestCase):
    def setUp(self):
        self.age = "2019-08-30_12-23-59"
        self.testfolder = Path("H5_Testfolder")
        # Metadata
        self.path_meta = Path("2019-08-30_12-23-59_test_meta_data.hdf5")
        self.output_frequency_type = 99
        self.output_frequency = 2
        self.general_sigma = 99.999
        self.num_rect = 2
        self.fibre_content_rect = np.array([0.09, 0.99])
        self.num_circ = 2
        self.fibre_content_circ = np.array([0.09, 0.99])
        self.num_runners = 2
        self.fibre_content_runners = np.array([0.09, 0.99])

        self.fvc_rect = np.array([0.99, 0.1])
        self.height_rect = np.array([0.09, 0.99])
        self.width_rect = np.array([1, 2.1])
        self.posx_rect = np.array([5, 30])
        self.posy_rect = np.array([5, 30])
        self.fvc_circ = np.array([0.99, 0.1])
        self.radius_circ = np.array([0.09, 0.99])
        self.posx_circ = np.array([5, 30])
        self.posy_circ = np.array([5, 30])
        self.fvc_runner = np.array([0.99, 0.1])
        self.height_runner = np.array([0.09, 0.99])
        self.width_runner = np.array([1, 2.1])
        self.posx_runner = np.array([5, 30])
        self.posy_runner = np.array([5, 30])
        self.pos_lower_leftx_runner = np.array([14.2, 21.9])
        self.pos_lower_lefty_runner = np.array([30.4, 21.6])

        # Results
        self.path_result = Path("2019-08-30_12-23-59_test_RESULT.erfh5")
        self.ent_id = np.array([1, 2, 3, 4, 5])
        self.index_ident = np.array([1, 2, 3, 4])
        self.index_val = np.array([0.001, 55, 2354, 87653])
        self.res = np.array(
            [
                [0, 1, 2, 3, 4],
                [0, 6.3, 20.5, 36.8, 73.1],
                [0, 0, 0, 0, 3],
                [0, 9, 99, 99.5, 99.9],
            ]
        )
        self.state1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.state99 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.1])
        self.state999 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1])

        self.test_meta = {
            "output_frequency_type": np.array([self.output_frequency_type]),
            "output_frequency": np.array([self.output_frequency]),
            "perturbation_factors": {},
            "shapes": {},
        }
        self.test_meta_perturbation = {
            "General_Sigma": np.array([self.general_sigma]),
            "Shapes": {
                "Rectangles": {
                    "Fiber_Content": self.fibre_content_rect,
                    "Num": np.array([self.num_rect]),
                },
                "Circles": {
                    "Fiber_Content": self.fibre_content_circ,
                    "Num": np.array([self.num_circ]),
                },
                "Runners": {
                    "Fiber_Content": self.fibre_content_runners,
                    "Num": np.array([self.num_runners]),
                },
            },
        }
        self.test_meta_shapes = {
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
        dd.io.save(str(os.getcwd() / self.testfolder / self.path_meta), None, compression=None)
        dd.io.save(str(os.getcwd() / self.testfolder / self.path_result), None, compression=None)
        # TODO mit leerer init
        self.test_object = hdf5db_object.HDF5Object(
            str(os.getcwd() / self.testfolder / self.path_meta),
            str(os.getcwd() / self.testfolder / self.path_result),
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
        os.remove(str(os.getcwd() / self.testfolder / self.path_meta))
        os.remove(str(os.getcwd() / self.testfolder / self.path_result))
        self.setup_test_data()

    def setup_test_data(self):
        self.test_meta["perturbation_factors"] = self.test_meta_perturbation
        self.test_meta["shapes"] = self.test_meta_shapes
        dd.io.save(
            str(os.getcwd() / self.testfolder / self.path_meta), self.test_meta, compression=None
        )
        dd.io.save(
            str(os.getcwd() / self.testfolder / self.path_result), self.test_result, compression=None
        )

    def test_load_save(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db2 = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        self.test_db.save(str(self.testfolder), "HDF5TESTDB")
        self.test_db2.load(str(self.testfolder), "HDF5TESTDB")
        self.assertTrue(self.test_db.__eq__(self.test_db2))
        os.remove(self.testfolder / Path("HDF5TESTDB.h5db"))

    def test_add_objects_from_path(self):
        self.test_result_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_result_db.add_object(self.test_object)
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        self.assertTrue(self.test_db.__eq__(self.test_result_db))

    # Select-tests
    def test_select_incorrect_entry(self):
        test_db = hdf5db_toolbox.HDF5DBToolbox()
        test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = test_db.select("path_meta", "!", "bla")
        self.assertEqual(len(test_db.hdf5_object_list), 1)
        self.assertEqual(applied, -1)
        applied = test_db.select("bla", ">", -1)
        self.assertEqual(len(test_db.hdf5_object_list), 1)
        self.assertEqual(applied, -1)
        applied = test_db.select("output_frequency", "!", 3)
        self.assertEqual(len(test_db.hdf5_object_list), 1)
        self.assertEqual(applied, -1)
        applied = test_db.select("output_frequency", "<", 0)
        self.assertEqual(len(test_db.hdf5_object_list), 1)
        self.assertEqual(applied, -1)

    # Metapath and resultpath
    def test_select_paths(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        # Illegal operator > and <
        path = self.path_meta
        str_paths = ["meta", "result"]
        for i in range(2):
            applied = self.test_db.select(
                "path_" + str_paths[i], "<", str(os.getcwd() / self.testfolder / path)
            )
            self.assertEqual(len(self.test_db.hdf5_object_list), 1)
            self.assertEqual(applied, -1)

            applied = self.test_db.select(
                "path_" + str_paths[i], ">", str(os.getcwd() / self.testfolder / path)
            )
            self.assertEqual(len(self.test_db.hdf5_object_list), 1)
            self.assertEqual(applied, -1)

            applied = self.test_db.select(
                "path_" + str_paths[i], "=", str(os.getcwd() / self.testfolder / path)
            )
            self.assertEqual(len(self.test_db.hdf5_object_list), 1)
            self.assertEqual(applied, 1)
            path = self.path_result

    def test_select_output_frequency(self):
        # Output_frequency
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = self.test_db.select("output_frequency", ">", 1)
        self.assertGreater(self.test_db.hdf5_object_list[0].output_frequency, 1)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("output_frequency", "<", 3)
        self.assertLess(self.test_db.hdf5_object_list[0].output_frequency, 3)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("output_frequency", "=", 2)
        self.assertEqual(self.test_db.hdf5_object_list[0].output_frequency, 2)
        self.assertEqual(applied, 1)
        # Output_frequency_type
        applied = self.test_db.select("output_frequency_type", ">", 98)
        self.assertGreater(self.test_db.hdf5_object_list[0].output_frequency_type, 98)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("output_frequency_type", "<", 100)
        self.assertLess(self.test_db.hdf5_object_list[0].output_frequency_type, 100)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("output_frequency_type", "=", 99)
        self.assertEqual(self.test_db.hdf5_object_list[0].output_frequency_type, 99)
        self.assertEqual(applied, 1)

    def test_select_general_sigma(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = self.test_db.select("general_sigma", ">", 99)
        # self.test_db.show_objects()
        self.assertGreater(self.test_db.hdf5_object_list[0].general_sigma, 99)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("general_sigma", "<", 100)
        self.assertLess(self.test_db.hdf5_object_list[0].general_sigma, 100)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("general_sigma", "=", 99.999)
        self.assertEqual(self.test_db.hdf5_object_list[0].general_sigma, 99.999)
        self.assertEqual(applied, 1)

    # Number of circles, rectangles, runners, shapes
    def test_select_num_shapes(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        str_shape = ["number_of_circles", "number_of_rectangles", "number_of_runners"]
        shape = [
            self.test_db.hdf5_object_list[0].number_of_circles,
            self.test_db.hdf5_object_list[0].number_of_rectangles,
            self.test_db.hdf5_object_list[0].number_of_runners,
        ]
        for i in range(3):
            applied = self.test_db.select(str_shape[i], ">", 1)
            self.assertGreater(shape[i], 1)
            self.assertEqual(applied, 1)
            applied = self.test_db.select(str_shape[i], "<", 3)
            self.assertLess(shape[i], 3)
            self.assertEqual(applied, 1)
            applied = self.test_db.select(str_shape[i], "=", 2)
            self.assertEqual(shape[i], 2)
            self.assertEqual(applied, 1)

        # Shapes
        applied = self.test_db.select("number_of_shapes", ">", 5)
        self.assertGreater(self.test_db.hdf5_object_list[0].number_of_shapes, 5)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("number_of_shapes", "<", 7)
        self.assertLess(self.test_db.hdf5_object_list[0].number_of_shapes, 7)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("number_of_shapes", "=", 6)
        self.assertEqual(self.test_db.hdf5_object_list[0].number_of_shapes, 6)
        self.assertEqual(applied, 1)

    def test_select_fibre_content(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        str_shape = [
            "fibre_content_circles",
            "fibre_content_rectangles",
            "fibre_content_runners",
        ]
        shape = [
            self.test_db.hdf5_object_list[0].fibre_content_circles,
            self.test_db.hdf5_object_list[0].fibre_content_rectangles,
            self.test_db.hdf5_object_list[0].fibre_content_runners,
        ]
        for i in range(3):
            applied = self.test_db.select(str_shape[i], ">", 0)
            self.assertGreater(np.amax(shape[i]), 0)
            self.assertEqual(applied, 1)
            applied = self.test_db.select(str_shape[i], "<", 1)
            self.assertLess(np.amin(shape[i]), 1)
            self.assertEqual(applied, 1)
            applied = self.test_db.select(str_shape[i], "=", 0.99)
            self.assertGreaterEqual(np.amax(shape[i]), 0.99)
            self.assertLessEqual(np.amin(shape[i]), 0.99)
            self.assertEqual(applied, 1)

    def test_select_shape(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        str_shape = ["circle", "rectangle", "runner"]
        fvc_shapes = [
            self.test_db.hdf5_object_list[0].fvc_circle,
            self.test_db.hdf5_object_list[0].fvc_rectangle,
            self.test_db.hdf5_object_list[0].fvc_runner,
        ]
        height_shapes = [
            self.test_db.hdf5_object_list[0].height_rectangle,
            self.test_db.hdf5_object_list[0].height_rectangle,
            self.test_db.hdf5_object_list[0].height_runner,
        ]
        width_shapes = [
            self.test_db.hdf5_object_list[0].width_rectangle,
            self.test_db.hdf5_object_list[0].width_rectangle,
            self.test_db.hdf5_object_list[0].width_runner,
        ]
        x_shapes = [
            self.test_db.hdf5_object_list[0].posx_circle,
            self.test_db.hdf5_object_list[0].posx_rectangle,
            self.test_db.hdf5_object_list[0].posx_runner,
        ]
        y_shapes = [
            self.test_db.hdf5_object_list[0].posy_circle,
            self.test_db.hdf5_object_list[0].posy_rectangle,
            self.test_db.hdf5_object_list[0].posy_runner,
        ]
        for i in range(3):
            for j in range(3):
                # Fvc
                # <
                applied = self.test_db.select("fvc_" + str_shape[i], "<", 1)
                self.assertLess(np.amax(fvc_shapes[j]), 1)
                self.assertEqual(applied, 1)
                # >
                applied = self.test_db.select("fvc_" + str_shape[i], ">", 0)
                self.assertGreater(np.amin(fvc_shapes[j]), 0)
                self.assertEqual(applied, 1)
                # =
                applied = self.test_db.select("fvc_" + str_shape[i], "=", 0.1)
                self.assertGreaterEqual(np.amax(fvc_shapes[j]), 0.5)
                self.assertLessEqual(np.amin(fvc_shapes[j]), 0.5)
                self.assertEqual(applied, 1)
                if str_shape[i] != "circle":
                    # Height
                    # <
                    applied = self.test_db.select("height_" + str_shape[i], "<", 1)
                    self.assertLess(np.amax(height_shapes[j]), 1)
                    self.assertEqual(applied, 1)
                    # >
                    applied = self.test_db.select("height_" + str_shape[i], ">", 0)
                    self.assertGreater(np.amin(height_shapes[j]), 0)
                    self.assertEqual(applied, 1)
                    # =
                    applied = self.test_db.select("height_" + str_shape[i], "=", 0.99)
                    self.assertGreaterEqual(np.amax(height_shapes[j]), 0.5)
                    self.assertLessEqual(np.amin(height_shapes[j]), 0.5)
                    self.assertEqual(applied, 1)
                    # Width
                    # <
                    applied = self.test_db.select("width_" + str_shape[i], "<", 3)
                    self.assertLess(np.amax(width_shapes[j]), 3)
                    self.assertEqual(applied, 1)
                    # >
                    applied = self.test_db.select("width_" + str_shape[i], ">", 0)
                    self.assertGreater(np.amin(width_shapes[j]), 0)
                    self.assertEqual(applied, 1)
                    # =
                    applied = self.test_db.select("width_" + str_shape[i], "=", 2.1)
                    self.assertGreaterEqual(np.amax(width_shapes[j]), 2)
                    self.assertLessEqual(np.amin(width_shapes[j]), 2)
                    self.assertEqual(applied, 1)
                # Position x
                # <
                applied = self.test_db.select("posx_" + str_shape[i], "<", 31)
                self.assertLess(np.amax(x_shapes[j]), 31)
                self.assertEqual(applied, 1)
                # >
                applied = self.test_db.select("posx_" + str_shape[i], ">", 4)
                self.assertGreater(np.amin(x_shapes[j]), 4)
                self.assertEqual(applied, 1)
                # =
                applied = self.test_db.select("posx_" + str_shape[i], "=", 5)
                self.assertGreaterEqual(np.amax(x_shapes[j]), 5)
                self.assertLessEqual(np.amin(x_shapes[j]), 5)
                self.assertEqual(applied, 1)
                # Position y
                # <
                applied = self.test_db.select("posy_" + str_shape[i], "<", 40)
                self.assertLess(np.amax(y_shapes[j]), 40)
                self.assertEqual(applied, 1)
                # >
                applied = self.test_db.select("posy_" + str_shape[i], ">", 4)
                self.assertGreater(np.amin(y_shapes[j]), 4)
                self.assertEqual(applied, 1)
                # =
                applied = self.test_db.select("posy_" + str_shape[i], "=", 30)
                self.assertGreaterEqual(np.amax(y_shapes[j]), 10)
                self.assertLessEqual(np.amin(y_shapes[j]), 10)
                self.assertEqual(applied, 1)

        # Position lower left x of runner
        # <
        applied = self.test_db.select("pos_lower_leftx_runner", "<", 30)
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), 30
        )
        self.assertEqual(applied, 1)
        # >
        applied = self.test_db.select("pos_lower_leftx_runner", ">", 10)
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), 10
        )
        self.assertEqual(applied, 1)
        # =
        applied = self.test_db.select("pos_lower_leftx_runner", "=", 14.2)
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), 15
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_leftx_runner), 15
        )
        self.assertEqual(applied, 1)
        # Position lower left y of runner
        # <
        applied = self.test_db.select("pos_lower_lefty_runner", "<", 40)
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), 40
        )
        self.assertEqual(applied, 1)
        # >
        applied = self.test_db.select("pos_lower_lefty_runner", ">", 20)
        self.assertGreater(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), 20
        )
        self.assertEqual(applied, 1)
        # =
        applied = self.test_db.select("pos_lower_lefty_runner", "=", 21.6)
        self.assertGreaterEqual(
            np.amax(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), 25
        )
        self.assertLessEqual(
            np.amin(self.test_db.hdf5_object_list[0].pos_lower_lefty_runner), 25
        )
        self.assertEqual(applied, 1)

    def test_select_fillinglevel(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = self.test_db.select("avg_level", ">", 0)
        self.assertGreater(self.test_db.hdf5_object_list[0].avg_level, 0)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("avg_level", "<", 1)
        self.assertLess(self.test_db.hdf5_object_list[0].avg_level, 1)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("avg_level", "=", self.test_object.avg_level)
        self.assertEqual(
            self.test_db.hdf5_object_list[0].avg_level, self.test_object.avg_level
        )
        self.assertEqual(applied, 1)

    def test_select_age(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = self.test_db.select("age", ">", "2019-08-30_12-23-58")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime("2019-08-30_12-23-58", "%Y-%m-%d_%H-%M-%S"),
        )
        self.assertEqual(applied, 1)
        applied = self.test_db.select("age", "<", "2019-09-29_12-24-59")
        self.assertLess(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime("2019-09-29_12-24-59", "%Y-%m-%d_%H-%M-%S"),
        )
        self.assertEqual(applied, 1)
        self.temp = self.age
        applied = self.test_db.select("age", "=", "2019-08-30_12-23-59")
        self.assertEqual(
            self.test_db.hdf5_object_list[0].age,
            datetime.strptime("2019-08-30_12-23-59", "%Y-%m-%d_%H-%M-%S"),
        )
        self.assertEqual(applied, 1)

    def test_select_num_sensors(self):
        self.test_db = hdf5db_toolbox.HDF5DBToolbox()
        self.test_db.add_objects_from_path(str(os.getcwd() / self.testfolder))
        applied = self.test_db.select("number_of_sensors", ">", 3)
        self.assertGreater(self.test_db.hdf5_object_list[0].number_of_sensors, 3)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("number_of_sensors", "<", 6)
        self.assertLess(self.test_db.hdf5_object_list[0].number_of_sensors, 6)
        self.assertEqual(applied, 1)
        applied = self.test_db.select("number_of_sensors", "=", 5)
        self.assertEqual(self.test_db.hdf5_object_list[0].number_of_sensors, 5)
        self.assertEqual(applied, 1)

    def tearDown(self):
        shutil.rmtree(os.getcwd() / self.testfolder)


if __name__ == "__main__":
    unittest.main()
