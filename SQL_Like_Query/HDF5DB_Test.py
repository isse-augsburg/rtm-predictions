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
        # Metadata
        self.output_frequency_type = np.array([99])
        self.general_sigma = np.array([99.999])
        self.num_rect = np.array([2])
        self.fibre_content_rect = np.array([0.09, 0.99])
        self.num_circ = np.array([2])
        self.fibreContentCirc = np.array([0.08, 0.88])
        self.num_runners = np.array([2])

        self.fibre_content_runners = np.array([-0.4, -0.9])
        self.fvc_rect = np.array([0.99, 0.1])
        self.height_rect = np.array([0.09, 0.99])
        self.width_rect = np.array((1, 2.1))
        self.posx_rect = np.array((5, 30))
        self.posy_rect = np.array((5, 30))
        self.fvc_circ = np.array((0.89, 0.25))
        self.radius_circ = np.array((0.09, 0.99))
        self.posy_circ = np.array((8, 26))
        self.posy_circ = np.array((50, 33))
        self.fvc_runners = np.array((0.22, 0.73))
        self.height_runners = np.array((0.4, 1.9))
        self.width_runners = np.array((1.6, 2.7))
        self.posx_runners = np.array((30.6, 22.5))
        self.posy_runners = np.array((15, 23, 8))
        self.pos_lower_leftx_runners = np.array((14.2, 21.9))
        self.pos_lower_lefty_runners = np.array((30.4, 21.6))

        # Results
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
            "perturbation_factors": {
                "General_Sigma": self.general_sigma,
                "Shapes": {
                    "Rectangles": {
                        "Fiber_Content": self.fibre_content_rect,
                        "Num": self.num_rect,
                    },
                    "Circles": {
                        "Fiber_Content": self.fibreContentCirc,
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
                    "posX": self.posy_circ,
                    "posY": self.posy_circ,
                },
                "Runner": {
                    "fvc": self.fvc_runners,
                    "height": self.height_runners,
                    "width": self.width_runners,
                    "posX": self.posx_runners,
                    "posY": self.posy_runners,
                    "posLowerLeftX": self.pos_lower_leftx_runners,
                    "posLowerLeftY": self.pos_lower_lefty_runners,
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

        dd.io.save("2019-08-30_12-23-59_test_meta_data.hdf5", None, compression=None)
        dd.io.save("2019-08-30_12-23-59_test_RESULT.erfh5", None, compression=None)
        self.test_object = HDF5Object(
            "2019-08-30_12-23-59_test_meta_data.hdf5",
            "2019-08-30_12-23-59_test_RESULT.erfh5",
        )
        self.test_object.meta_path = (
            os.getcwd()
            / Path("Tests")
            / Path("2019-08-30_12-23-59_test_meta_data.hdf5")
        )
        self.test_object.output_frequency_type = self.output_frequency_type
        self.test_object.general_sigma = self.general_sigma
        self.test_object.number_of_rectangles = self.num_rect
        self.test_object.fibre_content_rectangles = self.fibre_content_rect
        self.test_object.number_of_circles = self.num_circ
        self.test_object.fibre_content_circles = self.fibreContentCirc
        self.test_object.number_of_runners = self.num_runners
        self.test_object.number_of_sensors = np.ma.size(self.res, 1)
        self.test_object.number_of_shapes = (
            self.num_circ + self.num_rect + self.num_runners
        )
        self.test_object.fibre_content_runners = self.fibre_content_runners
        self.test_object.fvc_rectangle = self.fvc_rect
        self.test_object.height_rectangle = self.height_rect
        self.test_object.width_rectangle = self.width_rect
        self.test_object.posx_rectangle = self.posx_rect
        self.test_object.posy_rectangle = self.posy_rect
        self.test_object.fvc_circle = self.fvc_circ
        self.test_object.radius_circle = self.radius_circ
        self.test_object.posx_circle = self.posy_circ
        self.test_object.posy_circle = self.posy_circ
        self.test_object.fvc_runner = self.fvc_runners
        self.test_object.height_runner = self.height_runners
        self.test_object.width_runner = self.width_runners
        self.test_object.posx_runner = self.posx_runners
        self.test_object.posy_runner = self.posy_runners
        self.test_object.pos_lower_leftx_runner = self.pos_lower_leftx_runners
        self.test_object.pos_lower_lefty_runner = self.pos_lower_lefty_runners

        self.test_object.result_path = (
            os.getcwd() / Path("Tests") / Path("2019-08-30_12-23-59_test_RESULT.erfh5")
        )
        self.test_object.avg_level = np.sum(self.state999) / len(self.state999)
        self.test_object.age = datetime.strptime(
            "2019-08-30_12-23-59", "%Y-%m-%d_%H-%M-%S"
        )
        os.remove("2019-08-30_12-23-59_test_meta_data.hdf5")
        os.remove("2019-08-30_12-23-59_test_RESULT.erfh5")

    def setup_test_data(self):
        dd.io.save(
            "Tests/2019-08-30_12-23-59_test_meta_data.hdf5",
            self.test_meta,
            compression=None,
        )
        dd.io.save(
            "Tests/2019-08-30_12-23-59_test_RESULT.erfh5",
            self.test_result,
            compression=None,
        )

    def test_load_save(self):
        self.setup_test_data()
        self.test_db = HDF5DB()
        self.test_db2 = HDF5DB()
        self.test_db.add_objects_from_path("Tests")
        self.test_db.save("Tests", "HDF5TESTDB")
        self.test_db2.load("Tests", "HDF5TESTDB")
        self.assertTrue(self.test_db.__eq__(self.test_db2))
        os.remove(Path("Tests") / Path("HDF5TESTDB.h5db"))

    def test_add_objects_from_path(self):
        self.setup_test_data()
        self.test_result_db = HDF5DB()
        self.test_db = HDF5DB()
        self.test_result_db.add_object(self.test_object)
        self.test_db.add_objects_from_path("Tests")
        self.assertTrue(self.test_db.__eq__(self.test_result_db))

    def test_select(self):
        self.setup_test_data()
        self.test_db = HDF5DB()
        self.test_db.add_objects_from_path("Tests")
        print(self.test_db.show_objects())
        # Fillinglevel
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
        # Position y of circle
        self.temp = np.amax(self.test_object.posy_circle) + 1
        self.test_db.select("posy_circle", self.temp, "<")
        self.assertLess(
            np.amax(self.test_db.hdf5_object_list[0].posy_circle), self.temp
        )
        # Metapath and illegal operator >
        self.temp = len(self.test_db.hdf5_object_list)
        self.test_db.select("path_meta", "bla", "<")
        self.assertEqual(len(self.test_db.hdf5_object_list), self.temp)
        # Number of sensors
        self.temp = self.test_object.number_of_sensors - 1
        self.test_db.select("number_of_sensors", self.temp, ">")
        self.assertGreater(
            self.test_db.hdf5_object_list[0].number_of_sensors, self.temp
        )
        os.remove(Path("Tests") / Path("2019-08-30_12-23-59_test_meta_data.hdf5"))
        os.remove(Path("Tests") / Path("2019-08-30_12-23-59_test_RESULT.erfh5"))

if __name__ == "__main__":
    unittest.main()