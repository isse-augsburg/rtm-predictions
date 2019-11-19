import os
import unittest
from pathlib import Path

import h5py

from HDF5DB.hdf5_writer import create_hdf5, write_dict_to_hdf5
from Tests import resources_for_testing


class TestPyWriter(unittest.TestCase):
    def setUp(self):
        self.testfolder = Path(resources_for_testing.test_out_dir)
        self.filename = Path("test_for_hdf5writer.hdf5db")
        self.filename2 = Path("test_for_hdf5writer2.hdf5db")
        self.testdata = \
            {
                "General_Sigma": .001,
                "Shapes":
                    {
                        "Rectangles":
                            {
                                "Num": 1,
                                "Fiber_Content":
                                    [.7, .8]
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

        self.testdata2 = \
            {
                "Test": 1
            }

    def test_hdf5_writing(self):
        # For 2 keys
        hdf5 = create_hdf5(str(self.testfolder / self.filename))
        write_dict_to_hdf5(hdf5, self.testdata)
        test = h5py.File(str(self.testfolder / self.filename), "r")
        self.assertEqual(self.testdata.keys(), test.keys())

        # For 1 key
        hdf5 = create_hdf5(str(self.testfolder / self.filename2))
        write_dict_to_hdf5(hdf5, self.testdata2)
        test = h5py.File(str(self.testfolder / self.filename2), "r")
        self.assertEqual(self.testdata2.keys(), test.keys())
        test.close()

    def tearDown(self):
        os.remove(str(self.testfolder / self.filename))
        os.remove(str(self.testfolder / self.filename2))


if __name__ == '__main__':
    unittest.main()
