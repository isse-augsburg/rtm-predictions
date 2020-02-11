import logging
import unittest

import Resources.testing as resources
from Pipeline.data_loaders_IMG import DataloaderImages
from Pipeline.resampling import get_fixed_number_of_indices


class TestDataLoaderIMG(unittest.TestCase):
    def setUp(self):
        self.img_cache_dirname = resources.data_loader_img_file

    def test_get_fixed_number_of_elements_and_their_indices_from_various_sized_list(
            self):
        for i in [2, 10, 20, 33, 100]:
            for j in [2, 5, 8, 10, 20]:
                self.create_list_and_test(i, j)

    def create_list_and_test(self, list_length, n_elements):
        logger = logging.getLogger(__name__)
        logger.info(f'list len: {list_length}, required elements: {n_elements}')
        if n_elements > list_length:
            return
        x = get_fixed_number_of_indices(
            list_length, n_elements)

        self.assertEqual(len(x), n_elements)

    def test_ignore_useless(self):
        dl = DataloaderImages()
        dl2 = DataloaderImages(ignore_useless_states=False)
        res = dl.get_sensordata_and_flowfront(resources.test_useless_file)
        res2 = dl2.get_sensordata_and_flowfront(resources.test_useless_file)
        self.assertEqual(len(res2) - len(res), 17)  # since 17 frames are ignored in this file

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
