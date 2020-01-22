import logging
import unittest

import Resources.testing as resources
from Pipeline.resampling import get_fixed_number_of_indices


class TestDataLoaderIMG(unittest.TestCase):
    def setUp(self):
        self.img_cache_dirname = resources.data_loader_img_file

    # @unittest.skip("Currently not working")
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

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
