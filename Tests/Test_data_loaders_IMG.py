import logging
import unittest
from pathlib import Path
from time import time, sleep
from Pipeline.data_loaders_IMG import get_images_of_flow_front_and_permeability_map, \
    get_fixed_number_of_elements_and_their_indices_from_various_sized_list
import Tests.resources_for_testing as Resources



class TestDataLoaderIMG(unittest.TestCase):
    def setUp(self):
        self.fn = Resources.data_loader_img_file

    def test_get_fixed_number_of_elements_and_their_indices_from_various_sized_list(self):
        for i in [2, 10, 20, 33, 100]:
            for j in [2, 5, 8, 10, 20]:
                self.create_list_and_test(i, j)

    def create_list_and_test(self, list_length, n_elements):
        logger = logging.getLogger(__name__)
        logger.info(list_length, n_elements)
        if n_elements > list_length:
            return
        x = get_fixed_number_of_elements_and_their_indices_from_various_sized_list(list(range(list_length)), n_elements)
        self.assertEqual(len(x), n_elements)

    def test_get_images_of_flow_front_and_permeability_map(self):
        wanted_num = [20]
        for w in wanted_num:
            get_images_of_flow_front_and_permeability_map(self.fn, wanted_num=w)
            list_of_pngs = list((self.fn.parent / 'img_cache').glob('**/*'))
            self.assertIn(self.fn.parent / 'img_cache' / 'fiber_fraction.png', list_of_pngs)
            self.assertEqual(len(list_of_pngs), w + 1)
            self.tearDown()

    def tearDown(self):
        all_files = (self.fn.parent / 'img_cache').glob('**/*')
        [x.unlink() for x in all_files if x.is_file()]


if __name__ == '__main__':
    unittest.main()
