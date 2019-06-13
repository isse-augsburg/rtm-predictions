import unittest
from pathlib import Path
from time import time
from Pipeline.data_loaders_IMG import get_images_of_flow_front_and_permeability_map, \
    get_fixed_number_of_elements_and_their_indices_from_various_sized_list


class TestDataLoaderIMG(unittest.TestCase):
    def setUp(self):
        self.fn = Path(r'X:\s\t\stiebesi\code\tests\data_loader_IMG\2019-06-05_15-30-52_0_RESULT.erfh5')

    # def test_get_fixed_number_of_elements_and_their_indices_from_various_sized_list(self):
    #     for i in [2, 10, 33, 100]:
    #         for j in [2, 5, 8, 10]:
    #             self.create_list_and_test(i, j)
    #
    # def create_list_and_test(self, list_length, n_elements):
    #     if n_elements > list_length:
    #         return
    #     get_fixed_number_of_elements_and_their_indices_from_various_sized_list(list(range(list_length)), n_elements)

    def test_get_images_of_flow_front_and_permeability_map(self):
        get_images_of_flow_front_and_permeability_map(self.fn)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
