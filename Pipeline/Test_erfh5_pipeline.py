import unittest

from Pipeline import erfh5_pipeline as pipeline, data_loaders_IMG as dli, \
    data_gather as dg


class TestShapes(unittest.TestCase):
    def setUp(self):
        self.paths = [r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-11_15-14-48_100p']
        self.setup_test_train_validation_split()

    def setup_test_train_validation_split(self):
        num_samples = 100 # * Number of frames
        self.num_validation_samples = 400
        self.num_test_samples = 400
        self.generator = pipeline.ERFH5DataGenerator(data_paths=self.paths, num_validation_samples=self.num_validation_samples,
                                                num_test_samples=self.num_test_samples, max_queue_length=8096,
                                                data_processing_function=dli.get_sensordata_and_flowfront,
                                                data_gather_function=dg.get_filelist_within_folder, num_workers=10)

    def test_test_set(self):
        # self.setup_test_train_validation_split()
        self.assertEqual(self.num_test_samples, len(self.generator.get_test_samples()))

    def test_validation_set(self):
        # self.setup_test_train_validation_split()
        self.assertEqual(self.num_validation_samples, len(self.generator.get_validation_samples()))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()