import pickle
import shutil
import unittest
from pathlib import Path

from Pipeline import erfh5_pipeline as pipeline, data_loaders_IMG as dli, \
    data_gather as dg
import Tests.resources_for_testing as resources


class TestERFH5Pipeline(unittest.TestCase):
    def setUp(self):
        self.p = resources.test_pipeline_dir / '2019-07-11_15-14-48_100p'
        self.paths = [self.p]
        self.cached_files = []
        for i in range(100):
            self.cached_files.append(self.p / Path(f'{i}/2019-07-11_15-14-48_{i}_RESULT.erfh5'))

    def test_caching(self):
        self.num_validation_samples = 400
        self.num_test_samples = 400
        self.caching_path = resources.test_caching_dir
        self.generator = pipeline.ERFH5DataGenerator(data_paths=self.paths,
                                                     num_validation_samples=self.num_validation_samples,
                                                     num_test_samples=self.num_test_samples,
                                                     max_queue_length=1,
                                                     data_processing_function=dli.get_sensordata_and_flowfront,
                                                     data_gather_function=dg.get_filelist_within_folder,
                                                     num_workers=1,
                                                     cache_path=self.caching_path)
        p = Path(self.caching_path)
        filelists = p / 'filelists'
        with open(filelists / '2019-07-11_15-14-48_100p', 'rb') as f:
            fl = pickle.load(f)
        self.assertEqual(sorted(fl), sorted(self.cached_files))

    def tearDown(self):
        self.generator.end_threads()
        shutil.rmtree(self.caching_path)


class Test_dataset_split(unittest.TestCase):
    def setUp(self):
        self.paths = [resources.test_pipeline_dir / '2019-07-11_15-14-48_100p']
        num_samples = 100  # <- Number of files * Number of frames
        self.num_validation_samples = 400
        self.num_test_samples = 400
        self.generator = pipeline.ERFH5DataGenerator(data_paths=self.paths,
                                                     num_validation_samples=self.num_validation_samples,
                                                     num_test_samples=self.num_test_samples,
                                                     max_queue_length=8096,
                                                     data_processing_function=dli.get_sensordata_and_flowfront,
                                                     data_gather_function=dg.get_filelist_within_folder,
                                                     num_workers=10)

    def test_test_set(self):
        self.assertEqual(self.num_test_samples, len(self.generator.get_test_samples()))

    def test_validation_set(self):
        self.assertEqual(self.num_validation_samples, len(self.generator.get_validation_samples()))

    def tearDown(self):
        self.generator.end_threads()


if __name__ == '__main__':
    unittest.main()
