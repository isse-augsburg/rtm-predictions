import logging
import pickle
import re
import shutil
import socket
import unittest
from pathlib import Path

from Pipeline.erfh5_pipeline import transform_list_of_linux_paths_to_windows
from model_trainer_sensor_to_flow import SensorTrainer
import Tests.TestResources as resources


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [resources.test_training_src_dir / '2019-07-11_15-14-48_100p']
        self.load_datasets_path = resources.test_training_src_dir / '2019-09-06_15-44-58_63_sensors'
        self.expected_num_epochs_during_training = 5

    def test_training(self):
        st = SensorTrainer(data_source_paths=self.training_data_paths,
                           save_datasets_path=self.training_save_path,
                           epochs=self.expected_num_epochs_during_training)
        st.run_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / 'output.log') as f:
            content = f.read()
            epochs = re.findall('Mean Loss on Eval',  content)
            self.assertEqual(self.expected_num_epochs_during_training, len(epochs))

    def tearDown(self) -> None:
        shutil.rmtree(self.training_save_path)
        logging.shutdown()


class TestTrainingWithFixedDataset(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [resources.test_training_src_dir / '2019-07-11_15-14-48_100p']
        self.load_datasets_path = resources.test_training_src_dir / '2019-09-06_15-44-58_63_sensors'
        self.expected_num_epochs_during_training = 5
        self.num_validation_samples = 2000
        self.num_test_samples = 2000
        st = SensorTrainer(data_source_paths=[],
                           save_datasets_path=self.training_save_path,
                           load_datasets_path=self.load_datasets_path,
                           epochs=self.expected_num_epochs_during_training,
                           num_test_samples=self.num_test_samples,
                           num_validation_samples=self.num_validation_samples)
        self.gen = st.create_datagenerator(save_path=None, test_mode=False)


    def test_len_validation_set(self):
        self.assertEqual(self.num_validation_samples, len(self.gen.validation_list))

    def test_len_test_set(self):
        self.assertEqual(self.num_test_samples, len(self.gen.test_list))

    def test_gen_validation_fnames(self):
        with open(self.load_datasets_path / "validation_set.p", 'rb') as f:
            self.validation_fnames = pickle.load(f)
        if socket.gethostname() == 'swtse130':
            self.validation_fnames = transform_list_of_linux_paths_to_windows(self.validation_fnames)
        self.assertEqual(self.validation_fnames, self.gen.validation_fnames)

    def test_gen_test_fnames(self):
        with open(self.load_datasets_path / "test_set.p", 'rb') as f:
            self.test_fnames = pickle.load(f)
        if socket.gethostname() == 'swtse130':
            self.test_fnames = transform_list_of_linux_paths_to_windows(self.test_fnames)
        self.assertEqual(self.test_fnames, self.gen.test_fnames)

    def test_gen_training_fnames(self):
        with open(self.load_datasets_path / "training_set.p", 'rb') as f:
            self.training_fnames = pickle.load(f)
        if socket.gethostname() == 'swtse130':
            self.training_fnames = transform_list_of_linux_paths_to_windows(self.training_fnames)
        self.assertEqual(sorted(self.training_fnames), sorted(self.gen.paths))


if __name__ == '__main__':
    unittest.main()