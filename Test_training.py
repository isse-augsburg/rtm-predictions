import logging
import re
import shutil
import unittest
from pathlib import Path

from model_trainer_sensor_to_flow import SensorTrainer


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = Path(r'X:\s\t\stiebesi\code\tests\training')
        self.training_data_paths = [
            Path(r'X:\s\t\stiebesi\code\tests\training_data_source\2019-07-11_15-14-48_100p')]
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


if __name__ == '__main__':
    unittest.main()