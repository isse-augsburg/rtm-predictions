import logging
import re
import shutil
import unittest
from pathlib import Path

from model_trainer_sensor_to_flow import SensorTrainer
import Tests.resources_for_testing as resources


class TestEval(unittest.TestCase):
    def setUp(self):
        self.eval_path = resources.test_eval_dir
        self.eval_path_to_delete = self.eval_path / 'eval_on_test_set'

        self.expected_loss = 0.0026  # 0.0022 for 10; 0.0026 for 100; 0.0035 for 2000
        self.num_test_samples = 100
        self.expected_num_frames = self.num_test_samples * 3
        self.expected_num_epochs_during_training = 5

    def test_eval(self):
        self.st = SensorTrainer(data_source_paths=[],
                           save_datasets_path=self.eval_path_to_delete,
                           load_datasets_path=self.eval_path_to_delete,
                           num_test_samples=self.num_test_samples)
        self.st.inference_on_test_set(self.eval_path)
        with open(self.eval_path_to_delete / 'test_output.log') as f:
            content = f.read()
            loss = float(re.findall('\d+.\d+', re.findall('Eval:   \d+\.\d+',  content)[0])[0])
            self.assertEqual(loss, self.expected_loss)
        img_path = self.eval_path_to_delete / 'images'
        list_all_imgs = list(img_path.glob('**/*.jpg'))
        self.assertEqual(len(list_all_imgs), self.expected_num_frames)

    def tearDown(self) -> None:
        self.st.test_data_generator.end_threads()
        shutil.rmtree(self.eval_path_to_delete)
        logging.shutdown()


if __name__ == '__main__':
    unittest.main()