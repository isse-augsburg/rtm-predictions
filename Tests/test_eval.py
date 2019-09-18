import logging
import re
import shutil
import unittest

import numpy as np

import Tests.resources_for_testing as resources
from model_trainer_sensor_to_flow import SensorTrainer


class TestEval(unittest.TestCase):
    def setUp(self):
        self.eval_path = resources.test_eval_dir
        self.eval_output_path = resources.test_eval_output_path

        self.num_test_samples = 10
        if self.num_test_samples == 10:
            self.expected_loss = 0.0022
        if self.num_test_samples == 100:
            self.expected_loss = 0.0026
        if self.num_test_samples == 2000:
            self.expected_loss = 0.0035
        self.expected_num_frames = self.num_test_samples * 3
        self.expected_num_epochs_during_training = 5

    # @unittest.skipIf(resources.running_in_docker(),
    #                  "Skipped only on runner / in docker: not enough memory: currently not working")
    def test_eval(self):
        self.st = SensorTrainer(
            data_source_paths=[],
            save_datasets_path=self.eval_output_path,
            load_datasets_path=self.eval_output_path,
            num_test_samples=self.num_test_samples)
        self.st.inference_on_test_set(source_path=self.eval_path, output_path=self.eval_output_path)
        with open(self.eval_output_path / 'eval_on_test_set' / 'test_output.log') as f:
            content = f.read()
            loss = float(re.findall(r'\d+.\d+', re.findall(r'Eval: \d+\.\d+', content)[0])[0])
            self.assertEqual(np.round(loss, 4), self.expected_loss)
        img_path = self.eval_output_path / 'eval_on_test_set' / 'images'
        list_all_imgs = list(img_path.glob('**/*.jpg'))
        self.assertEqual(len(list_all_imgs), self.expected_num_frames)

    def tearDown(self) -> None:
        self.st.test_data_generator.end_threads()
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.eval_output_path)


if __name__ == '__main__':
    unittest.main()
