import logging
import re
import shutil
import unittest

import numpy as np
from Models.erfh5_DeconvModel import DeconvModel_efficient
from Pipeline.Utils.looping_strategies import DataLoaderListLoopingStrategy
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Trainer.evaluation import SensorToFlowfrontEvaluator
from Pipeline.data_loaders_IMG import DataloaderImages
from general_model_trainer import ModelTrainer
import Tests.resources_for_testing as resources


class TestEval(unittest.TestCase):
    def setUp(self):
        self.eval_path = resources.test_eval_dir
        self.test_src_dir = resources.test_src_dir
        self.eval_output_path = resources.test_eval_output_path
        self.checkpoint = resources.test_checkpoint
        self.num_test_samples = 10
        if self.num_test_samples == 10:
            self.expected_loss = 0.0013
        if self.num_test_samples == 100:
            self.expected_loss = 0.0026
        if self.num_test_samples == 2000:
            self.expected_loss = 0.0035
        self.expected_num_frames = self.num_test_samples * 3
        self.expected_num_epochs_during_training = 5

    def test_eval(self):
        dl = DataloaderImages((149, 117))
        self.st = ModelTrainer(
            DeconvModel_efficient(),
            [self.test_src_dir],
            self.eval_output_path,
            load_datasets_path=None,
            cache_path=None,
            batch_size=2,
            eval_freq=10,
            train_print_freq=10,
            epochs=1000,
            num_workers=10,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            looping_strategy=DataLoaderListLoopingStrategy(2),
        )

        self.st.inference_on_test_set(
            self.eval_output_path,
            SensorToFlowfrontEvaluator(
                self.eval_output_path / "eval_on_test_set", skip_images=False
            ), self.checkpoint
        )

        with open(self.eval_output_path / "eval_on_test_set" / "test_output.log") as f:
            content = f.read()
            loss = float(re.findall(r'\d+.\d+', re.findall(r'Eval: \d+\.\d+', content)[0])[0])
            self.assertEqual(np.round(loss, 4), self.expected_loss)
        img_path = self.eval_output_path / 'eval_on_test_set' / 'images'
        list_all_imgs = list(img_path.glob('**/*.jpg'))
        self.assertEqual(len(list_all_imgs), self.expected_num_frames)

    def tearDown(self) -> None:
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.eval_output_path)


if __name__ == '__main__':
    unittest.main()
