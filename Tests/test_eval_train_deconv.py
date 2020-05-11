import logging
import os
import re
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

import Resources.testing as test_resources
from Models.erfh5_DeconvModel import DeconvModelEfficient
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loaders_IMG import DataloaderImages
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import SensorToFlowfrontEvaluator


class TestEval(unittest.TestCase):
    def setUp(self):
        self.eval_path = test_resources.test_eval_dir
        self.test_src_dir = [
            test_resources.test_training_src_dir / '2019-07-11_15-14-48_5p']
        self.test_split_dir = test_resources.test_split_path
        self.eval_output_path = test_resources.test_eval_output_path
        self.checkpoint = test_resources.test_checkpoint
        self.training_save_path = test_resources.test_training_out_dir
        self.test_save_load_out_dir = test_resources.test_save_dataset_path
        self.test_save_load_out_dir.mkdir(exist_ok=True)
        self.num_test_samples = 10
        if self.num_test_samples == 10:
            self.expected_loss = 0.0132
        if self.num_test_samples == 100:
            self.expected_loss = 0.0026
        if self.num_test_samples == 2000:
            self.expected_loss = 0.0035
        self.expected_num_frames = self.num_test_samples * 3
        self.expected_num_epochs_during_training = 1

    def test_eval(self):
        dl = DataloaderImages((149, 117), ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.eval_output_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=2,
            train_print_frequency=10,
            epochs=self.expected_num_epochs_during_training,
            num_workers=10,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer: SensorToFlowfrontEvaluator(
                self.eval_output_path / "eval_on_test_set",
                skip_images=False,
                summary_writer=summary_writer
            ),
            data_root=test_resources.test_src_dir,
        )

        st.inference_on_test_set(
            self.eval_output_path,
            self.checkpoint,
            classification_evaluator_function=lambda summary_writer: SensorToFlowfrontEvaluator(
                self.eval_output_path / "eval_on_test_set",
                skip_images=False,
                summary_writer=summary_writer
            )
        )

        with open(self.eval_output_path / "eval_on_test_set" / "test_output.log") as f:
            content = f.read()
            loss = float(re.findall(r'\d+.\d+', re.findall(r'Eval: \d+\.\d+', content)[0])[0])
            self.assertEqual(np.round(loss, 4), self.expected_loss)
        img_path = self.eval_output_path / 'eval_on_test_set' / 'images'
        list_all_imgs = list(img_path.glob('**/*.jpg'))
        self.assertEqual(len(list_all_imgs), self.expected_num_frames)

    def test_training(self):
        num_epochs = 2
        dl = DataloaderImages((149, 117),
                              ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.training_save_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=16,
            train_print_frequency=10,
            epochs=num_epochs,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer:
            SensorToFlowfrontEvaluator(summary_writer=summary_writer),
            data_root=test_resources.test_src_dir,
        )
        st.start_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / 'output.log') as f:
            content = f.read()
            epochs = re.findall('Mean Loss on Eval', content)
            self.assertEqual(num_epochs, len(epochs))
            # Check if steps are growing / if there are doubled steps in the output
            steps = [int(re.findall(r'\d+', x)[0]) for x in re.findall(r'Duration of step.+\d:', content)]
            self.assertEqual(len(set(steps)), len(steps))

    @unittest.skip("Need to update nvidia driver to enable apex again.")
    def test_training_apex(self):
        num_epochs = 2
        dl = DataloaderImages((149, 117),
                              ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.training_save_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=16,
            train_print_frequency=10,
            epochs=num_epochs,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer:
            SensorToFlowfrontEvaluator(summary_writer=summary_writer),
            use_mixed_precision=True,
            data_root=test_resources.test_src_dir,
        )
        st.start_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / 'output.log') as f:
            content = f.read()
            epochs = re.findall('Mean Loss on Eval', content)
            self.assertEqual(num_epochs, len(epochs))
            # Check if steps are growing / if there are doubled steps in the output
            steps = [int(re.findall(r'\d+', x)[0]) for x in re.findall(r'Duration of step.+\d:', content)]
            self.assertEqual(len(set(steps)), len(steps))

    def test_save_load_training(self):
        num_epochs = 2
        dl = DataloaderImages((149, 117),
                              ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.training_save_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=16,
            train_print_frequency=10,
            epochs=num_epochs,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer: SensorToFlowfrontEvaluator(
                summary_writer=summary_writer),
            data_root=test_resources.test_src_dir,
        )
        st.start_training()

        num_epochs = 2
        dl = DataloaderImages((149, 117),
                              ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.training_save_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=16,
            train_print_frequency=10,
            epochs=num_epochs,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer: SensorToFlowfrontEvaluator(
                summary_writer=summary_writer),
            data_root=test_resources.test_src_dir,
        )
        st.start_training()

    def test_training_load_optimizer(self):
        dl = DataloaderImages((149, 117),
                              ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.training_save_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=16,
            train_print_frequency=10,
            epochs=self.expected_num_epochs_during_training,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            loss_criterion=torch.nn.BCELoss(),
            optimizer_path=self.checkpoint,
            classification_evaluator_function=lambda summary_writer:
            SensorToFlowfrontEvaluator(summary_writer=summary_writer),
            data_root=test_resources.test_src_dir,
        )
        st.start_training()
        after = len(st.optimizer.state.keys())
        """ Optimizer has now more than 0 states, therefore was loaded """
        self.assertGreater(after, 0)

    def test_eval_preparation(self):
        dl = DataloaderImages((149, 117), ignore_useless_states=False)
        st = ModelTrainer(
            lambda: DeconvModelEfficient(),
            self.test_src_dir,
            self.eval_output_path,
            load_datasets_path=self.test_split_dir,
            cache_path=None,
            batch_size=2,
            train_print_frequency=10,
            epochs=0,
            num_workers=4,
            num_validation_samples=2,
            num_test_samples=self.num_test_samples,
            data_processing_function=dl.get_sensordata_and_flowfront,
            data_gather_function=get_filelist_within_folder_blacklisted,
            classification_evaluator_function=lambda summary_writer:
            SensorToFlowfrontEvaluator(summary_writer=summary_writer),
            data_root=test_resources.test_src_dir,
        )
        st.start_training()
        dirs = [e for e in self.eval_output_path.iterdir() if e.is_dir()]
        code_dir = dirs[0] / 'rtm-predictions'
        slurm_script = dirs[0] / 'run_model_eval.sh'
        self.assertTrue(os.path.isdir(code_dir))
        self.assertTrue(os.path.isfile(slurm_script))
        with open(slurm_script) as f:
            lines = f.read().splitlines()
            tokens = lines[-1].split()
            self.assertEqual(dirs[0], Path(tokens[-3]))
        st.writer.flush()
        st.writer.close()

    def tearDown(self) -> None:
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        if self.eval_output_path.exists():
            shutil.rmtree(self.eval_output_path)
        if self.training_save_path.exists():
            shutil.rmtree(self.training_save_path)
        if self.test_save_load_out_dir.exists():
            shutil.rmtree(self.test_save_load_out_dir)


if __name__ == '__main__':
    unittest.main()
