import logging
import re
import shutil
import unittest

import torch

import Resources.testing as resources
from Models.erfh5_ConvModel import DrySpotModel
from Pipeline import data_gather as dg
from Pipeline.data_loader_dryspot import DataloaderDryspots
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator


class TestTrainingDryspotFF(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [resources.test_training_src_dir / 'dry_spot_from_ff']
        self.expected_num_epochs_during_training = 1
        dlds = DataloaderDryspots(image_size=(143, 111), ignore_useless_states=False)
        self.dt = ModelTrainer(
            lambda: DrySpotModel(),
            data_source_paths=self.training_data_paths,
            save_path=self.training_save_path,
            batch_size=10,
            epochs=self.expected_num_epochs_during_training,
            num_validation_samples=5,
            num_test_samples=5,
            data_gather_function=dg.get_filelist_within_folder,
            data_processing_function=dlds.get_flowfront_bool_dryspot,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator_function=lambda summary_writer:
            BinaryClassificationEvaluator(summary_writer=summary_writer,
                                          save_path=self.training_save_path,
                                          skip_images=True),
            data_root=resources.test_src_dir,
        )

    def test_training(self):
        self.dt.start_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / "output.log") as f:
            content = f.read()
            epochs = re.findall("Mean Loss on Eval", content)
            self.assertTrue(len(epochs) > 0)

    def tearDown(self) -> None:
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.training_save_path)


if __name__ == "__main__":
    unittest.main()
