import logging
import re
import shutil
import unittest

import Resources.testing as resources
from Trainer.GenericTrainer import ModelTrainer
import torch
from Models.erfh5_pressuresequence_CRNN import ERFH5_PressureSequence_Model
import Pipeline.data_gather as dg
import Pipeline.data_loader_sensor as dls
from Trainer.evaluation import BinaryClassificationEvaluator


class TestOkNotOkTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [
            resources.test_training_src_dir / "2019-07-11_15-14-48_100p"
        ]
        self.expected_num_epochs_during_training = 1

    def test_training_ok_notok(self):
        model_trainer = ModelTrainer(
            lambda: ERFH5_PressureSequence_Model(),
            self.training_data_paths,
            self.training_save_path,
            epochs=self.expected_num_epochs_during_training,
            data_gather_function=dg.get_filelist_within_folder,
            data_processing_function=dls.sensorgrid_simulationsuccess,
            num_validation_samples=1,
            num_test_samples=1,
            loss_criterion=torch.nn.BCELoss(),
            classification_evaluator=BinaryClassificationEvaluator()
        )

        model_trainer.start_training()

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
