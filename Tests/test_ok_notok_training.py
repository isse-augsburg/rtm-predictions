import logging
import re
import shutil
import unittest

import Tests.resources_for_testing as resources
import model_trainer_ok_notok as mt


class TestOkNotOkTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [
            resources.test_training_src_dir / "2019-07-11_15-14-48_100p"
        ]
        self.expected_num_epochs_during_training = 1

    def test_training_ok_notok(self):
        self.model_trainer = mt.create_model_trainer(
            data_source_paths=self.training_data_paths,
            save_path=self.training_save_path,
            epochs=self.expected_num_epochs_during_training,
            num_validation_samples=5,
            num_test_samples=0,
        )

        mt.run_training(self.model_trainer)

        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / "output.log") as f:
            content = f.read()
        epochs = re.findall("Mean Loss on Eval", content)
        self.assertTrue(len(epochs) > 0)

    def tearDown(self) -> None:
        self.model_trainer.data_generator.end_threads()
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.training_save_path)


if __name__ == "__main__":
    unittest.main()
