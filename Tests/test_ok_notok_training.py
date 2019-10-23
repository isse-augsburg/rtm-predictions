import logging
import re
import shutil
import unittest

import Tests.resources_for_testing as resources
from model_trainer_ok_notok import SuccessTrainer


class TestOkNotOkTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [resources.test_training_src_dir / '2019-07-11_15-14-48_100p']
        self.expected_num_epochs_during_training = 1

    # @unittest.skip('Needs to be fixed: Lukas.')
    def test_training_ok_notok(self):
        self.st = SuccessTrainer(
            data_source_paths=self.training_data_paths,
            save_path=self.training_save_path,
            epochs=self.expected_num_epochs_during_training,
            num_validation_samples=5,
            num_test_samples=0,
        )

        self.st.run_training()

        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / "output.log") as f:
            content = f.read()
        epochs = re.findall("Mean Loss on Eval", content)
        self.assertTrue(len(epochs) > 0)

    def tearDown(self) -> None:
        self.st.training_data_generator.end_threads()
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        shutil.rmtree(self.training_save_path)


if __name__ == "__main__":
    unittest.main()
