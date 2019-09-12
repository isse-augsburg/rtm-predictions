import logging
import re
import shutil
import unittest
from pathlib import Path

from model_trainer_ok_notok import SuccessTrainer


class TestOkNotOkTraining(unittest.TestCase):
    def setUp(self):
        self.training_save_path = Path(
            "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/code/tests/training"
        )
        self.training_data_paths = [
            Path(
                "/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/s/t/stiebesi/code/tests/training_data_source/2019-07-11_15-14-48_100p"
            )
        ]
        self.expected_num_epochs_during_training = 1

    def test_training_ok_notok(self):
        st = SuccessTrainer(
            data_source_paths=self.training_data_paths,
            save_path=self.training_save_path,
            epochs=self.expected_num_epochs_during_training,
            num_validation_samples=5,
            num_test_samples=0,
        )
        st.run_training()

        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        with open(dirs[0] / "output.log") as f:
            content = f.read()
            epochs = re.findall("Mean Loss on Eval", content)
            self.assertTrue(len(epochs) > 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.training_save_path)
        logging.shutdown()


if __name__ == "__main__":
    unittest.main()
