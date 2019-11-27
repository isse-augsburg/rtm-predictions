import logging
import os
import shutil
import unittest
from pathlib import Path

import Tests.resources_for_testing as resources
from model_trainer_sensor_to_flow import SensorTrainer


class TestEvalPreparation(unittest.TestCase):
    def setUp(self):
        self.training_save_path = resources.test_training_out_dir
        self.training_data_paths = [
            resources.test_training_src_dir / '2019-07-11_15-14-48_100p']
        self.load_datasets_path = resources.test_training_src_dir / \
            '2019-09-06_15-44-58_63_sensors'
        self.expected_num_epochs_during_training = 0
        self.st = SensorTrainer(data_source_paths=self.training_data_paths,
                                save_datasets_path=self.training_save_path,
                                epochs=self.expected_num_epochs_during_training,
                                batch_size=1)

    def test_eval_preparation(self):
        self.st.run_training()
        dirs = [e for e in self.training_save_path.iterdir() if e.is_dir()]
        code_dir = dirs[0] / 'code'
        slurm_script = dirs[0] / 'run_model_eval.sh'
        self.assertTrue(os.path.isdir(code_dir))
        self.assertTrue(os.path.isfile(slurm_script))
        with open(slurm_script) as f:
            lines = f.read().splitlines()
            tokens = lines[-1].split()
            self.assertEqual(dirs[0], Path(tokens[-1]))
            self.assertTrue(os.path.isfile(Path(tokens[-4])))

    def tearDown(self) -> None:
        if self.st.training_data_generator is not None:
            self.st.training_data_generator.end_threads()
        logging.shutdown()
        r = logging.getLogger("")
        [r.removeHandler(x) for x in r.handlers]
        if self.training_save_path.exists():
            shutil.rmtree(self.training_save_path)
        logging.shutdown()


if __name__ == '__main__':
    unittest.main()
