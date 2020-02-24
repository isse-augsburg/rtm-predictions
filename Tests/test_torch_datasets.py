import logging
import tempfile
import unittest
from pathlib import Path

import torch

import Pipeline.data_gather as dg
import Resources.testing as test_resources
import Resources.training as tr_resources
from Models.erfh5_fullyConnected import S20DryspotModelFCWide
from Pipeline.data_loader_flowfront_sensor import DataloaderFlowfrontSensor
from Trainer.ModelTrainer import ModelTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils.data_utils import change_win_to_unix_path_if_needed


class TestSaveDatasetsTorch(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_dataset_resources = test_resources.test_src_dir / self.__class__.__name__

    def create_trainer_and_start(self, out_path, load_torch_dataset_path, epochs=1):
        dlds = DataloaderFlowfrontSensor(sensor_indizes=((1, 8), (1, 8)))
        m = ModelTrainer(lambda: S20DryspotModelFCWide(),
                         data_source_paths=tr_resources.get_data_paths_debug(),
                         save_path=out_path,
                         load_datasets_path=None,
                         # load_datasets_path=self.torch_dataset_resources / "reference_datasets",
                         cache_path=None,
                         num_validation_samples=8,
                         num_test_samples=8,
                         num_workers=0,
                         epochs=epochs,
                         data_processing_function=dlds.get_flowfront_sensor_bool_dryspot,
                         data_gather_function=dg.get_filelist_within_folder_blacklisted,
                         loss_criterion=torch.nn.BCELoss(),
                         optimizer_function=lambda params: torch.optim.AdamW(params, lr=1e-4),
                         classification_evaluator_function=lambda summary_writer:
                         BinaryClassificationEvaluator(summary_writer=summary_writer),
                         save_torch_dataset_path=out_path / Path(__file__).stem,
                         load_torch_dataset_path=load_torch_dataset_path,
                         )
        m.start_training()
        return m

    def compare_old_new_dataset(self, old_data: list, new_data: list):
        for i in range(len(old_data)):
            odata, olabel, osourcepath = old_data[i]
            ndata, nlabel, nsourcepath = new_data[i]
            self.assertTrue(torch.equal(odata, ndata))
            self.assertTrue(torch.equal(olabel, nlabel))
            osourcepath = change_win_to_unix_path_if_needed(osourcepath["sourcefile"])
            self.assertEqual(osourcepath, nsourcepath["sourcefile"])

    def test_save_train_val_test_sets(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsSaving") as tempdir:
            out_path = Path(tempdir)
            m = self.create_trainer_and_start(out_path, out_path / Path(__file__).stem, epochs=2)
            self.assertTrue((out_path / Path(__file__).stem / "train_set_torch.p").is_file())
            self.assertTrue((out_path / Path(__file__).stem / "val_set_torch.p").is_file())
            m.inference_on_test_set()
            self.assertTrue((out_path / Path(__file__).stem / "test_set_torch.p").is_file())
            logging.shutdown()

    def test_load_all_data_sets(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir), self.torch_dataset_resources / "all")
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_train_set)
            logging.shutdown()

    def test_load_train_set_only(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir), self.torch_dataset_resources / "train_only")
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertFalse(m.data_generator.loaded_val_set)
            self.assertFalse(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/val_set_torch.p", "rb") as f:
                saved_val_set = torch.load(f)
            with open(self.torch_dataset_resources / "all/test_set_torch.p", "rb") as f:
                saved_test_set = torch.load(f)
            self.compare_old_new_dataset(saved_val_set, m.data_generator.get_validation_samples())
            self.compare_old_new_dataset(saved_test_set, m.data_generator.get_test_samples())
            logging.shutdown()

    def test_load_train_and_test_set_only(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir), self.torch_dataset_resources / "train_and_test_only")
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertFalse(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/val_set_torch.p", "rb") as f:
                saved_val_set = torch.load(f)
            self.compare_old_new_dataset(saved_val_set, m.data_generator.get_validation_samples())
            logging.shutdown()

    def test_load_train_and_val_set_only(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir), self.torch_dataset_resources / "train_and_val_only")
            self.assertTrue(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertFalse(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/test_set_torch.p", "rb") as f:
                saved_test_set = torch.load(f)
            self.compare_old_new_dataset(saved_test_set, m.data_generator.get_test_samples())
            logging.shutdown()

    def test_load_val_and_test_set_only(self):
        with tempfile.TemporaryDirectory(prefix="TorchDataSetsLoading") as tempdir:
            m = self.create_trainer_and_start(Path(tempdir), self.torch_dataset_resources / "val_test_only")
            self.assertFalse(m.data_generator.loaded_train_set)
            self.assertTrue(m.data_generator.loaded_val_set)
            self.assertTrue(m.data_generator.loaded_test_set)
            with open(self.torch_dataset_resources / "all/train_set_torch.p", "rb") as f:
                saved_train_set = torch.load(f)
            old_paths = set([change_win_to_unix_path_if_needed(x[2]["sourcefile"]) for x in saved_train_set])
            new_paths = set([x[2]["sourcefile"][0] for x in m.data_generator])
            self.assertEqual(old_paths, new_paths)
            logging.shutdown()