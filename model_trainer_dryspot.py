import argparse
import getpass
import logging
import math
import os
import pickle
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from Models.erfh5_ConvModel import DrySpotModel
from Pipeline import (
    torch_datagenerator as td,
    data_gather as dg,
    data_loader_dryspot
)
from Pipeline.erfh5_pipeline import transform_list_of_linux_paths_to_windows
from Trainer.GenericTrainer import MasterTrainer
from Trainer.evaluation import BinaryClassificationEvaluator
from Utils import logging_cfg
from Utils.eval_utils import eval_preparation
from Utils.training_utils import transform_to_tensor_and_cache, apply_blacklists


class DrySpotTrainer:
    def __init__(self,
                 data_source_paths,
                 save_datasets_path,
                 load_datasets_path=None,
                 cache_path=None,
                 batch_size=1,
                 eval_freq=2,
                 train_print_freq=2,
                 epochs=10,
                 num_workers=10,
                 num_validation_samples=10,
                 num_test_samples=10,
                 model=DrySpotModel()):
        self.train_print_frequency = train_print_freq
        self.initial_timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_path = cache_path
        self.data_source_paths = data_source_paths
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.save_datasets_path = save_datasets_path
        self.load_datasets_path = save_datasets_path
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.training_data_generator = None
        self.test_data_generator = None
        self.model = model

    def create_datagenerator(self, save_path, data_processing_function):
        try:
            generator = td.LoopingDataGenerator(
                self.data_source_paths,
                dg.get_filelist_within_folder,
                data_processing_function,
                batch_size=self.batch_size,
                epochs=self.epochs,
                num_validation_samples=self.num_validation_samples,
                num_test_samples=self.num_test_samples,
                split_save_path=self.load_datasets_path,
                num_workers=self.num_workers,
                cache_path=self.cache_path,
                looping_strategy=td.SimpleListLoopingStrategy()
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()
        return generator

    def inference_on_test_set(self, output_path, source_path):
        save_path = output_path / "eval_on_test_set"
        save_path.mkdir(parents=True, exist_ok=True)

        logging_cfg.apply_logging_config(save_path, eval=True)

        logger = logging.getLogger(__name__)

        if socket.gethostname() == "swt-dgx1":
            logger.info('Invoking data parallel model.')
            self.model = nn.DataParallel(self.model).to("cuda:0")
        else:
            self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info("Generating Test Generator")
        self.test_data_generator = self.create_datagenerator(save_path,
                                                             data_loader_dryspot.get_flowfront_bool_dryspot_143x111,
                                                             )
        evaluator = BinaryClassificationEvaluator(save_path=save_path, skip_images=False)
        eval_wrapper = MasterTrainer(
            self.model,
            self.test_data_generator,
            classification_evaluator=evaluator,
        )
        eval_wrapper.load_checkpoint(source_path / "checkpoint.pth")

        with open(source_path / "test_set.p", "rb") as f:
            test_set = pickle.load(f)
        data_list = self.create_data_set_from_paths(test_set)

        eval_wrapper.eval(data_list, test_mode=True)
        logging.shutdown()

    def create_data_set_from_paths(self, test_set):
        test_set = transform_list_of_linux_paths_to_windows(test_set)
        data_list = []
        full = False
        for p in test_set:
            instance = self.test_data_generator.data_function(p)
            if instance is None:
                continue
            for num, i in enumerate(instance):
                transform_to_tensor_and_cache(i, data_list)
                if len(data_list) >= self.num_test_samples:
                    full = True
            if full:
                data_list = data_list[:self.num_test_samples]
                break
        return data_list

    def run_training(self):
        save_path = self.save_datasets_path / self.initial_timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        logging_cfg.apply_logging_config(save_path)
        logger = logging.getLogger(__name__)

        logger.info("Saving code and generating SLURM script for later evaluation")
        eval_preparation(save_path, os.path.abspath(__file__))

        evaluator = BinaryClassificationEvaluator(save_path=save_path, skip_images=True)

        logger.info("Generating Generator")
        self.training_data_generator = self.create_datagenerator(save_path,
                                                                 data_loader_dryspot.get_flowfront_bool_dryspot_143x111,
                                                                 )

        logger.info("Generating Model")
        if torch.cuda.is_available():
            logger.info("Model to GPU")
        if socket.gethostname() == "swt-dgx1":
            self.model = nn.DataParallel(self.model).to("cuda:0")
        else:
            self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        train_wrapper = MasterTrainer(
            self.model,
            self.training_data_generator,
            comment=get_comment(),
            loss_criterion=torch.nn.MSELoss(),
            savepath=save_path,
            learning_rate=0.0001,
            calc_metrics=False,
            train_print_frequency=self.train_print_frequency,
            eval_frequency=self.eval_freq,
            classification_evaluator=evaluator
        )
        train_wrapper.start_training()
        logging.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training or test.')
    parser.add_argument('--eval', action='store_true', help='Run a test.')
    parser.add_argument('--eval_path', type=str, default=None, help='Full directory path of trained model (to test).')
    args = parser.parse_args()
    run_eval = args.eval
    eval_path = args.eval_path

    if run_eval and eval_path is None:
        logger = logging.getLogger(__name__)
        logger.error("No eval_path given. You should specify the --eval_path argument if you would like to run a test.")
        logger.error(parser.format_help())
        logging.shutdown()
        sys.exit()

    num_samples_runs = 2000000  # 2.1 M - blacklist, still a guestimate # 9080 * 188  # guestimate ~ 188 p. Sim.
    _train_print_freq = 10
    if socket.gethostname() == "swt-dgx1":
        _home = Path('/cfs/home')
        _cache_path = None
        _batch_size = 1024
        _eval_freq = int(num_samples_runs / _batch_size)
        # _eval_freq = 70
        if getpass.getuser() == "stiebesi":
            _save_path = Path("/cfs/share/cache/output_simon")
        elif getpass.getuser() == "schroeni":
            _save_path = Path("/cfs/share/cache/output_niklas")
        else:
            _save_path = Path("/cfs/share/cache/output")
        _epochs = 1000
        _num_workers = 18
        _num_validation_samples_frames = _batch_size * 8
        _num_test_samples_frames = _batch_size * 8

    elif socket.gethostname() == "swtse130":
        _home = Path('X:/')
        _cache_path = None
        _batch_size = 128
        _eval_freq = 30
        _save_path = Path(r"C:\Users\stiebesi\CACHE\train_out")
        _epochs = 2
        _num_workers = 10
        _num_validation_samples_frames = 100
        _num_test_samples_frames = 100

    else:
        _home = Path("/cfs/home")
        _cache_path = None
        _batch_size = 128
        _eval_freq = 30
        _save_path = Path(r"/cfs/share/cache/output_johannes")
        _epochs = 2
        _num_workers = 10
        _num_validation_samples_frames = 100
        _num_test_samples_frames = 100

    _data_root = _home / "s/t/stiebesi/data/RTM/Leoben/output/with_shapes"

    if not run_eval:
        _data_source_paths = [
            # _data_root / "2019-07-23_15-38-08_5000p",
            _data_root / "2019-07-24_16-32-40_5000p",
            # _data_root / "2019-07-29_10-45-18_5000p",
            # _data_root / "2019-08-23_15-10-02_5000p",
            # _data_root / "2019-08-24_11-51-48_5000p",
            # _data_root / "2019-08-25_09-16-40_5000p",
            # _data_root / "2019-08-26_16-59-08_6000p",
            # _data_root / '2019-09-06_17-03-51_10000p',
            _data_root / '2019-11-08_15-40-44_5000p'
        ]
    else:
        _data_source_paths = []

    # TODO Move to a data_gather func
    _data_source_paths = apply_blacklists(_data_source_paths)

    # Running with the same data sets
    _load_datasets_path = _home / 's/t/stiebesi/data/RTM/Leoben/reference_datasets/dryspot_detection'
    # _load_datasets_path = None

    st = DrySpotTrainer(cache_path=_cache_path,
                        data_source_paths=_data_source_paths,
                        batch_size=_batch_size,
                        eval_freq=_eval_freq,
                        train_print_freq=math.ceil(_eval_freq / 50),
                        save_datasets_path=_save_path,
                        load_datasets_path=_load_datasets_path,
                        epochs=_epochs,
                        num_workers=_num_workers,
                        num_validation_samples=_num_validation_samples_frames,
                        num_test_samples=_num_test_samples_frames,
                        model=DrySpotModel(),
                        )

    if not run_eval:
        st.run_training()
    else:
        if socket.gethostname() != "swtse130":
            st.inference_on_test_set(source_path=eval_path,
                                     output_path=eval_path)
    logging.shutdown()
