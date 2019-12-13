import logging
import socket
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from Pipeline import (
    torch_datagenerator as td,
)
from Trainer.GenericTrainer import MasterTrainer
from Utils import logging_cfg
from Utils.eval_utils import eval_preparation


class ModelTrainer:
    def __init__(
        self,
        model,
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
        data_processing_function=None,
        data_gather_function=None,
        looping_strategy=None,
    ):
        self.train_print_frequency = train_print_freq
        self.initial_timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_path = cache_path
        self.data_source_paths = data_source_paths
        self.batch_size = batch_size
        self.eval_freq = eval_freq
        self.save_datasets_path = save_datasets_path
        self.load_datasets_path = load_datasets_path
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.data_processing_function = data_processing_function
        self.data_gather_function = data_gather_function
        self.data_generator = None
        self.test_data_generator = None
        self.model = model
        self.looping_strategy = looping_strategy

    def create_datagenerator(self, save_path):
        try:
            generator = td.LoopingDataGenerator(
                self.data_source_paths,
                self.data_gather_function,
                self.data_processing_function,
                batch_size=self.batch_size,
                epochs=self.epochs,
                num_validation_samples=self.num_validation_samples,
                num_test_samples=self.num_test_samples,
                split_save_path=self.load_datasets_path or save_path,
                num_workers=self.num_workers,
                cache_path=self.cache_path,
                looping_strategy=self.looping_strategy
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()
        return generator

    def run_training(
        self,
        loss_criterion,
        learning_rate,
        calc_metrics,
        classification_evaluator,
    ):
        save_path = self.save_datasets_path / self.initial_timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        logging_cfg.apply_logging_config(save_path)

        logger = logging.getLogger(__name__)

        logger.info(f"Generating Generator || Batch size: {self.batch_size}")
        data_generator = self.create_datagenerator(save_path)

        logger.info("Saving code and generating SLURM script for later evaluation")
        eval_preparation(save_path)

        logger.info("Generating Model")
        if torch.cuda.is_available():
            logger.info("Model to GPU")
        if socket.gethostname() == "swt-dgx1":
            self.model = nn.DataParallel(self.model).to("cuda:0")
        else:
            self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        train_wrapper = MasterTrainer(
            self.model,
            data_generator,
            loss_criterion=loss_criterion,
            save_path=save_path,
            learning_rate=learning_rate,
            calc_metrics=calc_metrics,
            train_print_frequency=self.train_print_frequency,
            eval_frequency=self.eval_freq,
            classification_evaluator=classification_evaluator,
        )
        logger.info("The Training Will Start Shortly")

        train_wrapper.start_training()
        logging.shutdown()

    def inference_on_test_set(
            self,
            output_path: Path,
            classification_evaluator):
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
        data_generator = self.create_datagenerator(save_path)
        eval_wrapper = MasterTrainer(
            self.model,
            data_generator,
            classification_evaluator=classification_evaluator,
        )
        eval_wrapper.load_checkpoint(output_path / "checkpoint.pth")

        data_list = data_generator.get_test_samples()

        eval_wrapper.eval(data_list, test_mode=True)
        logging.shutdown()
