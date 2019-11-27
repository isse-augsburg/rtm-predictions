import logging
import socket
from datetime import datetime
import torch
from torch import nn
from Pipeline import erfh5_pipeline as pipeline
from Trainer.GenericTrainer import MasterTrainer
from Utils import logging_cfg


class ModelTrainer:
    def __init__(self,
                 data_source_paths,
                 save_datasets_path,
                 load_datasets_path=None,
                 cache_path=None,
                 batch_size=1,
                 max_queue_length=4,
                 eval_freq=2,
                 train_print_freq=2,
                 epochs=10,
                 num_workers=10,
                 num_validation_samples=10,
                 num_test_samples=10,
                 data_processing_function=None,
                 data_gather_function=None,
                 model=None):
        self.train_print_frequency = train_print_freq
        self.initial_timestamp = str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.cache_path = cache_path
        self.data_source_paths = data_source_paths
        self.batch_size = batch_size
        self.max_queue_length = max_queue_length
        self.eval_freq = eval_freq
        self.save_datasets_path = save_datasets_path
        self.load_datasets_path = load_datasets_path
        self.epochs = epochs
        self.num_workers = num_workers
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.data_processing_function = data_processing_function
        self.data_gather_function = data_gather_function
        self.training_data_generator = None
        self.test_data_generator = None
        self.model = model

    def create_datagenerator(self, save_path, test_mode=False):
        try:
            generator = pipeline.ERFH5DataGenerator(
                data_paths=self.data_source_paths,
                num_validation_samples=self.num_validation_samples,
                num_test_samples=self.num_test_samples,
                batch_size=self.batch_size,
                epochs=self.epochs,
                max_queue_length=self.max_queue_length,
                data_processing_function=self.data_processing_function,
                data_gather_function=self.data_gather_function,
                num_workers=self.num_workers,
                cache_path=self.cache_path,
                save_path=save_path,
                load_datasets_path=self.load_datasets_path,
                test_mode=test_mode,
            )
            return generator
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()

    def run_training(self, comment, loss_criterion, learning_rate, calc_metrics, classification_evaluator):
        save_path = self.save_datasets_path / self.initial_timestamp
        save_path.mkdir(parents=True, exist_ok=True)
        logging_cfg.apply_logging_config(save_path)

        logger = logging.getLogger(__name__)

        logger.info("Generating Generator")
        self.training_data_generator = self.create_datagenerator(save_path, test_mode=False)

        logger.info("Generating Model")

        logger.info("Model to GPU")
        if socket.gethostname() == "swt-dgx1":
            self.model = nn.DataParallel(self.model).to("cuda:0")
        else:
            self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        train_wrapper = MasterTrainer(
            self.model,
            self.training_data_generator,
            comment=comment,
            loss_criterion=loss_criterion,
            savepath=save_path,
            learning_rate=learning_rate,
            calc_metrics=calc_metrics,
            train_print_frequency=self.train_print_frequency,
            eval_frequency=self.eval_freq,
            classification_evaluator=classification_evaluator
        )
        logger.info("The Training Will Start Shortly")

        train_wrapper.start_training()
        logging.shutdown()
