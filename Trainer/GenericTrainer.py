import logging
import socket
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Pipeline import torch_datagenerator as td
from Utils import logging_cfg
from Utils.eval_utils import eval_preparation
from Utils.training_utils import count_parameters, CheckpointingStrategy


class ModelTrainer:
    """Unified class for training a model.
    Args:
        model: Pytorch model to be trained.
        data_source_paths: List of file paths containing the files for
                           training.
        save_path: Path for saving outputs.
        load_datasets_path: Path containing dedidacted Datasets in a pickled
                            format.
        cache_path: Path containing cached objects.
        batch_size: Batch size for training.
        train_print_frequency: Frequency (in steps) in which infos about the 
                          current loss gets logged.
        epochs: Number of epochs for training.
        dummy_epoch: If set to True, a dummy epoch will be fetched before training.
                     This results in better shuffling for the first epoch
        num_workers: Number of processes for processing data.
        num_validation_samples: Number of samples for the validation set.
        num_test_samples: Number of samples for the test set.
        data_processing_function: function object used by the data generator
                                  for transforming paths into desired data.
        data_gather_function: function object used by the data generator for
                              gathering the paths to the single files.
        loss_criterion: Loss criterion for training.
        learning_rate: Learning rate for training.
        classification_evaluator_function: Classification Evaluator for evaluating the
                                  models performance.
    """

    def __init__(
        self,
        model_creation_function,
        data_source_paths,
        save_path,
        load_datasets_path=None,
        cache_path=None,
        batch_size=1,
        train_print_frequency=10,
        epochs=10,
        dummy_epoch=True,
        num_workers=10,
        num_validation_samples=10,
        num_test_samples=10,
        data_processing_function=None,
        data_gather_function=None,
        looping_strategy=None,
        cache_mode=td.CachingMode.Both,
        loss_criterion=None,
        optimizer_function=lambda params: torch.optim.Adam(params, lr=0.0001),
        lr_scheduler_function=None,
        optimizer_path=None,
        classification_evaluator_function=None,
        checkpointing_strategy=CheckpointingStrategy.Best
    ):
        initial_timestamp = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_path = save_path / initial_timestamp
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.cache_path = cache_path
        self.train_print_frequency = train_print_frequency
        self.data_source_paths = data_source_paths
        self.batch_size = batch_size
        self.load_datasets_path = load_datasets_path
        self.epochs = epochs
        self.dummy_epoch = dummy_epoch
        self.num_workers = num_workers
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.data_processing_function = data_processing_function
        self.data_gather_function = data_gather_function
        self.looping_strategy = looping_strategy
        self.cache_mode = cache_mode
        self.data_generator = None
        self.test_data_generator = None
        self.model = None
        self.model_creation_function = model_creation_function
        self.model_name = "Model"
        self.logger = logging.getLogger(__name__)
        self.best_loss = np.finfo(float).max

        self.optimizer_function = optimizer_function
        self.lr_scheduler_function = lr_scheduler_function
        self.lr_scheduler = None
        self.optimizer_path = optimizer_path
        self.optimizer = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_criterion = loss_criterion
        self.checkpointing = checkpointing_strategy
        self.classification_evaluator_function = classification_evaluator_function
        self.classification_evaluator = None
        self.writer = None

    def __create_datagenerator(self):
        try:
            generator = td.LoopingDataGenerator(
                self.data_source_paths,
                self.data_gather_function,
                self.data_processing_function,
                batch_size=self.batch_size,
                num_validation_samples=self.num_validation_samples,
                num_test_samples=self.num_test_samples,
                split_load_path=self.load_datasets_path,
                split_save_path=self.save_path,
                num_workers=self.num_workers,
                cache_path=self.cache_path,
                cache_mode=self.cache_mode,
                looping_strategy=self.looping_strategy
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()
        return generator

    def __print_info(self):
        self.logger.info("###########################################")
        self.logger.info(">>> Model Trainer INFO <<<")
        self.logger.info(f"Loss criterion: {self.loss_criterion}")
        self.logger.info(f"Optimizer: {self.optimizer}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Parameter count: {count_parameters(self.model)}")
        self.logger.info("###########################################")
        self.writer.add_text("Info/Model", f"{self.model_name}")
        self.writer.add_text("Info/LossCriterion", f"{self.loss_criterion}")
        self.writer.add_text("Info/BatchSize", f"{self.batch_size}")

    def __create_model_and_optimizer(self):
        logger = logging.getLogger(__name__)
        logger.info("Generating Model")
        if self.model is None:
            self.model = self.model_creation_function()
            self.model_name = self.model._get_name()

        if "swt-dgx" in socket.gethostname():
            logger.info("Invoking data parallel model.")
            self.model = nn.DataParallel(self.model).to("cuda:0")
        else:
            self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.optimizer_path is None:
            self.optimizer = self.optimizer_function(self.model.parameters())
            if self.lr_scheduler_function is not None:
                self.lr_scheduler = self.lr_scheduler_function(self.optimizer)
        else:
            self.logger.info(f'Loading optimizer state from {self.optimizer_path}')
            self.optimizer = self.optimizer_function(self.model.parameters())
            checkpoint = torch.load(self.optimizer_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def start_training(self,):
        """ Sets up training and logging and starts train loop
        """
        # self.save_path.mkdir(parents=True, exist_ok=True)
        logging_cfg.apply_logging_config(self.save_path)
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.classification_evaluator = self.classification_evaluator_function(sw=self.writer)

        logger = logging.getLogger(__name__)
        logger.info(f"Generating Generator")

        self.data_generator = self.__create_datagenerator()

        logger.info("Saving code and generating SLURM script for later evaluation")
        eval_preparation(self.save_path)

        self.__create_model_and_optimizer()

        self.__print_info()

        if self.dummy_epoch:
            logger.info("Prefetching a dummy epoch to get proper shuffling on the first training epoch")
            start_time = time.time()
            for i, _ in enumerate(self.data_generator):
                ctime = time.time()
                if ctime - start_time > 60:
                    start_time = ctime
                    logger.info(f"Fetched {i} batches.")
            logger.info(f"Total number of samples: {len(self.data_generator)}")

        logger.info("Running eval before training to see, if any training happens")
        validation_loss = self.__eval(self.data_generator.get_validation_samples(), 0, 0)
        self.writer.add_scalar("Validation/Loss", validation_loss, 0)
        logger.info("The Training Will Start Shortly")
        self.__train_loop()

        logging.shutdown()

    def __train_loop(self):
        start_time = time.time()
        eval_step = 1
        step_count = 0
        for epoch in range(self.epochs):
            i = 0
            self.logger.info(f"Starting epoch {epoch}")
            epoch_start = time.time()
            for inputs, label, aux in self.data_generator:
                inputs = inputs.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.loss_criterion(outputs, label)
                self.writer.add_scalar("Training/Loss", loss.item(), step_count)
                loss.backward()
                self.optimizer.step()
                if i % self.train_print_frequency == 0 and i != 0:
                    time_delta = time.time() - start_time

                    progress = i / (len(self.data_generator) / self.batch_size)
                    eta = (len(self.data_generator) / self.batch_size - i) * ((time.time() - epoch_start) / i)

                    hours = f"{eta//3600}h " if eta // 3600 > 0 else ""
                    self.logger.info(
                        f"Loss: {loss.item():12.4f} || Duration of step {step_count:6}: {time_delta:10.2f} s; "
                        f"{progress*100:.2f}% of epoch done; ETA {hours}{(eta%3600)//60:.0f}min {eta%60:.0f}s"
                    )
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    start_time = time.time()

                i += 1
                step_count += 1

            validation_loss = self.__eval(self.data_generator.get_validation_samples(), eval_step, step_count)
            self.writer.add_scalar("Validation/Loss", validation_loss, step_count)
            eval_step += 1

    def __eval(self, data_set, eval_step=0, step_count=0, test_mode=False):
        """Evaluators must have a commit, print and reset function. commit
            updates the evaluator with the current step,
            print can show all relevant stats and reset resets the internal
            structure if needed."
        """

        with torch.no_grad():
            self.model.eval()
            loss = 0
            count = 0
            for i, (data, label, aux) in enumerate(
                self.__batched(data_set, self.batch_size)
            ):
                auxs = list(td.split_aux_dicts(aux))
                data = data.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                # data = torch.unsqueeze(data, 0)
                # label = torch.unsqueeze(label, 0)
                output = self.model(data)
                current_loss = self.loss_criterion(output, label).item()
                loss = loss + current_loss
                count += 1
                output = output.cpu()
                label = label.cpu()
                data = data.cpu()
                if self.classification_evaluator is not None:
                    for c in range(output.size()[0]):
                        self.classification_evaluator.commit(
                            output[c], label[c], data[c], auxs[c]
                        )

            loss = loss / count
            self.logger.info(f"{eval_step} Mean Loss on Eval: {loss:8.8f}")

            if self.classification_evaluator is not None:
                self.classification_evaluator.print_metrics(step_count)
                self.classification_evaluator.reset()

            self.model.train()
            if not test_mode:
                if self.checkpointing == CheckpointingStrategy.Best and loss < self.best_loss:
                    self.__save_checkpoint(eval_step, loss)
                    self.best_loss = loss
                elif self.checkpointing == CheckpointingStrategy.All:
                    self.__save_checkpoint(eval_step, loss, fn=f"checkpoint_{eval_step}.pth")

            return loss

    def __save_checkpoint(self, eval_step, loss, fn="checkpoint.pth"):
        torch.save(
            {
                "epoch": eval_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.save_path / Path(fn),
        )

    def __load_checkpoint(self, path):
        """Loads the parameters of a previously saved model and optimizer,
        loss and epoch.
        See the official PyTorch docs for more details:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html

        Args:
            path (string): Path to the stored checkpoint.
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location="cpu")

        new_model_state_dict = OrderedDict()
        model_state_dict = checkpoint["model_state_dict"]
        if socket.gethostname() != "swt-dgx1":
            for k, v in model_state_dict.items():
                name = k[7:]  # remove `module.`
                new_model_state_dict[name] = v
            self.model.load_state_dict(new_model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return epoch, loss

    def __batched(self, data_l: list, batch_size: int):
        return DataLoader(data_l, batch_size=batch_size, shuffle=False)

    def inference_on_test_set(self, output_path: Path, checkpoint_path: Path, classification_evaluator):
        """Start evaluation on a dedicated test set. 
        Args:
            output_path:
            classificaton: Evaluator object that should be used for the test run.
        """
        save_path = output_path / "eval_on_test_set"
        save_path.mkdir(parents=True, exist_ok=True)
        self.classification_evaluator = self.classification_evaluator_function(sw=None)

        logging_cfg.apply_logging_config(save_path, eval=True)

        logger = logging.getLogger(__name__)

        self.__create_model_and_optimizer()

        logger.info("Generating Test Generator")
        data_generator = self.__create_datagenerator()
        logger.info("Loading Checkpoint")
        self.__load_checkpoint(checkpoint_path)

        data_list = data_generator.get_test_samples()
        tmp_evaluator = self.classification_evaluator
        self.classification_evaluator = classification_evaluator
        self.__eval(data_list, test_mode=True)
        self.classification_evaluator = tmp_evaluator
        logging.shutdown()
