import logging
import os
import socket
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import Resources.training as r
from Pipeline import torch_datagenerator as td
from Utils import logging_cfg
from Utils.data_utils import handle_torch_caching
from Utils.eval_utils import eval_preparation
from Utils.training_utils import count_parameters, CheckpointingStrategy

try:
    from apex import amp
except ImportError:
    if os.name != "nt":
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex for mixed precision.")
    else:
        print("Currently no support for apex on windows. Continuing ...")


class ModelTrainer:
    """Unified class for training a model.

    Args:
        model_creation_function: Pytorch model to be trained. Passed as a lambda call,
                                 e.g. lambda: YourModel()
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
        produce_torch_datasets_only: Only write cache data, useful for producing and publishing certain datasets.
                                     Output directory is the caching directory, that comes out of the
                                     handle_torch_caching method. The root dir is saved in
                                     Ressources.training.datasets_dryspots_torch
        num_workers: Number of processes for processing data.
        num_validation_samples: Number of samples for the validation set.
        num_test_samples: Number of samples for the test set.
        data_root: The root directory for the training data.
        data_processing_function: function object used by the data generator
                                  for transforming paths into desired data.
        data_gather_function: function object used by the data generator for
                              gathering the paths to the single files.
        cache_mode: From enum CachingMode in Pipeline.TorchDataGeneratorUtils.torch_internal.py.
                      Specifies if and how caching is done.
        loss_criterion: Loss criterion for training.
        optimizer_function: Object of a Torch optimizer. Passed as a lambda call, e.g.
                            lambda: params: torch.optim.Adam(params, lr=0.0001)
        classification_evaluator: Classification Evaluator for evaluating the
                                  models performance.
        checkpointing_strategy: From enum CheckpointingStrategy in Pipeline.TorchDataGeneratorUtils.torch_internal.py.
                                Specifies which checkpoints are stored during training.
    """

    def __init__(
        self,
        model_creation_function,
        data_source_paths: list,
        save_path,
        load_datasets_path=None,
        cache_path=None,
        batch_size: int = 1,
        train_print_frequency: int = 10,
        epochs: int = 10,
        dummy_epoch=True,
        produce_torch_datasets_only=False,
        num_workers: int = 10,
        num_validation_samples: int = 10,
        num_test_samples: int = 10,
        data_root: Path = None,
        data_processing_function=None,
        data_gather_function=None,
        looping_strategy=None,
        cache_mode=td.CachingMode.FileList,
        loss_criterion=MSELoss(),
        optimizer_function=lambda params: torch.optim.Adam(params, lr=0.0001),
        lr_scheduler_function=None,
        optimizer_path=None,
        classification_evaluator_function=None,
        checkpointing_strategy=CheckpointingStrategy.Best,
        run_eval_step_before_training=False,
        dont_care_num_samples=False,
        use_mixed_precision=False,
        sampler=None,
        caching_torch=True,
        demo_path=None,
        resize_label_to=(0, 0),
        load_test_set_in_training_mode=False
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
        self.produce_torch_datasets_only = produce_torch_datasets_only
        if produce_torch_datasets_only and not dummy_epoch:
            raise ValueError("Can't do a cache only run without enabling dummy_epoch!")
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
        self.sampler = sampler
        self.demo_path = demo_path
        if demo_path is not None:
            caching_torch = False

        if caching_torch:
            load_and_save_path, data_loader_hash = handle_torch_caching(
                self.data_processing_function, self.data_source_paths, self.sampler, self.batch_size)
            self.data_loader_hash = data_loader_hash
            self.load_torch_dataset_path = load_and_save_path
            self.save_torch_dataset_path = load_and_save_path
        else:
            self.data_loader_hash = "NOT_CACHING"
            self.load_torch_dataset_path = None
            self.save_torch_dataset_path = None

        if self.demo_path is not None:
            self.data_loader_hash = "DEMO_MODE"
            self.load_torch_dataset_path = Path(self.demo_path)
            self.save_torch_dataset_path = Path(self.demo_path)

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
        self.run_eval_step_before_training = run_eval_step_before_training
        self.dont_care_num_samples = dont_care_num_samples

        self.use_mixed_precision = use_mixed_precision
        self.resize_label = resize_label_to
        self.load_test_set_in_training_mode = load_test_set_in_training_mode

        self.data_root = data_root

    def __create_datagenerator(self, test_mode=False):
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
                split_data_root=self.data_root,
                num_workers=self.num_workers,
                cache_path=self.cache_path,
                cache_mode=self.cache_mode,
                looping_strategy=self.looping_strategy,
                save_torch_dataset_path=self.save_torch_dataset_path,
                load_torch_dataset_path=self.load_torch_dataset_path,
                dont_care_num_samples=self.dont_care_num_samples,
                test_mode=test_mode,
                sampler=self.sampler,
                load_test_set_in_training_mode=self.load_test_set_in_training_mode,
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("Fatal Error:")
            exit()
        return generator

    def __get_model_def(self):
        model_as_str = self.model_name + ":  \n"
        if self.model.__class__.__name__ == "DataParallel":
            m = list(self.model.children())[0]
        else:
            m = self.model
        for c in m.named_children():
            if len(list(c[1].parameters())) == 0:
                model_as_str += str(c)
                model_as_str += "  \n"
                continue
            else:
                # If first parameter of layer is frozen, so is the rest of the layer -> parameters()[0]
                model_as_str += "~~  " if not list(c[1].parameters())[0].requires_grad else ""
                model_as_str += str(c)
                model_as_str += " ~~   \n" if not list(c[1].parameters())[0].requires_grad else "  \n"

        return model_as_str

    def __print_info(self):
        param_count = count_parameters(self.model)
        sched_str = self.lr_scheduler.__class__.__name__ + f"  \n{self.lr_scheduler.state_dict()}" \
            if self.lr_scheduler is not None else "None"
        self.logger.info("###########################################")
        self.logger.info(">>> Model Trainer INFO <<<")
        self.logger.info(f"Loss criterion: {self.loss_criterion}")
        self.logger.info(f"Optimizer: {self.optimizer}")
        self.logger.info(f"LR scheduler: {sched_str}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"Parameter count: {param_count}")
        self.logger.info("###########################################")
        self.writer.add_text("General/LossCriterion", f"{self.loss_criterion}")
        self.writer.add_text("General/BatchSize", f"{self.batch_size}")
        self.writer.add_text("General/MixedPrecision", f"{self.use_mixed_precision}")
        optim_str = str(self.optimizer).replace("\n", "  \n")
        self.writer.add_text("Optimizer/Optimizer", f"{optim_str}")
        self.writer.add_text("Optimizer/LRScheduler", f"{sched_str}")
        self.writer.add_text("Model/Structure", f"{self.__get_model_def()}")
        self.writer.add_text("Model/ParamCount", f"{param_count}")
        if hasattr(self.model, "round_at") and self.model.round_at is not None:
            self.writer.add_text("Model/Threshold", f"{self.model.round_at}")
        self.writer.add_text("Data/SourcePaths", f"{[str(p) for p in self.data_source_paths]}")
        self.writer.add_text("Data/CheckpointSourcePath", f"{self.load_datasets_path}")
        dl_info = self.data_processing_function.__self__.__dict__
        dl_info["data_processing_function"] = self.data_processing_function.__name__
        dl_str = '  \n'.join([f"{k}: {dl_info[k]}" for k in dl_info if dl_info[k] is not None])
        self.writer.add_text("Data/DataLoader", f"{dl_str}")

    def __create_model_and_optimizer(self):
        logger = logging.getLogger(__name__)
        logger.info("Generating Model")
        if not self.use_mixed_precision:
            if self.model is None:
                self.model = self.model_creation_function()
                self.model_name = self.model.__class__.__name__

                if "swt-dgx" in socket.gethostname():
                    logger.info("Invoking data parallel model.")
                    self.model = nn.DataParallel(self.model).to("cuda:0")
                else:
                    self.model = self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

            self.create_optimizer_and_lr_scheduler()
        else:
            if self.model is None:
                self.model = self.model_creation_function()
                self.model_name = self.model.__class__.__name__

            self.create_optimizer_and_lr_scheduler()
            self.model = self.model.cuda()
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            if "swt-dgx" in socket.gethostname():
                logger.info("Invoking data parallel model.")
                self.model = nn.DataParallel(self.model).to("cuda:0")

    def create_optimizer_and_lr_scheduler(self):
        if self.optimizer is None:
            if self.optimizer_path is None:
                self.optimizer = self.optimizer_function(self.model.parameters())
            else:
                self.logger.info(f'Loading optimizer state from {self.optimizer_path}')
                self.optimizer = self.optimizer_function(self.model.parameters())
                checkpoint = torch.load(self.optimizer_path)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler_function is not None:
            self.lr_scheduler = self.lr_scheduler_function(self.optimizer)

    def start_training(self,):
        """ Sets up training and logging and starts train loop
        """
        # self.save_path.mkdir(parents=True, exist_ok=True)
        if self.demo_path is not None:
            print(f"Running in demo mode. Please refer to {self.save_path} for logs et al.")
        logging_cfg.apply_logging_config(self.save_path)
        self.writer = SummaryWriter(log_dir=self.save_path)
        self.classification_evaluator = self.classification_evaluator_function(summary_writer=self.writer)

        logger = logging.getLogger(__name__)
        logger.info(f"Generating Generator")

        self.data_generator = self.__create_datagenerator()
        if self.data_generator.loaded_train_set:
            self.dummy_epoch = False

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

        if self.run_eval_step_before_training:
            logger.info("Running eval before training to see, if any training happens")
            validation_loss = self.__eval(self.data_generator.get_validation_samples(), 0, 0)
            self.writer.add_scalar("Validation/Loss", validation_loss, 0)

        if self.produce_torch_datasets_only:
            logger.info(f"Triggering caching, saving all datasets to {self.save_torch_dataset_path}")
            logger.info("Training dataset ...")
            iter(self.data_generator)
            logger.info("Validation dataset ...")
            _ = self.data_generator.get_validation_samples()
            logger.info("Test dataset ...")
            _ = self.data_generator.get_test_samples()
        else:
            logger.info("The Training Will Start Shortly")
            self.__train_loop()

        logging.shutdown()

    def __train_loop(self):
        start_time = time.time()
        eval_step = 0
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
                label = self.resize_label_if_necessary(label)
                loss = self.loss_criterion(outputs, label)
                self.writer.add_scalar("Training/Loss", loss.item(), step_count)
                if not self.use_mixed_precision:
                    loss.backward()
                else:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                self.optimizer.step()
                if i % self.train_print_frequency == 0 and i != 0:
                    time_delta = time.time() - start_time

                    progress = i / (len(self.data_generator) / self.batch_size)
                    eta = (len(self.data_generator) / self.batch_size - i) * ((time.time() - epoch_start) / i)

                    hours = f"{eta // 3600}h " if eta // 3600 > 0 else ""
                    self.logger.info(
                        f"Loss: {loss.item():12.4f} || Duration of step {step_count:6}: {time_delta:10.2f} s; "
                        f"{progress * 100:.2f}% of epoch done; ETA {hours}{(eta % 3600) // 60:.0f}min {eta % 60:.0f}s"
                    )
                    start_time = time.time()

                i += 1
                step_count += 1

            validation_loss = self.__eval(self.data_generator.get_validation_samples(), eval_step, step_count)
            self.writer.add_scalar("Validation/Loss", validation_loss, step_count)
            if self.lr_scheduler is not None:
                old_lr = [pg['lr'] for pg in self.optimizer.state_dict()['param_groups']]
                self.lr_scheduler.step()
                self.logger.info(f"LR scheduler step; LR: {old_lr} -> "
                                 f"{[pg['lr'] for pg in self.optimizer.state_dict()['param_groups']]}")
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
                label = self.resize_label_if_necessary(label)
                current_loss = self.loss_criterion(output, label).item()
                loss = loss + current_loss
                count += 1
                output = output.cpu()
                label = label.cpu()
                data = data.cpu()
                if self.classification_evaluator is not None:
                    self.classification_evaluator.commit(output, label, data, auxs)

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

    def resize_label_if_necessary(self, label):
        """
        Resize the label: saves online storage by making it possible to use the bigger image labels of 1140 sensors
        also for 80 and 20 sensors
        :param label:
        :return:
        """
        if self.resize_label != (0, 0):
            label = torch.nn.functional.interpolate(label.reshape(-1, 1, label.shape[1], label.shape[2]),
                                                    self.resize_label)
            label = label.squeeze()
        return label

    def __save_checkpoint(self, eval_step, loss, fn=r.chkp):
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
            path (Path): Path to the stored checkpoint.
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location="cpu")

        new_model_state_dict = OrderedDict()
        model_state_dict = checkpoint["model_state_dict"]
        if "swt-dgx" not in socket.gethostname():
            for k, v in model_state_dict.items():
                if k.startswith("module"):
                    k = k[7:]  # remove `module.`
                new_model_state_dict[k] = v
            self.model.load_state_dict(new_model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return epoch, loss

    def __batched(self, data_l: list, batch_size: int):
        return DataLoader(data_l, batch_size=batch_size, shuffle=False)

    def inference_on_test_set(self,
                              output_path: Path = None,
                              checkpoint_path: Path = None,
                              classification_evaluator_function=None):
        """Start evaluation on a dedicated test set. 
        Args:
            output_path: Directory for test outputs.
            checkpoint_path : ...
            classification_evaluator_function: lambda with Evaluator object that should be used for the test run.
        """
        if output_path is not None:
            save_path = output_path / "eval_on_test_set"
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.save_path

        if self.demo_path is not None:
            print(f"Eval - running in demo mode. Please refer to {save_path.absolute()} for log / results.")

        logging_cfg.apply_logging_config(save_path, eval=True)

        logger = logging.getLogger(__name__)

        self.__create_model_and_optimizer()

        logger.info("Generating Test Generator")
        data_generator = self.__create_datagenerator(test_mode=True)
        logger.info("Loading Checkpoint")
        if checkpoint_path is not None:
            logger.info(f"Loading Checkpoint: {checkpoint_path}")
            self.__load_checkpoint(checkpoint_path)
        else:
            logger.info(f"Loading Checkpoint: {self.save_path / r.chkp}")
            self.__load_checkpoint(self.save_path / r.chkp)

        data_list = data_generator.get_test_samples()
        if classification_evaluator_function is not None:
            self.classification_evaluator = classification_evaluator_function(summary_writer=None)
        logger.info("Starting inference")
        self.__eval(data_list, test_mode=True)
        logger.info("Inference completed.")
        logging.shutdown()
