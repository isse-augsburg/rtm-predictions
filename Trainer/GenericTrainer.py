import logging
import socket
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Utils.training_utils import count_parameters


class MasterTrainer:
    """Class that runs train and evaluation loops of PyTorch models
    automatically.

    Args: 
        model: PyTorch model that should be trained.
        generator: ERFH5_DataGenerator that provides the data.
        loss_criterion: Loss criterion for training.
        train_print_frequency: Frequency of printing the current loss, in
        iterations.
        eval_frequency: Frequency of running a evaluation frequency on held out
        validation set, in iterations.
        learning_rate: Optimizer's learning rate
        classification_evaluator: Optional object for evaluating
        classification, see evaluation.py for more details
    """

    def __init__(
            self,
            model,
            generator,
            loss_criterion=torch.nn.MSELoss(),
            train_print_frequency=10,
            eval_frequency=100,
            save_path=None,
            eval_func=None,
            learning_rate=0.00001,
            calc_metrics=False,
            classification_evaluator=None,
    ):
        self.generator = generator
        self.epochs = self.generator.epochs
        self.validation_list = self.generator.get_validation_samples()
        self.model = model
        self.train_print_frequency = train_print_frequency
        self.eval_frequency = eval_frequency
        self.save_path = save_path
        self.loss_criterion = loss_criterion
        self.learning_rate = learning_rate
        self.loss_criterion = loss_criterion.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.eval_func = eval_func
        self.calc_metrics = calc_metrics
        self.classification_evaluator = classification_evaluator
        self.best_loss = np.finfo(float).max
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(self.save_path)

    def start_training(self):
        """ Prints information about the used train config and starts the
        training of the trainer's model
        """
        self.__print_info()
        self.__train()
        self.logger.info("Test set missing. So no testing.")
        # self.__eval()
        self.logger.info("TRAINING COMPLETE.")
        logging.shutdown()
        self.generator.end_threads()

    def __print_info(self):
        self.logger.info("###########################################")
        self.logger.info(">>> Model Trainer INFO <<<")
        self.logger.info(f"Loss criterion: {self.loss_criterion}")
        self.logger.info(f"Optimizer: {self.optimizer}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Evaluation frequency: {self.eval_frequency}")
        self.logger.info(f"Model:\n{self.model}")
        self.logger.info(f"Param count: {count_parameters(self.model)}")
        self.logger.info("###########################################")

    def __train(self):
        start_time = time.time()
        eval_step = 0
        time_sum = 0
        i_of_epoch = 0
        for i, (inputs, label) in enumerate(self.generator):
            i_of_epoch += 1
            inputs = inputs.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.loss_criterion(outputs, label)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("Training", loss.item(), i)
            if i % self.train_print_frequency == 0 and i != 0:
                time_delta = time.time() - start_time
                time_sum += time_delta
                self.logger.info(
                    f"Loss: {loss.item():12.4f} || Duration of step {i:6}: {time_delta:10.2f} s; "
                    f"|| Q: {self.generator.get_current_queue_length()}, "
                    f"|| {((i % self.eval_frequency) / self.eval_frequency)*100:.2f} % of Epoch"
                )
                start_time = time.time()

            if i % self.eval_frequency == 0 and i != 0:
                validation_loss = self.eval(self.validation_list, eval_step)
                self.writer.add_scalar("Validation", validation_loss, i)
                time_sum = 0
                eval_step += 1
                i_of_epoch = 0

            if eval_step >= self.epochs:
                break

    def eval(self, data_set, eval_step=0, test_mode=False):
        """Evaluators must have a commit, print and reset function. commit
            updates the evaluator with the current step,
            print can show all relevant stats and reset resets the internal
            structure if needed."
        """

        with torch.no_grad():
            self.model.eval()
            loss = 0
            count = 0
            for i, (data, label) in enumerate(self.__batched(data_set, self.generator.batch_size)):
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
                            output[c], label[c], data[c])

            loss = loss / count
            self.logger.info(f"{eval_step} Mean Loss on Eval: {loss:8.8f}")

            if self.classification_evaluator is not None:
                self.classification_evaluator.print_metrics()
                self.classification_evaluator.reset()

            self.model.train()
            if not test_mode:
                if loss < self.best_loss:
                    self.save_checkpoint(eval_step, loss)
                    self.best_loss = loss
        return current_loss

    def save_checkpoint(self, eval_step, loss):
        torch.save(
            {
                "epoch": eval_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            self.save_path / Path("checkpoint.pth"),
        )

    def load_checkpoint(self, path):
        """Loads the parameters of a previously saved model and optimizer,
        loss and epoch.
        See the official PyTorch docs for more details:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html

        ARgs:
            path (string): Path to the stored checkpoint.
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

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
        dats = []
        labs = []
        for data, label in data_l:
            dats.append(data)
            labs.append(label)
            if len(dats) == batch_size:
                yield torch.stack(dats), torch.stack(labs)
                dats = []
                labs = []
        if len(dats) > 1:
            yield torch.stack(dats), torch.stack(labs)
