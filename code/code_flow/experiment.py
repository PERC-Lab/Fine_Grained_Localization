"""Class to represent an experiment that is run"""

import os
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from .utils import Log, Results

# TODO
# Make logging a callback function


class BinaryclassExperiment():

    def __init__(self,
                 name,
                 model,
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 optimizer,
                 criterion,
                 epochs,
                 max_clip,
                 config,
                 dirpath,
                 description=None) -> None:

        self.name = name
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.max_clip = max_clip
        self.config = config
        self.dirpath = Path(dirpath) / Path(name)
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        self.logger = Log(name, self.dirpath)
        self.writer = SummaryWriter(Path(self.dirpath/"_tensorboard"))
        self.description = description

    def train(self, log_every):
        h = self.model.init_hidden()
        step = 0
        start_time = time.time()
        epoch_losses = []
        for *inputs, labels in self.train_dataloader:

            predictions, h = self.model(*inputs)
            step += 1
            loss = self.criterion(predictions, labels.float())
            self.optimizer.zero_grad()
            loss.backward()
            epoch_losses.append(loss.item())

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.max_clip)
            self.optimizer.step()

            # TODO Make this callback function
            if (step % log_every == 0) and (step > 0):
                elapsed_time = time.time() - start_time
                self.logger.info(f"Step: {step}\n"
                                 f"Time Elapsed: {elapsed_time:.3f}s\n")
                self.evaluate()
                self.model.train()
                start_time = time.time()

        return sum(epoch_losses) / len(epoch_losses)

    def evaluate(self, test=False,):
        losses = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            self.model.eval()

            if test:
                dataloader = self.test_dataloader
            else:
                dataloader = self.valid_dataloader

            for *inputs, labels in dataloader:

                pred, h = self.model(*inputs)

                loss = self.criterion(pred, labels.float())
                losses.append(loss.item())
                predictions.extend([1 if f > 0.5 else 0 for f in pred])
                true_labels.extend([f for f in labels])

            predictions = torch.tensor(predictions)
            true_labels = torch.tensor(true_labels)
            result = Results(predictions, true_labels, np.mean(losses))
            return result

    # TODO This will also be moved to a logger
    @staticmethod
    def plot_confusion_matrix(predictions, labels):
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        fig = disp.figure_
        return fig

    def save_model(self, epoch, loss, accuracy):
        """Saves the model and optimizer state. Also saves loss and acc"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }, self.dirpath.joinpath("_model.pt"))

    def load_model(self):
        """Loads the last saved results of the model state dict"""
        checkpoint = torch.load(self.dirpath.joinpath("_model.pt"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return

    def run(self, log_every):

        self.logger.debug(
            f"Experiment Name: {self.name}\n\n{50 * 'x'}\n"
            f"Configurations\n{50 * 'x'}\n\n"

            f"{30 * '-'}\nSeeds\n{30 * '-'}\n"
            f"{pformat(dict(self.config.seeds))}\n\n"

            f"{30 * '-'}\nFilepaths\n{30 * '-'}\n"
            f"{pformat(dict(self.config.filepaths))}\n\n"

            f"{30 * '-'}\nLabels\n{30 * '-'}\n"
            f"{dict(self.config.labels)}\n\n"

            f"{30 * '-'}\nDataset Config\n{30 * '-'}\n"
            f"{dict(self.config.dataset)}\n\n"

            f"{30 * '-'}\nModel Config\n{30 * '-'}\n"
            f"{dict(self.config.model)}\n\n"
            f"Model\n\n{self.model}\n\n"

            f"{30 * '-'}\nTraining Config\n{30 * '-'}\n"
            f"{dict(self.config.training)}\n\n"

            f"Optimizer: {str(self.optimizer.__class__)}\n"
            f"Loss: {str(self.criterion)}\n\n")

        best_accuracy = -np.Inf
        start = time.time()

        for i in range(self.epochs):
            self.logger.info(f"{50 * '='}\n{50 * '='}\n"
                             f"Epoch: {i+1}/{self.epochs}\n")
            training_loss = self.train(log_every)

            result = self.evaluate()
            # All this can be a single callback function!
            # TODO Move this to logger
            self.writer.add_scalar('Training Loss', training_loss, i+1)
            self.logger.debug(f"\nTraining Loss: {training_loss:.4f}\n")
            self.logger.info(f"Valid Loss : {result.loss:.4f}\n"
                             f"Valid Accuracy: {result.accuracy:.4f}\n"
                             f"Valid Recall: {result.recall:.4f}\n"
                             f"Valid F1 Score: {result.f1_score:4f}\n")
            # TODO Move this to logger
            self.writer.add_scalar('Validation Loss', result.loss, i+1)
            self.writer.add_figure(f"Epoch: {i+1}",
                                   self.plot_confusion_matrix(
                                       result.pred,
                                       result.labels),
                                   i+1
                                   )

            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                self.save_model(i+1, result.loss, result.accuracy)
                self.logger.info(f"Saving model after epoch: {i+1}\n")

        # Load model checkpoint
        self.load_model()
        test_results = self.evaluate(test=True)
        self.logger.info(f"Test Loss : {test_results.loss:.4f}\n"
                         f"Test Accuracy: {test_results.accuracy:.4f}\n"
                         f"Test Recall: {test_results.recall:.4f}\n"
                         f"Test F1 Score: {test_results.f1_score:4f}\n")
        # TODO Move this to logger
        self.writer.add_figure(f"Test Set",
                               self.plot_confusion_matrix(
                                   test_results.pred,
                                   test_results.labels),
                               0)

        end = time.time()
        self.logger.info(f"{50 * 'x'}\n"
                         f"Time Taken: {end - start:.4f}s\n"
                         f"{50 * 'x'}\n")