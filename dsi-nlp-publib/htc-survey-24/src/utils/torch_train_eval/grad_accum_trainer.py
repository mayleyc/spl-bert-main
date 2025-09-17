from typing import Dict, Callable, Any

import torch.utils.data as td
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from . import evaluation as val
from .early_stopper import EarlyStopping
from .model import TrainableModel
from .trainer import Trainer


class GradientAccumulatorTrainer(Trainer):

    def __init__(self, model: TrainableModel, train_params: Dict[str, Any], loss_fn: Callable, optimizer: Optimizer,
                 es: EarlyStopping, metrics: val.MetricSet = None, unpack_flag: bool = False):
        """

        @param model: model to train
        @param train_params: dictionary of parameters for training
        @param loss_fn: loss function object
        @param optimizer: optimizer object
        @param es: early stopper object.
        """
        super().__init__(model, train_params, loss_fn, optimizer, es, metrics, unpack_flag)
        self.num_accumulations: int = train_params["simulated_bs"] // train_params["BATCH_SIZE"]

    def train_one_epoch(self, training_loader: td.DataLoader, epoch_num: int):
        """
        Train a single epoch.

        @param training_loader: Data loader for a single epoch
        @param epoch_num: current epoch number
        """
        # Visual progress bar
        tqdm_bar = tqdm(training_loader, total=len(training_loader), unit="batch")
        tqdm_bar.set_description_str("Training  ")  # Spaces to pad it to 10 letters
        # Accumulated and last loss for this epoch
        running_loss = 0.
        last_loss = 0.

        # Process the batch
        for i, data in enumerate(training_loader):
            tqdm_bar.update(1)
            # ---------------------- UPDATE ------------------------
            x, y = self.data_unpack(data)
            # 1. Pass input to the models to get outputs
            # --- FORWARD PASS ---
            # 1. Pass input to the models to get outputs
            outputs = self.model(x)
            # 2. Pass resulting outputs and eventual additional arguments to the loss function
            # NOTE: y is never passed (suppose unpack_flag=False)
            loss = self._loss_fn(**outputs, device=self.device) if isinstance(outputs, dict) else self._loss_fn(
                *outputs, device=self.device)
            # Normalize loss to account for batch accumulation
            loss = loss / self.num_accumulations
            # --- BACKWARD PASS ---
            loss.backward()
            # --- WEIGHT UPDATE ---
            if ((i + 1) % self.num_accumulations == 0) or (i + 1 == len(training_loader)):
                self._optimizer.step()
                self._optimizer.zero_grad()  # Reset gradient for batch
            # ---
            running_loss += loss.item()
            # ---------------------- REPORTING ------------------------
            if i % self._log_every_batches == 0:
                last_loss: float = running_loss / self._log_every_batches  # loss per batch
                running_loss = 0.
                progress: str = f"loss: {last_loss:.4f}"
                tqdm_bar.set_postfix_str(progress)
                # TensorBoard
                if self.board:
                    tb_x = epoch_num * len(training_loader) + i + 1
                    self.board.add_scalar("Loss/train", last_loss, tb_x)
        tqdm_bar.close()
        return last_loss
