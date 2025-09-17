import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Callable, Any, Tuple, Optional, Union

import torch
import torch.utils.data as td
import yaml
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import evaluation as val
from .early_stopper import EarlyStopping
from .model import TrainableModel
from ..generic_functions import extract_number, random_string

"""
Usage:
    Instantiate the Trainer with:
        - the model instance (type `model.TrainableModel`)
        - the dictionary of parameters
        - a callable loss function
        - the optimizer instance (of type `torch.optim.Optimizer`)
        - the Early stopping object (of type `early_stopper.EarlyStopping` or None to disable it)
        - the metric set, using torchmetrics's metrics
    The loss is a callable object/function that takes the same arguments returned by model's forward method.
"""


class Trainer:
    # Names to assign to checkpoint objects
    _epoch_state_name = "epoch.json"
    _model_checkpoint_name = "model.pt"
    _opt_checkpoint_name = "optimizer.pt"

    def __init__(self, model: TrainableModel, train_params: Dict[str, Any],
                 loss_fn: Callable, optimizer: Optimizer, es: EarlyStopping, metrics: val.MetricSet = None,
                 unpack_flag: bool = False, add_start_time_folder: bool = True):
        """

        @param model: model to train
        @param train_params: dictionary of parameters for training
        @param loss_fn: loss function object
        @param optimizer: optimizer object
        @param es: early stopper object.
        """
        # Path to folder for dumps
        self._model_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_folder = Path(f"{train_params['MODEL_FOLDER']}_{self._model_start_time}") if add_start_time_folder \
            else Path(train_params["MODEL_FOLDER"])
        # Generic parameters
        self.device: torch.device = torch.device(train_params["DEVICE"])
        self._epochs: int = train_params["EPOCHS"]
        self._start_epoch: int = 0
        self._log_every_batches: int = 100
        self._unpack_batch = unpack_flag
        # Loss function, Optimizer
        self._loss_fn: Callable = loss_fn
        self._optimizer = optimizer
        # Evaluation metrics
        self._metrics: val.MetricSet = metrics if metrics is not None else val.MetricSet()
        self._metrics._device = self.device
        self._metrics_old_value: Optional[Dict[str, float]] = None
        self._validation_loss_old_value: Optional[float] = None
        self._evaluate_metrics_every_k_epochs: int = train_params.get("EVALUATE_EVERY_K_EPOCHS", 1)
        # Torch device (CPU, GPU)
        self.model: Any[TrainableModel, torch.nn.Module] = model.to(self.device)
        self.model.device = self.device
        # Add the device attribute to all submodules
        for sm in self.model.submodules():
            sm.device = self.device
            sm.to(self.device)
        # Early stopper
        self.early_stopper: EarlyStopping = es
        self.epochs_to_keep: int = train_params["EPOCHS_TO_KEEP"]
        self.always_save: bool = train_params["SAVE_NON_IMPROVING"]
        # If reload_path model
        reload: bool = train_params.get("RELOAD", None)
        self.last_saved_checkpoint: Path = None
        if reload:
            self.model_folder = self.load_previous(reload_path=train_params["PATH_TO_RELOAD"])
            self.last_saved_checkpoint: Path = Path(train_params["PATH_TO_RELOAD"])
        # Board writer
        enable_board: bool = train_params.get("TENSORBOARD", None)
        self.board = None
        if enable_board:
            self.board = SummaryWriter(str(self.model_folder / f"Tensorboard_{self._model_start_time}"))
        # Save model configuration
        self._save_train_specs(train_params)

    def _save_train_specs(self, train_params):
        # Save training parameters
        self.model_folder.mkdir(parents=True, exist_ok=True)
        with open(self.model_folder / "train_config.config.yml", mode="w", encoding="UTF-8") as f:
            yaml.dump(train_params, f)

    def _save_run_specs(self):
        # Save run specifics
        run_specs: Dict = {
            "model_name": self.model.__class__.__name__,
            "model_class_src": self.model.__module__,
            "kwargs": {**self.model.constructor_args()}
        }
        with open(self.model_folder / "run_config.json", mode="w", encoding="UTF-8") as f:
            json.dump(run_specs, f, indent=4)

    def data_unpack(self, data_batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Takes a batch yielded by the dataloader object.
        If 'unpack_flag' is set it assumes every dataset sample can be unpacked in X, y,
        which are respectively passed to the model and the loss/metric functions
        Otherwise it assumes nothing about the dataset shape, and passes the whole batch to the model

        @param data_batch: a batch of samples given by a dataloader object
        @return: a pair of objects, the first must be passed to the model,
            the second (if present) must be passed to loss/metrics in addition to the model outputs
        """
        if self._unpack_batch:
            inputs, labels = data_batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            return inputs, labels
        return data_batch, None

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
            self._optimizer.zero_grad()  # Reset gradient for batch
            # ---------------------- UPDATE ------------------------
            x, y = self.data_unpack(data)
            # 1. Pass input to the models to get outputs
            outputs = self.model(x)
            # 2. Pass resulting outputs and eventual additional arguments to the loss function
            # NOTE: y is never passed (suppose unpack_flag=False)
            loss = self._loss_fn(**outputs, device=self.device) if isinstance(outputs, dict) else self._loss_fn(
                *outputs, device=self.device)
            loss.backward()
            self._optimizer.step()
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

    def validate_one_epoch(self, validation_loader: td.DataLoader, do_metrics: bool):
        tqdm_bar = tqdm(validation_loader, total=len(validation_loader), unit="batch")
        tqdm_bar.set_description_str("Validating")  # Spaces to pad it to 10 letters
        # ---
        running_val_loss: float = 0.
        i: int = 0
        for i, val_data in enumerate(validation_loader):
            tqdm_bar.update(1)
            # ---------------------- UPDATE ------------------------
            val_x, val_y = self.data_unpack(val_data)
            val_outputs = self.model(val_x)
            # Evaluate metrics
            if do_metrics:
                metrics = self._metrics(val_outputs, val_y) if val_y is not None else self._metrics(val_outputs)
            else:
                metrics = dict()
            print_metrics: str = val.format_metrics(metrics)
            # Loss computation
            val_loss = self._loss_fn(**val_outputs, device=self.device) if isinstance(val_outputs,
                                                                                      dict) else self._loss_fn(
                *val_outputs, device=self.device)
            running_val_loss += val_loss.item()
            # ---------------------- REPORTING ------------------------
            # Visual progress
            if i % self._log_every_batches == 0:
                progress: str = f"val_loss: {val_loss:.4f} {print_metrics}"
                tqdm_bar.set_postfix_str(progress)
        avg_val_loss: float = running_val_loss / (i + 1)
        tqdm_bar.close()
        # Final metric computation
        metric_values = {k: v.cpu().item() for k, v in self._metrics.compute().items()} if do_metrics else dict()
        # Return all values
        return avg_val_loss, metric_values

    def train(self, training_loader: td.DataLoader, validation_loader: Optional[td.DataLoader] = None):
        self._save_run_specs()
        for epoch_number in range(self._start_epoch, self._epochs):
            print(f"--- EPOCH {epoch_number + 1} / {self._epochs} ---")
            # ---------------------- UPDATE ------------------------
            # -- TRAINING --
            # Gradient on for training
            self.model.train(True)
            avg_loss: float = self.train_one_epoch(training_loader, epoch_number)

            # -- VALIDATION --
            avg_val_loss = .0
            metric_values = dict()
            do_evaluation: bool = ((epoch_number + 1) % self._evaluate_metrics_every_k_epochs) == 0
            if validation_loader is not None:
                # Gradient off for reporting
                self.model.train(False)
                with torch.no_grad():
                    avg_val_loss, metric_values = self.validate_one_epoch(validation_loader, do_evaluation)
            # ---------------------- REPORTING ------------------------
            # -- Write to board ---
            if self.board:
                self.board.add_scalars("Training vs. Validation Loss",
                                       {"Training": avg_loss, "Validation": avg_val_loss},
                                       epoch_number + 1)
                self.board.flush()
            # -- Print end of epoch results --

            old_val_loss_value_print = self._validation_loss_old_value if self._validation_loss_old_value else 0.0
            print(f"Epoch results || {'Train loss':<10}: {avg_loss:>10.4f}")
            if validation_loader is not None:
                print(
                    f"              || {'Val loss':<10}: {old_val_loss_value_print:>10.4f} {'->':^5} {avg_val_loss:>4.4f}")
            self._validation_loss_old_value = avg_val_loss
            for metric_name, metric_value in metric_values.items():
                # Keep track of the validation metrics/loss in previous epoch
                old_value = self._metrics_old_value[metric_name] if self._metrics_old_value else 0.0
                print(f"              || {metric_name:<10}: {old_value:>10.4f} {'->':^5} {metric_value:>4.4f} ")
            if do_evaluation:  # if metrics were not computed keep old values
                self._metrics_old_value = metric_values
            # -- SAVE --
            es_report = {"loss": avg_val_loss, **metric_values}
            has_improved: bool = False  # flag to decide whether to save model
            if validation_loader is not None and do_evaluation:
                # Do ES check only when validation set is provided and according to evaluation frequency
                has_improved = self.early_stopper is None or self.early_stopper(es_report)
            elif validation_loader is None:
                # If no validation set, save all epochs
                has_improved = True
            # else save checkpoint
            # Always save if improved, or if flag enabled
            if self.always_save or has_improved:
                self.save_epoch_checkpoint(epoch_number, es_report)
            if self.early_stopper is not None and self.early_stopper.exit_flag:
                break

    def save_epoch_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save a model + optimizer to a specific folder.

        :param epoch: epoch number, used for folder naming
        :param metrics: dictionary of metrics of current epoch to save
        """
        # --- SAVE MODEL ---
        # Create root model dir if it does not exist
        self.model_folder.mkdir(exist_ok=True)
        # Create subfolder for the epoch dump
        current_dump_folder = self.model_folder / f"epoch_{epoch + 1}"
        # If epoch folder already exists, move it to new directory
        if current_dump_folder.is_dir() and current_dump_folder.exists():
            # If an epoch checkpoint with this number is already present, move it to a new directory and warn the user
            new_name = str(current_dump_folder) + "_old_" + random_string()
            logging.warning(
                f"Dump folder '{current_dump_folder}' already exists. You are probably overwriting an "
                f"existing epoch dump. The old one will be renamed {new_name}")
            # Rename the old checkpoint
            shutil.move(str(current_dump_folder), new_name)
            # shutil.rmtree(current_dump_folder)
        # Create target dump directory
        current_dump_folder.mkdir(exist_ok=False)

        # Save model weights and optimizer state
        torch.save(self.model.state_dict(), current_dump_folder / self._model_checkpoint_name)
        torch.save(self._optimizer.state_dict(), current_dump_folder / self._opt_checkpoint_name)

        # Save epoch number and metrics.
        # We save the current epoch number (counting from 0), the metrics dictionary and the best score saved in the ES
        # Currently we do not save other configurations like the watched metric name in the ES, delta value and patience counter
        epoch_state_data = dict(epoch_number=epoch, metrics=metrics,
                                best_score=self.early_stopper.best_score if self.early_stopper is not None else None)
        with open(current_dump_folder / self._epoch_state_name, mode="w") as ed_file:
            json.dump(epoch_state_data, ed_file)

        # Only keep last N epochs
        epoch_folders_list = [f for f in [x[0] for x in os.walk(self.model_folder)][1:] if re.match(".+epoch_\\d+$", f)]
        if len(epoch_folders_list) > self.epochs_to_keep:
            earliest_epoch = min(epoch_folders_list, key=extract_number)
            shutil.rmtree(earliest_epoch)
            # Sanity check
            assert len([f for f in [x[0] for x in os.walk(self.model_folder)][1:] if
                        re.match(".+epoch_\\d+$", f)]) <= self.epochs_to_keep

        # Remember the path to the last checkpoint (not used by Trainer, but for convenience)
        self.last_saved_checkpoint = current_dump_folder

    def load_previous(self, reload_path: Union[Path, str], model_only: bool = False) -> Path:
        """
        Load previously saved model. If the path passed is to a specific epoch that epoch will be loaded.
        If the path points to a dump folder with multiple epochs the last one will be loaded

        @param reload_path: path to model checkpoint, either to main folder or to specific epoch
        @param model_only: Only load weights and epoch state (not the optimizer)
        """
        # Folder of model
        reload_path = Path(reload_path)
        model_path = reload_path / self._model_checkpoint_name
        if reload_path.is_dir() and (not model_path.is_file()):
            # Passed path is the root folder of a dump, in this case the last epoch is loaded
            print("*** Loading the last epoch ... ***")
            epoch_folders_list = [x[0] for x in os.walk(reload_path)][1:]
            folder_to_load = max(epoch_folders_list, key=extract_number)
            folder_to_load = Path(folder_to_load)
            model_folder = reload_path
        else:
            # Else, the model path and opt path point to the specific epoch folder, so we load them directly
            model_folder = reload_path.parent
            folder_to_load = reload_path

        model_path = folder_to_load / self._model_checkpoint_name
        epoch_state_path = folder_to_load / self._epoch_state_name
        if not model_only:
            optimizer_path = folder_to_load / self._opt_checkpoint_name

        # Load model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # Load epoch state
        with open(epoch_state_path, mode="r") as ed_file:
            data: Dict = json.load(ed_file)
        self._validation_loss_old_value = data["metrics"].pop("loss")
        self._metrics_old_value = data["metrics"]
        if self.early_stopper is not None:
            self.early_stopper.best_score = data["best_score"]
        self._start_epoch = data["epoch_number"] + 1
        # Load optimizer state
        if not model_only:
            self._optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        print(f"*** Loaded model state from {str(folder_to_load):s}")
        return model_folder
