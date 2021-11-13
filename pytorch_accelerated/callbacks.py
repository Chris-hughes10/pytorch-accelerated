# Copyright © 2021 Chris Hughes
import logging
import sys
import time
from abc import ABC
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StopTrainingError(Exception):
    pass


class TrainerCallback(ABC):
    def on_init_end(self, trainer, **kwargs):
        """
        Event called at the end of trainer initialisation.
        """
        pass

    def on_train_run_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of training run.
        """
        pass

    def on_train_run_end(self, trainer, **kwargs):
        """
        Event called at the end of training run.
        """
        pass

    def on_train_epoch_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """
        pass

    def on_train_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_train_step_end(self, trainer, **kwargs):
        """
        Event called at the end of a training step.
        """
        pass

    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of a training epoch.
        """
        pass

    def on_eval_epoch_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of an evaluation epoch.
        """
        pass

    def on_eval_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_eval_step_end(self, trainer, **kwargs):
        """
        Event called at the end of an evaluation step.
        """
        pass

    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of evaluation.
        """
        pass

    def on_stop_training_error(self, trainer, **kwargs):
        pass


class CallbackHandler:
    """
    Responsible for calling a list of callbacks. This class calls the callbacks in the order that they are given.
    """

    def __init__(self, callbacks):
        self.callbacks = []
        self.add_callbacks(callbacks)

    def add_callbacks(self, callbacks):
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            raise ValueError(
                f"You attempted to add multiple instances of the callback {cb_class} to a single Trainer"
                f" The list of callbacks already present is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    def __iter__(self):
        return self.callbacks

    def clear_callbacks(self):
        self.callbacks = []

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def call_event(self, event, *args, **kwargs):
        for callback in self.callbacks:
            getattr(callback, event)(
                *args,
                **kwargs,
            )


class PrintMetricsCallback(TrainerCallback):
    def _print_metrics(self, trainer, metric_names):
        for metric_name in metric_names:
            trainer.print(
                f"\n{metric_name}: {trainer.run_history.get_latest_metric(metric_name)}"
            )

    def on_train_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" in metric
        ]

        self._print_metrics(trainer, metric_names)

    def on_eval_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" not in metric
        ]
        self._print_metrics(trainer, metric_names)


class ProgressBarCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_epoch_begin(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._train_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_train_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)

    def on_eval_epoch_begin(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._eval_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_eval_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)


class PrintProgressCallback(TrainerCallback):
    def on_train_run_begin(self, trainer, **kwargs):
        trainer.print("\nStarting training run")

    def on_train_epoch_begin(self, trainer, **kwargs):
        trainer.print(f"\nStarting epoch {trainer.run_history.current_epoch}")
        time.sleep(0.01)

    def on_train_run_end(self, trainer, **kwargs):
        trainer.print("Finishing training run")


class SaveBestModelCallback(TrainerCallback):
    def __init__(
        self,
        save_dir=None,
        watch_metric="eval_loss_epoch",
        greater_is_better=False,
        reset_on_train=True,
    ):
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.save_dir = save_dir
        self.reset_on_train = reset_on_train

    def on_train_run_begin(self, args, **kwargs):
        if self.save_dir is None:
            self.save_dir = "model.pt"

        if self.reset_on_train:
            self.best_metric = None

    def on_eval_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
            trainer.save_model(
                save_path=self.save_dir,
                checkpoint_kwargs={self.watch_metric: self.best_metric},
            )
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self.best_metric = current_metric
                trainer.save_model(
                    save_path=self.save_dir,
                    checkpoint_kwargs={"loss": self.best_metric},
                )

    def on_train_run_end(self, trainer, **kwargs):
        trainer.print(
            f"Loading checkpoint with {self.watch_metric}: {self.best_metric}"
        )
        trainer.load_checkpoint(self.save_dir)


class EarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: Optional[float] = 0.01,
        watch_metric="eval_loss_epoch",
        greater_is_better=False,
        reset_on_train=True,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.early_stopping_patience_counter = 0
        self.best_metric = None
        self.operator = np.greater if self.greater_is_better else np.less
        self.reset_on_train = reset_on_train

    def on_train_run_begin(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None
            self.early_stopping_patience_counter = 0

    def on_eval_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            is_improvement = self.operator(current_metric, self.best_metric)
            improvement_above_threshold = (
                abs(current_metric - self.best_metric) > self.early_stopping_threshold
            )

            if is_improvement and improvement_above_threshold:
                self.best_metric = current_metric
                self.early_stopping_patience_counter = 0
            else:
                self.early_stopping_patience_counter += 1

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            raise StopTrainingError(
                f"Stopping training due to no improvement after {self.early_stopping_patience} epochs"
            )


class TerminateOnNaNCallback(TrainerCallback):
    """A callback that terminates training if loss is NaN."""

    def __init__(self):
        self.triggered = False

    def check_for_nan_after_batch(self, batch_output, step=None):
        """Test if loss is NaN and interrupts training."""
        loss = batch_output["loss"]
        if torch.isinf(loss) or torch.isnan(loss):
            self.triggered = True
            raise StopTrainingError(f"Stopping training due to NaN loss in {step} step")

    def on_train_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="training")

    def on_eval_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="validation")

    def on_train_run_end(self, trainer, **kwargs):
        if self.triggered:
            sys.exit("Exiting due to NaN loss")
