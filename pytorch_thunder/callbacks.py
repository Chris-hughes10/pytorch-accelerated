import logging
from abc import ABC
from datetime import datetime
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class StopTrainingError(Exception):
    pass


class TrainerCallback(ABC):
    def on_init_end(self, trainer, **kwargs):
        """
        Event called at the end of the initialization.
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
        Event called at the beginning of an epoch.
        """
        pass

    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_eval_epoch_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of evaluation.
        """
        pass

    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of evaluation.
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

    def on_eval_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_eval_step_end(self, trainer, **kwargs):
        """
        Event called at the end of a training step.
        """
        pass

    def on_stop_training_error(self, trainer, **kwargs):
        pass


class CallbackHandler(TrainerCallback):
    """class that just calls the list of callbacks in order."""

    def __init__(self, callbacks):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

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
    def on_train_epoch_end(self, trainer, **kwargs):
        trainer._accelerator.print(
            f"training loss: {trainer.run_history['metrics']['train_loss_epoch'][-1]}"
        )

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer._accelerator.print(
            f"validation loss: {trainer.run_history['metrics']['eval_loss_epoch'][-1]}"
        )


class PrintProgressCallback(TrainerCallback):
    @staticmethod
    def print(trainer, message):
        if trainer.run_config["is_local_process_zero"]:
            trainer._accelerator.print(message)

    def on_train_run_begin(self, trainer, **kwargs):
        self.print(trainer, "Starting run")

    def on_train_epoch_begin(self, trainer, **kwargs):
        self.print(trainer, f"Starting epoch {trainer.run_history['epoch']}")

    def on_train_run_end(self, trainer, **kwargs):
        self.print(trainer, "Finishing run")


class SaveBestModelCallback(TrainerCallback):
    def __init__(
        self,
        save_dir=None,
        watch_metric="eval_loss_epoch",
        greater_is_better=False,
    ):
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.save_dir = save_dir

    def on_train_run_begin(self, args, **kwargs):
        self.best_metric = None
        if self.save_dir is None:
            self.save_dir = str(datetime.now()).split(".")[0].replace(" ", "_")

    def save_model(self, trainer):
        if trainer.run_config["is_local_process_zero"]:
            torch.save(
                {
                    "loss": self.best_metric,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                },
                self.save_dir,
            )

    def on_eval_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history["metrics"][self.watch_metric][-1]
        if self.best_metric is None:
            self.best_metric = current_metric
            self.save_model(trainer)
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self.best_metric = current_metric
                self.save_model(trainer)


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
        current_metric = trainer.run_history["metrics"][self.watch_metric][-1]
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

    def check_for_nan_after_batch(self, batch_output):
        """Test if loss is NaN and interrupts training."""
        loss = batch_output["loss"]
        if torch.isinf(loss) or torch.isnan(loss):
            raise StopTrainingError("Stopping training due to NaN loss")

    def on_train_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output)

    def on_eval_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output)
