# Copyright © 2021 Chris Hughes
from datetime import datetime
import inspect
import logging
import sys
import time
from abc import ABC
from pathlib import Path
from typing import Optional, Union


import numpy as np
import torch
from pytorch_accelerated.tracking import LossTracker
from pytorch_accelerated.utils import DataLoaderSlice, ModelEma
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_default_callbacks(progress_bar=True):
    if progress_bar:
        default_callbacks = (
            MoveModulesToDeviceCallback,
            TerminateOnNaNCallback,
            PrintProgressCallback,
            ProgressBarCallback,
            LogMetricsCallback,
        )
    else:
        default_callbacks = (
            MoveModulesToDeviceCallback,
            TerminateOnNaNCallback,
            PrintProgressCallback,
            LogMetricsCallback,
        )

    return default_callbacks


class StopTrainingError(Exception):
    """
    An exception which can be raised in order to stop a training run early.
    """

    pass


class CallbackMethodNotImplementedError(Exception):
    pass


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """

    def on_init_end(self, trainer, **kwargs):
        """
        Event called at the end of trainer initialisation.
        """
        pass

    def on_training_run_start(self, trainer, **kwargs):
        """
        Event called at the start of training run.
        """
        pass

    def on_train_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training epoch.
        """
        pass

    def on_train_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_train_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of a training step.

        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss`
        """
        pass

    def on_train_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of a training epoch.
        """
        pass

    def on_eval_epoch_start(self, trainer, **kwargs):
        """
        Event called at the beginning of an evaluation epoch.
        """
        pass

    def on_eval_step_start(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of an evaluation step.

        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        pass

    def on_eval_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of evaluation.
        """
        pass

    def on_training_run_epoch_end(self, trainer, **kwargs):
        """
        Event called during a training run after both training and evaluation epochs have been completed.
        """
        pass

    def on_training_run_end(self, trainer, **kwargs):
        """
        Event called at the end of training run.
        """
        pass

    def on_evaluation_run_start(self, trainer, **kwargs):
        """
        Event called at the start of an evaluation run.
        """
        pass

    def on_evaluation_run_end(self, trainer, **kwargs):
        """
        Event called at the end of an evaluation run.
        """
        pass

    def on_stop_training_error(self, trainer, **kwargs):
        """
        Event called when a stop training error is raised
        """
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            raise CallbackMethodNotImplementedError


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """

    def __init__(self, callbacks):
        self.callbacks = []
        self.add_callbacks(callbacks)
        self._enabled = True

    def add_callbacks(self, callbacks):
        """
        Add a list of callbacks to the callback handler

        :param callbacks: a list of :class:`TrainerCallback`
        """
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        """
        Add a callbacks to the callback handler

        :param callback: an instance of a subclass of :class:`TrainerCallback`
        """
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in {c.__class__ for c in self.callbacks}:
            existing_callbacks = "\n".join(cb for cb in self.callback_list)

            raise ValueError(
                f"You attempted to add multiple instances of the callback {cb_class} to a single Trainer"
                f" The list of callbacks already present is\n: {existing_callbacks}"
            )
        self.callbacks.append(cb)

    def __iter__(self):
        return self.callbacks

    def clear_callbacks(self):
        self.callbacks = []

    @property
    def callback_list(self):
        return [cb.__class__.__name__ for cb in self.callbacks]

    def call_event(self, event, *args, **kwargs):
        """
        For each callback which has been registered, sequentially call the method corresponding to the
        given event.

        :param event: The event corresponding to the method to call on each callback
        :param args: a list of arguments to be passed to each callback
        :param kwargs: a list of keyword arguments to be passed to each callback
        """
        if self._enabled:
            for callback in self.callbacks:
                try:
                    getattr(callback, event)(
                        *args,
                        **kwargs,
                    )
                except CallbackMethodNotImplementedError as e:
                    continue


class LogMetricsCallback(TrainerCallback):
    """
    A callback that logs the latest values of any metric which has been updated in
    the trainer's run history. By default, this just prints to the command line once per machine.

    Metrics prefixed with 'train' are logged at the end of a training epoch, all other
    metrics are logged after evaluation.

    This can be subclassed to create loggers for different platforms by overriding the :meth:`~LogMetricsCallback.log_metrics` method.
    """

    def on_train_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" in metric
        ]

        self._log_latest_metrics(trainer, metric_names)

    def on_eval_epoch_end(self, trainer, **kwargs):
        metric_names = [
            metric
            for metric in trainer.run_history.get_metric_names()
            if "train" not in metric
        ]
        self._log_latest_metrics(trainer, metric_names)

    def _log_latest_metrics(self, trainer, metric_names):
        latest_metrics = self._get_latest_metrics(trainer, metric_names)
        self.log_metrics(trainer, latest_metrics)

    def _get_latest_metrics(self, trainer, metric_names):
        return {
            metric_name: trainer.run_history.get_latest_metric(metric_name)
            for metric_name in metric_names
        }

    def log_metrics(self, trainer, metrics: dict):
        for metric_name, metric_value in metrics.items():
            trainer.print(f"\n{metric_name}: {metric_value}")


class ProgressBarCallback(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        self.pbar = None
        self.eval_pbar = None

    def on_train_epoch_start(self, trainer, **kwargs):
        self.pbar = tqdm(
            total=len(trainer._train_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_train_step_end(self, trainer, **kwargs):
        self.pbar.update(1)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.pbar.close()
        time.sleep(0.01)

    def on_eval_epoch_start(self, trainer, **kwargs):
        self.eval_pbar = tqdm(
            total=len(trainer._eval_dataloader),
            disable=not trainer._accelerator.is_local_main_process,
        )

    def on_eval_step_end(self, trainer, **kwargs):
        self.eval_pbar.update(1)

    def on_eval_epoch_end(self, trainer, **kwargs):
        self.eval_pbar.close()
        time.sleep(0.01)


class PrintProgressCallback(TrainerCallback):
    """
    A callback which prints a message at the start and end of a run,
    as well as at the start of each epoch.
    """

    def on_training_run_start(self, trainer, **kwargs):
        trainer.print("\nStarting training run")

    def on_train_epoch_start(self, trainer, **kwargs):
        trainer.print(f"\nStarting epoch {trainer.run_history.current_epoch}")
        time.sleep(0.01)

    def on_training_run_end(self, trainer, **kwargs):
        trainer.print("Finishing training run")

    def on_eval_epoch_start(self, trainer, **kwargs):
        trainer.print(f"\nStarting eval epoch")

    def on_evaluation_run_start(self, trainer, **kwargs):
        trainer.print("\nStarting evaluation run")

    def on_evaluation_run_end(self, trainer, **kwargs):
        trainer.print("Finishing evaluation run")


class SaveBestModelCallback(TrainerCallback):
    """
    A callback which saves the best model during a training run, according to a given metric.
    The best model weights are loaded at the end of the training run.
    """

    def __init__(
        self,
        save_path="best_model.pt",
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        load_saved_checkpoint: bool = True,
    ):
        """

        :param save_path: The path to save the checkpoint to. This should end in ``.pt``.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param reset_on_train: whether to reset the best metric on subsequent training runs. If ``True``, only the metrics observed during the current training run will be compared.
        :param save_optimizer: whether to also save the optimizer as part of the model checkpoint
        :param save_scheduler: whether to also save the scheduler as part of the model checkpoint
        :param load_saved_checkpoint: whether to load the saved checkpoint at the end of the training run
        """
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.operator = np.greater if self.greater_is_better else np.less
        self.best_metric = None
        self.best_metric_epoch = None
        self.save_path = save_path
        self.reset_on_train = reset_on_train
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.load_saved_checkpoint = load_saved_checkpoint

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_metric_epoch = trainer.run_history.current_epoch
            trainer.save_checkpoint(
                save_path=self.save_path,
                checkpoint_kwargs={self.watch_metric: self.best_metric},
                save_optimizer=self.save_optimizer,
            )
        else:
            is_improvement = self.operator(current_metric, self.best_metric)

            if is_improvement:
                self.best_metric = current_metric
                self.best_metric_epoch = trainer.run_history.current_epoch
                trainer.save_checkpoint(
                    save_path=self.save_path,
                    checkpoint_kwargs={"loss": self.best_metric},
                    save_optimizer=self.save_optimizer,
                    save_scheduler=self.save_scheduler,
                )

    def on_training_run_end(self, trainer, **kwargs):
        if self.load_saved_checkpoint:
            trainer.print(
                f"Loading checkpoint with {self.watch_metric}: {self.best_metric} from epoch {self.best_metric_epoch}"
            )
            trainer.load_checkpoint(self.save_path)


class EarlyStoppingCallback(TrainerCallback):
    """
    A callback which stops training early if progress is not being observed.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.01,
        watch_metric="eval_loss_epoch",
        greater_is_better: bool = False,
        reset_on_train: bool = True,
    ):
        """

        :param early_stopping_patience: the number of epochs with no improvement after which training will be stopped.
        :param early_stopping_threshold: the minimum change in the ``watch_metric`` to qualify as an improvement, i.e. an absolute change of less than this threshold, will count as no improvement.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param reset_on_train: whether to reset the best metric on subsequent training runs. If ``True``, only the metrics observed during the current training run will be compared.
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.watch_metric = watch_metric
        self.greater_is_better = greater_is_better
        self.early_stopping_patience_counter = 0
        self.best_metric = None
        self.operator = np.greater if self.greater_is_better else np.less
        self.reset_on_train = reset_on_train

    def on_training_run_start(self, args, **kwargs):
        if self.reset_on_train:
            self.best_metric = None
            self.early_stopping_patience_counter = 0

    def on_training_run_epoch_end(self, trainer, **kwargs):
        current_metric = trainer.run_history.get_latest_metric(self.watch_metric)
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            is_improvement = self.operator(current_metric, self.best_metric)
            improvement = abs(current_metric - self.best_metric)
            improvement_above_threshold = improvement > self.early_stopping_threshold

            if is_improvement and improvement_above_threshold:
                trainer.print(
                    f"\nImprovement of {improvement} observed, resetting counter. "
                )
                self.best_metric = current_metric
                self.early_stopping_patience_counter = 0
                self.__print_counter_status(trainer)
            else:
                trainer.print(
                    "No improvement above threshold observed, incrementing counter. "
                )
                self.early_stopping_patience_counter += 1
                self.__print_counter_status(trainer)

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            raise StopTrainingError(
                f"Stopping training due to no improvement after {self.early_stopping_patience} epochs"
            )

    def __print_counter_status(self, trainer):
        trainer.print(
            f"Early stopping counter: {self.early_stopping_patience_counter}/{self.early_stopping_patience}"
        )


class TerminateOnNaNCallback(TrainerCallback):
    """
    A callback that terminates the training run if a ``NaN`` loss is observed during either training or
    evaluation.
    """

    def __init__(self):
        self.triggered = False

    def check_for_nan_after_batch(self, batch_output, step=None):
        """Test if loss is NaN and interrupts training."""
        loss = batch_output["loss"]
        if torch.isinf(loss).any() or torch.isnan(loss).any():
            self.triggered = True
            raise StopTrainingError(f"Stopping training due to NaN loss in {step} step")

    def on_train_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="training")

    def on_eval_step_end(self, trainer, batch_output, **kwargs):
        self.check_for_nan_after_batch(batch_output, step="validation")

    def on_training_run_end(self, trainer, **kwargs):
        if self.triggered:
            sys.exit("Exiting due to NaN loss")


class MoveModulesToDeviceCallback(TrainerCallback):
    """
    A callback which moves any :class:`~pytorch_accelerated.trainer.Trainer` attributes which are instances of
    :class:`torch.nn.Module` on to the appropriate device at the start of a training or evaluation run.

    .. Note::
        This does **not** include the model, as this will be prepared separately by the
        :class:`~pytorch_accelerated.trainer.Trainer`'s internal instance of :class:`accelerate.Accelerator`.

    """

    def _get_modules(self, trainer):
        return inspect.getmembers(trainer, lambda x: isinstance(x, nn.Module))

    def _move_modules_to_device(self, trainer):
        modules = self._get_modules(trainer)

        for module_name, module in modules:
            if module_name != "model":
                module.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_modules_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_modules_to_device(trainer)


class LimitBatchesCallback(TrainerCallback):
    """
    A callback that that limits the number of batches used during training and evaluation.

    This callback will be automatically added to the trainer if the environment variable ``PT_ACC_LIMIT_BATCHES`` is set.
    """

    def __init__(self, num_batches):
        self.num_batches = num_batches

    def on_training_run_start(self, trainer, **kwargs):
        trainer._train_dataloader = DataLoaderSlice(
            trainer._train_dataloader, self.num_batches
        )
        trainer._eval_dataloader = DataLoaderSlice(
            trainer._eval_dataloader, self.num_batches
        )

    def on_evaluation_run_start(self, trainer, **kwargs):
        trainer._eval_dataloader = DataLoaderSlice(
            trainer._eval_dataloader, self.num_batches
        )


class ModelEmaCallback(SaveBestModelCallback):
    """
    A callback which maintains and saves an exponential moving average of the weights of the model that is currently
    being trained.

    This callback offers the option of evaluating the EMA model during. If enabled, this is done by running an additional
    validation after each training epoch, which will use additional GPU resources. During this additional epoch,
    only the provided callbacks will be executed.

    .. Note:: This callback is sensitive to the order that it is executed. This should be placed after any callbacks that
        modify state (e.g. metrics) but before any callbacks that read state (e.g. loggers) or :class:`ConvertSyncBatchNormCallback`.


    """

    def __init__(
        self,
        decay: float = 0.99,
        evaluate_during_training: bool = True,
        save_path: str = "ema_model.pt",
        watch_metric: str = "ema_model_eval_loss_epoch",
        greater_is_better: bool = False,
        model_ema=ModelEma,
        callbacks=(),
    ):
        """
        :param decay: the amount of decay to use, which determines how much of the previous state will be maintained.
        :param evaluate_during_training: whether to evaluate the EMA model during training. If True, an additional validation epoch will be conducted after each training epoch, which will use additional GPU resources, and the best model will be saved. If False, the saved EMA model checkpoint will be updated at the end of each epoch.
        :param watch_metric: the metric used to compare model performance. This should be accessible from the trainer's run history. This is only used when ``evaluate_during_training`` is enabled.
        :param greater_is_better: whether an increase in the ``watch_metric`` should be interpreted as the model performing better.
        :param model_ema: the class which is responsible for maintaining the moving average of the model.
        :param callbacks: an iterable of callbacks that will be executed during the evaluation loop of the EMA model

        """
        super().__init__(
            save_path=save_path,
            watch_metric=watch_metric,
            greater_is_better=greater_is_better,
            reset_on_train=False,
            save_optimizer=False,
        )
        self.decay = decay
        self.ema_model = None
        self._track_prefix = "ema_model_"
        self.evaluate_during_training = evaluate_during_training
        self.model_ema_cls = model_ema
        self.callback_handler = CallbackHandler(callbacks)

    def on_training_run_start(self, trainer, **kwargs):
        self.ema_model = self.model_ema_cls(
            trainer._accelerator.unwrap_model(trainer.model), decay=self.decay
        )
        if self.evaluate_during_training:
            self.ema_model.to(trainer.device)

    def on_train_epoch_end(self, trainer, **kwargs):
        self.ema_model.update(trainer._accelerator.unwrap_model(trainer.model))

    def on_eval_epoch_end(self, trainer, **kwargs):
        if self.evaluate_during_training:
            model = trainer.model
            trainer.model = self.ema_model.module
            run_history_prefix = trainer.run_history.metric_name_prefix
            trainer_callback_handler = trainer.callback_handler

            trainer.print("Running evaluation on EMA model")

            trainer.callback_handler = self.callback_handler
            trainer.run_history.set_metric_name_prefix(self._track_prefix)
            trainer._run_eval_epoch(trainer._eval_dataloader)

            trainer.model = model
            trainer.callback_handler = trainer_callback_handler
            trainer.run_history.set_metric_name_prefix(run_history_prefix)

    def on_training_run_epoch_end(self, trainer, **kwargs):
        model = trainer.model
        trainer.model = self.ema_model.module

        if self.evaluate_during_training:
            super().on_training_run_epoch_end(trainer)
        else:
            trainer.save_checkpoint(save_path=self.save_path, save_optimizer=False)

        trainer.model = model

    def on_training_run_end(self, trainer, **kwargs):
        # Overriding, as we do not want to load the EMA model
        pass


class ConvertSyncBatchNormCallback(TrainerCallback):
    """
    A callback which converts all BatchNorm*D layers in the model to :class:`torch.nn.SyncBatchNorm` layers.
    """

    def on_training_run_start(self, trainer, **kwargs):
        if trainer.run_config.is_distributed:
            trainer.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)


class StepBasedEvaluationCallback(TrainerCallback):
    """
    A callback which enables running an evaluation epoch every N steps during a training epoch.

    :param eval_every_n_steps: the number of steps after which to run an evaluation epoch
    """

    def __init__(self, eval_every_n_steps: int):
        self.eval_every_n_steps = eval_every_n_steps

    def on_train_step_end(self, trainer, step, **kwargs):
        if step != 0 and step % self.eval_every_n_steps == 0:
            trainer.print(f"\nRunning evaluation after {step} training steps")
            original_loss_tracker = trainer._loss_tracker
            trainer._loss_tracker = LossTracker()
            trainer._run_eval_epoch(
                trainer._eval_dataloader,
                is_intermediate=True,
            )
            trainer._loss_tracker = original_loss_tracker
            trainer.run_history.delete_metric("intermediate_eval_loss")
            trainer.print(f"\nResuming training...")


class LimitEvalStepsCallback(TrainerCallback):
    """
    A callback that limits the number of eval steps during an evaluation epoch

    :param num_eval_steps: the total number of evaluation steps to run across all processes
    :param limit_intermediate_only: whether to limit the number of intermediate evaluations only

    .. Note::
        When used together this callback should be placed before :class:`StepBasedEvaluationCallback` and
        :class:`ProgressBarCallback` in the list of callbacks.

    """

    def __init__(self, num_eval_steps: int, limit_intermediate_only=True):
        self.num_eval_steps = num_eval_steps
        self.limit_intermediate_only = limit_intermediate_only
        self._original_eval_dataloader = None

    def on_eval_epoch_start(self, trainer, is_intermediate=False, **kwargs):
        if is_intermediate or not self.limit_intermediate_only:
            self._original_eval_dataloader = trainer._eval_dataloader

            steps_per_process = self.num_eval_steps // trainer.run_config.num_processes

            trainer._eval_dataloader = DataLoaderSlice(
                self._original_eval_dataloader, steps_per_process
            )
            trainer.print(
                f"Limiting evaluation to {steps_per_process} steps per process"
            )

    def on_eval_epoch_end(self, trainer, is_intermediate=False, **kwargs):
        if is_intermediate or not self.limit_intermediate_only:
            trainer._eval_dataloader = self._original_eval_dataloader


class WSDCheckpointCallback(TrainerCallback):
    """Manages checkpointing for WSD and WSD-S learning rate schedules.

    This callback saves both pre-decay and post-decay checkpoints during training with WSD-style
    schedules and automatically syncs with :class:`~pytorch_accelerated.schedulers.wsd_scheduler.WSDLrScheduler` for checkpoint timing.

    For single checkpoint configurations:
        - Pre-decay checkpoint is saved just before learning rate decay starts
        - Post-decay checkpoint is saved at the end of training

    For multiple checkpoints:
        - Pre-decay checkpoint saved before each decay phase
        - Post-decay checkpoint saved after each decay phase

    For WSD vs WSD-S:
        - WSD resumes from pre-decay checkpoints (discarding decay progress)
        - WSD-S resumes from post-decay checkpoints (preserving decay progress)

    :param save_dir: Directory to save checkpoints
    :type save_dir: str
    :param save_optimizer: Whether to save optimizer state
    :type save_optimizer: bool
    :param save_scheduler: Whether to save scheduler state
    :type save_scheduler: bool
    :param initial_checkpoint: Path to checkpoint to load at start of training. For WSD-S,
        use post-decay checkpoint. For WSD, use pre-decay checkpoint.
    :type initial_checkpoint: Union[str, Path], optional

    :raises ValueError: If trainer's scheduler doesn't implement get_checkpoint_steps()

    Example:
        No Checkpoint:
            >>> callback = WSDCheckpointCallback(
            ...     save_dir="checkpoints",
            ... )

        WSD-S usage:
            >>> callback = WSDCheckpointCallback(
            ...     save_dir="checkpoints",
            ...     initial_checkpoint="checkpoint_50000_post_decay.pt"
            ... )

        WSD usage:
            >>> callback = WSDCheckpointCallback(
            ...     save_dir="checkpoints",
            ...     initial_checkpoint="checkpoint_45000_pre_decay.pt"
            ... )
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        initial_checkpoint: Optional[Union[str, Path]] = None,
    ):
        """
        :param save_dir: Directory to save checkpoints
        :param save_optimizer: Whether to save optimizer state
        :param save_scheduler: Whether to save scheduler state
        :param initial_checkpoint: Optional path to checkpoint to load at start of training
        """
        self.save_dir = Path(save_dir)
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.initial_checkpoint = (
            Path(initial_checkpoint) if initial_checkpoint else None
        )

        # Tracking state
        self.last_checkpoint_step = None
        self.checkpoint_steps = None
        self.decay_fraction = None
        self.decay_info = None

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, step: int, checkpoint_type: str) -> Path:
        return self.save_dir / f"checkpoint_{step}_{checkpoint_type}.pt"

    def _save_checkpoint(self, trainer, step: int, checkpoint_type: str):
        checkpoint_path = self._get_checkpoint_path(step, checkpoint_type)
        trainer.save_checkpoint(
            checkpoint_path,
            save_optimizer=self.save_optimizer,
            save_scheduler=self.save_scheduler,
            checkpoint_kwargs={
                "step": step,
                "checkpoint_type": checkpoint_type,
                "total_steps": trainer.run_config.max_num_train_steps,
                "decay_fraction": self.decay_fraction,
                "timestamp": datetime.now().isoformat(),
            },
        )
        trainer.print(f"\nSaved {checkpoint_type} checkpoint at step {step}")

    def on_training_run_start(self, trainer, **kwargs):
        """Initialize checkpoint tracking state and load initial checkpoint if specified."""
        if not hasattr(trainer.scheduler, "get_checkpoint_steps"):
            raise ValueError(
                "Scheduler must implement get_checkpoint_steps(). "
                "Are you using WSDLrScheduler?"
            )

        # Get checkpoint steps and decay info from scheduler
        self.checkpoint_steps = set(trainer.scheduler.get_checkpoint_steps())
        self.decay_fraction = trainer.scheduler.decay_phase_ratio
        self.decay_info = trainer.scheduler.get_decay_info()

        # Load initial checkpoint if specified
        if self.initial_checkpoint and self.initial_checkpoint.exists():
            trainer.print(f"\nLoading checkpoint from {self.initial_checkpoint}")
            checkpoint = trainer.load_checkpoint(self.initial_checkpoint)
            self.last_checkpoint_step = checkpoint.get("step")
            trainer.print(
                f"Loaded {checkpoint['checkpoint_type']} checkpoint from step {self.last_checkpoint_step}"
            )

    def on_train_step_end(self, trainer, step: int, **kwargs):
        """Handle checkpoint saving and progress logging"""

        # Calculate global step accounting for distributed training and gradient accumulation
        total_steps = (
            (trainer.run_history.current_epoch - 1)
            * trainer.run_config.num_update_steps_per_epoch
            + step // trainer.run_config.gradient_accumulation_steps
        )

        # Skip if we've already saved at this step
        if total_steps == self.last_checkpoint_step:
            return

        # Get current phase info from scheduler
        phase_info = trainer.scheduler.get_phase_info(total_steps)
        pre_decay_step = phase_info["pre_decay_step"]
        period_end = phase_info["period_end"]

        # Save pre-decay checkpoint when entering decay phase
        if total_steps == pre_decay_step:
            trainer.print(
                f"\nWSD Lr Scheduler entering decay phase at step {total_steps}"
            )
            self._save_checkpoint(trainer, total_steps, "wsd_pre_decay")
            self.last_checkpoint_step = total_steps

        # If we've completed the decay phase
        elif total_steps == period_end:
            self._save_checkpoint(trainer, total_steps, "wsd_post_decay")
            self.last_checkpoint_step = total_steps

    def on_training_run_end(self, trainer, **kwargs):
        """Save final checkpoint if we haven't already"""
        # Get the final step number
        total_steps = trainer.run_config.max_num_train_steps

        # If we haven't saved the final checkpoint yet
        if self.last_checkpoint_step != total_steps:
            # Get final phase info
            phase_info = trainer.scheduler.get_phase_info(total_steps)
            period_end = phase_info["period_end"]

            # Verify this is actually the end of a period
            if total_steps == period_end:
                self._save_checkpoint(trainer, total_steps, "wsd_post_decay")
                self.last_checkpoint_step = total_steps
