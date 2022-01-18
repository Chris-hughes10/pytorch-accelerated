# Copyright © 2021 Chris Hughes
import math
import os
from collections import Callable
from enum import Enum
from functools import partial
from typing import Iterable

import torch
from accelerate import Accelerator, DistributedType
from torch.utils.data import DataLoader

from pytorch_accelerated.callbacks import (
    CallbackHandler,
    LogMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    StopTrainingError,
    ProgressBarCallback,
    MoveModulesToDeviceCallback,
)
from pytorch_accelerated.run_config import TrainerRunConfig
from pytorch_accelerated.tracking import RunHistory, InMemoryRunHistory, LossTracker

DEFAULT_CALLBACKS = (
    MoveModulesToDeviceCallback,
    TerminateOnNaNCallback,
    PrintProgressCallback,
    ProgressBarCallback,
    LogMetricsCallback,
)


class TrainerPlaceholderValues(Enum):
    """
    Some learning rate schedulers require information such as the total number of steps that will take place during a training run.
    As this information is not accessible prior to creating the training dataloader - which will be done as part of the
    :meth:`~Trainer.train` method - a placeholder value can be used in the cases, as demonstrated below::

        from functools import Partial

        from pytorch_accelerated import TrainerPlaceholderValues
        from torch.optim.lr_scheduler import OneCycleLR

        create_scheduler_fn = partial(
                    OneCycleLR,
                    max_lr=config.lr,
                    epochs=TrainerPlaceholderValues.NUM_EPOCHS,
                    steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
                )

    These placeholders will be replaced by the trainer with the correct values during training.
    """

    NUM_EPOCHS = "trainer.run_config.num_epochs"
    NUM_UPDATE_STEPS_PER_EPOCH = "trainer.run_config.num_update_steps_per_epoch"
    TRAIN_DATALOADER_LEN = "len(trainer._train_dataloader)"
    EVAL_DATALOADER_LEN = "len(trainer._eval_dataloader)"

    @classmethod
    def placeholder_set(cls):
        return {placeholder for placeholder in cls}


def replace_trainer_placeholder_values(trainer, instance):
    """If the instance is partial and contains keywords, will replace these, returning a new function."""

    if isinstance(instance, partial):
        placeholders = TrainerPlaceholderValues.placeholder_set()
        keywords = list(instance.keywords.items())

        new_keywords = {}

        for keyword, value in keywords:
            if value in placeholders:
                new_keywords[keyword] = eval(value.value)
            else:
                new_keywords[keyword] = value

        instance = partial(instance.func, *instance.args, **new_keywords)

    return instance


class Trainer:
    """
    The Trainer is designed to encapsulate an entire training loop for a specific task, bringing together the model,
    loss function and optimizer, and providing a specification of the behaviour to execute for each step of the training
    process.

    The trainer has been implemented such that it provides (overridable) implementations of the parts of training
    that rarely change after they have been defined – such as creating a data loader, or how a batch of data is fed to
    the model – whilst remaining decoupled from components that are likely to change, such as the model, dataset,
    loss function and optimizer.
    """

    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        callbacks=DEFAULT_CALLBACKS,
        run_history=None,
    ):
        """
        Create a new trainer object which can be used to train the given model using the provided loss function and optimizer.

        :param model: a subclass of nn.Module to be trained
        :param loss_func: the loss function to use when training the model
        :param optimizer: the optimizer to update the model's parameters
        :param callbacks: a list of callbacks to use during training runs. If a list of callbacks is not provided, the default selection will be used.
        :param run_history: an instance of a RunHistory subclass to track training runs. If this is not provided, a new one will be created.

        The callbacks that are used by default are (
        :class:`~pytorch_accelerated.callbacks.MoveModulesToDeviceCallback`,
        :class:`~pytorch_accelerated.callbacks.TerminateOnNaNCallback`,
        :class:`~pytorch_accelerated.callbacks.PrintProgressCallback`,
        :class:`~pytorch_accelerated.callbacks.ProgressBarCallback`,
        :class:`~pytorch_accelerated.callbacks.LogMetricsCallback`,
        )

        """
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.callback_handler = CallbackHandler(
            callbacks,
        )
        self.run_history: RunHistory = (
            run_history if run_history is not None else InMemoryRunHistory()
        )
        self._accelerator = self._create_accelerator()
        self._loss_tracker = LossTracker()
        # placeholders which will be set during training
        self.create_scheduler_fn = None
        self.scheduler = None
        self.collate_fn = None
        self.train_dataset = None
        self.eval_dataset = None
        self._train_dataloader = None
        self._train_dl_kwargs = None
        self._eval_dl_kwargs = None
        self._eval_dataloader = None
        self.run_config: TrainerRunConfig = None

        self.callback_handler.call_event("on_init_end", self)

    def _create_accelerator(self):
        """
        Create an instance of :class:`accelerate.Accelerator` which will be used to manage training.
        :return:
        """
        return Accelerator()

    def create_train_dataloader(
        self, batch_size: int, train_dl_kwargs: dict = None
    ) -> Iterable:
        """
        Create a dataloader to be used during training. This is initialised with the train_dataset and collate function which have been passed to the Trainer.

        If no arguments are passed, the arguments returned by :meth:`Trainer.get_default_train_dl_kwargs` are used.

        .. Note:: if batch size is included in train_dl_kwargs, this takes precedence over the batch_size argument.

        :param batch_size: the batch size to use per device
        :param train_dl_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see :class:`torch.utils.data.DataLoader`

        :return: an instance of :class:`~torch.utils.data.DataLoader`
        """

        default_train_dl_kwargs = self.get_default_train_dl_kwargs(batch_size)

        if train_dl_kwargs is not None:
            default_train_dl_kwargs.update(train_dl_kwargs)

        self._train_dl_kwargs = default_train_dl_kwargs

        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **self._train_dl_kwargs
        )

    def create_eval_dataloader(
        self, batch_size: int, eval_dl_kwargs: dict = None
    ) -> Iterable:
        """
        Create a dataloader to be used during evaluation. This is initialised with the eval_dataset and collate function which have been passed to the Trainer.

        If no arguments are passed, the arguments returned by :meth:`Trainer.get_default_eval_dl_kwargs` are used.

        .. Note:: if batch size is included in eval_dl_kwargs, this takes precedence over the batch_size argument.

        :param batch_size: the batch size to use per device
        :param eval_dl_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see :class:`torch.utils.data.DataLoader`

        :return: an instance of :class:`torch.utils.data.DataLoader`
        """
        default_eval_dl_kwargs = self.get_default_eval_dl_kwargs(batch_size)

        if eval_dl_kwargs is not None:
            default_eval_dl_kwargs.update(eval_dl_kwargs)

        self._eval_dl_kwargs = default_eval_dl_kwargs

        return DataLoader(
            dataset=self.eval_dataset,
            collate_fn=self.collate_fn,
            **self._eval_dl_kwargs
        )

    def create_scheduler(self):
        """
        Create a learning rate scheduler based on the ``create_scheduler_fn`` function which has been passed to the Trainer.
        :return: a learning rate scheduler instance
        """
        scheduler_type = replace_trainer_placeholder_values(
            self, self.create_scheduler_fn
        )
        return scheduler_type(self.optimizer)

    def training_run_start(self):
        """
        This method is called at the start of a training run.
        """
        pass

    def train_epoch_start(self):
        """
        This method is called at the start of a training epoch.

        The default behaviour of this method is to call ``self.model.train()``

        """
        self.model.train()

    def calculate_train_batch_loss(self, batch) -> dict:
        """
        Calculates the training loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the :meth:`~callbacks.TrainerCallback.on_train_step_end` callback method.

        :param batch: the output of the train dataloader
        :return: A dictionary containing the training loss, model outputs and batch size. Can include any keys, but must include the keys 'loss', 'model_outputs' and 'batch_size'
        """
        xb, yb = batch

        model_outputs = self.model(xb)
        loss = self.loss_func(model_outputs, yb)

        return {
            "loss": loss,
            "model_outputs": model_outputs,
            "batch_size": yb.size(0),
        }

    def backward_step(self, loss):
        """
        Use the accelerator to perform the backward pass on the calculated value of the loss returned by :meth:`~Trainer.calculate_train_batch_loss`.
        If gradient accumulation is enabled, this loss has been scaled by 1 / accumulation steps.

        :param loss: The loss tensor returned by :meth:`~Trainer.calculate_train_batch_loss`.
        """
        self._accelerator.backward(loss)

    def optimizer_step(self):
        """
        Performs a single optimization step and updates the parameters which have been passed to ``self.optimizer``.
        """
        self.optimizer.step()

    def scheduler_step(self):
        """
        Performs a single scheduler step if ``self.scheduler`` has been assigned.

        """
        if self.scheduler is not None:
            self.scheduler.step()

    def optimizer_zero_grad(self):
        """
        Sets the gradients of all optimized ``torch.Tensor`` s to zero.
        """
        self.optimizer.zero_grad()

    def train_epoch_end(self):
        """
        This method is called at the end of each training epoch.
        """
        pass

    def eval_epoch_start(self):
        """
        This method is called at the start of an evaluation epoch.

        The default behaviour of this method is to call ``self.model.eval()``
        """
        self.model.eval()

    def calculate_eval_batch_loss(self, batch) -> dict:
        """
        Calculates the evaluation loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the :meth:`~callbacks.TrainerCallback.on_eval_step_end` callback.

        :param batch: the output of the eval dataloader
        :return: A dictionary containing the evaluation loss, model outputs and batch size. Can include any keys, but must include the keys ``loss``, ``model_outputs`` and ``batch_size``
        """
        with torch.no_grad():
            xb, yb = batch
            model_outputs = self.model(xb)
            val_loss = self.loss_func(model_outputs, yb)

        return {
            "loss": val_loss,
            "model_outputs": model_outputs,
            "batch_size": yb.size(0),
        }

    def eval_epoch_end(self):
        """
        This method is called at the end of an evaluation epoch.
        """
        pass

    def training_run_epoch_end(self):
        """
        This method is called during a training run after both training and evaluation epochs have been completed.
        """
        pass

    def training_run_end(self):
        """
        This method is called at the end of a training run.
        """
        pass

    def evaluation_run_start(self):
        """
        This method is called at the start of an evaluation run.
        """
        pass

    def evaluation_run_end(self):
        """
        This method is called at the end of an evaluation run.
        """
        pass

    def train(
        self,
        train_dataset,
        num_epochs,
        eval_dataset=None,
        per_device_batch_size=8,
        max_num_train_steps=None,
        gradient_accumulation_steps=1,
        gradient_clip_value=None,
        create_scheduler_fn: Callable = None,
        train_dataloader_kwargs: dict = None,
        eval_dataloader_kwargs: dict = None,
        reset_run_history=True,
        collate_fn=None,
    ):
        """
        Start a training run. If an evaluation dataset is provided, this routine will include both training and evaluation epochs.

        .. Note::
            As the optimizer needs to be internally prepared prior to training, in order to use a learning rate scheduler,
            a factory function must be provided to ``create_scheduler_fn``. This must be a function which accepts the optimizer as a single parameter
            and returns an instance of a learning rate scheduler. Passing an instance of a learning rate scheduler will not work here.

        :param train_dataset: the dataset to use during training epochs
        :param num_epochs: the number of epochs to train for
        :param eval_dataset: the dataset to use during evaluation epochs, if this is not provided, evaluation is skipped.
        :param per_device_batch_size: the batch size to use per device
        :param max_num_train_steps: the maximum number of steps to train for. If provided, this will override num_epochs
        :param gradient_accumulation_steps: accumulate gradients to the specified number of steps to simulate a bigger batch size. By default, this is set to ``1``
        :param gradient_clip_value: if specified, the gradients of the model's parameters will be clipped to the range ``[-gradient_clip_value, gradient_clip_value]``
        :param create_scheduler_fn: a function which accepts an optimizer as an argument and returns a learning rate scheduler
        :param train_dataloader_kwargs: : a dictionary of keyword arguments to pass to the training dataloader constructor, for details see :class:`torch.utils.data.DataLoader`
        :param eval_dataloader_kwargs: a dictionary of keyword arguments to pass to the evaluation dataloader constructor, for details see :class:`torch.utils.data.DataLoader`
        :param reset_run_history: reset any run history saved by the trainer from previous training runs
        :param collate_fn: the collate function to be used by the training and evaluation dataloaders
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.create_scheduler_fn = create_scheduler_fn
        self.collate_fn = collate_fn

        if reset_run_history:
            self.run_history.reset()

        self._train_dataloader = self.create_train_dataloader(
            batch_size=per_device_batch_size, train_dl_kwargs=train_dataloader_kwargs
        )

        if self.eval_dataset is not None:
            self._eval_dataloader = self.create_eval_dataloader(
                batch_size=per_device_batch_size, eval_dl_kwargs=eval_dataloader_kwargs
            )

        self._prepare_model_optimizer_and_dataloaders()

        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
            per_device_batch_size=per_device_batch_size,
            gradient_clip_value=gradient_clip_value,
        )

        if self.create_scheduler_fn is not None:
            self.scheduler = self.create_scheduler()

        self._run_training()

    def evaluate(
        self,
        dataset=None,
        per_device_batch_size=8,
        dataloader_kwargs: dict = None,
        collate_fn=None,
    ):
        """
        Start an evaluation run.

        .. Note:: Starting an evaluation run will reset the :class:`Trainer`'s run history.

        .. Note:: During distributed evaluation, if the `per_device_batch_size` * the number of processes used does not exactly divide the dataset, and `drop_last=False` has not been passed as a dataloader kwarg, the dataloader will repeat from the start in processes that run out of batches. This should be taken into consideration when calculating metrics.

        :param dataset: the dataset to use during evaluation
        :param per_device_batch_size: the batch size to use per device
        :param dataloader_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see :class:`torch.utils.data.DataLoader`
        :param collate_fn: the collate function to be used by the dataloader
        """
        self.eval_dataset = dataset
        self.collate_fn = collate_fn

        self.run_history.reset()

        self._train_dataloader = None
        self._eval_dataloader = self.create_eval_dataloader(
            batch_size=per_device_batch_size, eval_dl_kwargs=dataloader_kwargs
        )

        self._prepare_model_optimizer_and_dataloaders()

        if self.run_config is None:
            self.run_config = self._create_run_config(
                num_epochs=1,
                gradient_accumulation_steps=0,
                max_num_train_steps=None,
                per_device_batch_size=per_device_batch_size,
                gradient_clip_value=None,
            )

        self._run_evaluation()

    def get_default_train_dl_kwargs(self, batch_size) -> dict:
        """
        Return the default arguments that will be used by the training dataloader.

        :param batch_size: the batch size to use during training
        :return: a dictionary containing the default arguments for the training dataloader
        """
        return {
            "shuffle": True,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(
                os.cpu_count() // torch.cuda.device_count()
                if torch.cuda.is_available()
                else os.cpu_count(),
                1,
            ),
        }

    def get_default_eval_dl_kwargs(self, batch_size) -> dict:
        """
        Return the default arguments that will be used by the evaluation dataloader.

        :param batch_size: the batch size to use during evaluation
        :return: a dictionary containing the default arguments for the evaluation dataloader
        """
        return {
            "shuffle": False,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(
                os.cpu_count() // torch.cuda.device_count()
                if torch.cuda.is_available()
                else os.cpu_count(),
                1,
            ),
        }

    @property
    def device(self):
        """
        Use the internal instance of :class:`accelerate.Accelerator` to get the appropriate device
        :return: an instance of `torch.device`
        """
        return self._accelerator.device

    def _prepare_model_optimizer_and_dataloaders(self):
        """
        Uses the trainer's instance of :class:`accelerate.Accelerator` to wrap the model, optimizer and dataloaders in any wrappers necessary for training.
        (e.g. :class:`torch.nn.parallel.DistributedDataParallel`) and ensures the parameters are placed on the appropriate device.

        By default, this will convert each dataloader to an instance of :class:`accelerate.data_loader.DataLoaderShard`. Depending on the value of the `drop_last` attribute of the dataloaders,
        either iterations will stop at the first batch that would be too small / not present on all processes or loop with batches from the beginning on processes which run out of data, so that all batch sizes are the same size.

        .. Note:: This may change the length of the dataloaders, so this should be called *before* the number of update steps per epoch is calculated, i.e. to initialise a learning rate scheduler
        """
        self._accelerator.free_memory()
        self._accelerator = self._create_accelerator()

        components = [self.model, self.optimizer]

        if self._train_dataloader is not None:
            components.append(self._train_dataloader)

        if self._eval_dataloader is not None:
            components.append(self._eval_dataloader)

        prepared_components = self._accelerator.prepare(*components)

        self.model = prepared_components[0]
        self.optimizer = prepared_components[1]

        if self._train_dataloader is not None:
            self._train_dataloader = prepared_components[2]
            if self._eval_dataloader is not None:
                self._eval_dataloader = prepared_components[3]

        elif self._eval_dataloader is not None:
            self._eval_dataloader = prepared_components[2]

    def _create_run_config(
        self,
        per_device_batch_size,
        num_epochs,
        gradient_accumulation_steps,
        max_num_train_steps,
        gradient_clip_value,
    ) -> TrainerRunConfig:
        """
        Create an instance of :class:`~pytorch_accelerated.run_config.TrainerRunConfig` representing the current state of the trainer.

        :param per_device_batch_size: the batch size per device
        :param num_epochs: the number of epochs in the current training run
        :param gradient_accumulation_steps: the number of gradient accumulation steps which will be used during the training run
        :param max_num_train_steps: If specified, the maximum number of steps to train for. If present, this will take precedence over ``num_epochs``
        :param gradient_clip_value: the value used to determine the threshold to clip the gradients of the model's parameters

        """
        if self._train_dl_kwargs is not None:
            train_per_device_batch_size = self._train_dl_kwargs.get(
                "batch_size", per_device_batch_size
            )
        else:
            train_per_device_batch_size = per_device_batch_size

        if self._eval_dl_kwargs is not None:
            eval_per_device_batch_size = self._eval_dl_kwargs.get(
                "batch_size", train_per_device_batch_size
            )
        else:
            eval_per_device_batch_size = train_per_device_batch_size

        if self._train_dataloader is not None:
            num_update_steps_per_epoch = math.ceil(
                len(self._train_dataloader) / gradient_accumulation_steps
            )
        else:
            num_update_steps_per_epoch = 0

        if max_num_train_steps is None:
            max_num_train_steps = num_epochs * num_update_steps_per_epoch
        else:
            num_epochs = math.ceil(max_num_train_steps / num_update_steps_per_epoch)

        config = {
            "num_epochs": num_epochs,
            "train_per_device_batch_size": train_per_device_batch_size,
            "train_dl_kwargs": self._train_dl_kwargs,
            "eval_per_device_batch_size": eval_per_device_batch_size,
            "eval_dl_kwargs": self._eval_dl_kwargs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "train_total_batch_size": train_per_device_batch_size
            * self._accelerator.num_processes
            * gradient_accumulation_steps,
            "eval_total_batch_size": eval_per_device_batch_size
            * self._accelerator.num_processes,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
            "max_num_train_steps": max_num_train_steps,
            "is_local_process_zero": self._accelerator.is_local_main_process,
            "is_world_process_zero": self._accelerator.is_main_process,
            "is_distributed": True
            if self._accelerator.distributed_type != DistributedType.NO
            else False,
            "using_fp16": self._accelerator.use_fp16,
            "gradient_clip_value": gradient_clip_value,
        }

        return TrainerRunConfig(**config)

    def _run_training(self):
        """
        The method responsible for the orchestration of the high level steps which will be executed during a training run.
        """
        self.training_run_start()
        self.callback_handler.call_event(
            "on_training_run_start",
            self,
        )
        for epoch in range(self.run_config.num_epochs):
            try:
                self._run_train_epoch(self._train_dataloader)
                if self.eval_dataset is not None:
                    self._run_eval_epoch(self._eval_dataloader)
                self.run_history._increment_epoch()
                self.training_run_epoch_end()
                self.callback_handler.call_event(
                    "on_training_run_epoch_end",
                    self,
                )
            except StopTrainingError as e:
                self._accelerator.print(e)
                self.callback_handler.call_event(
                    "on_stop_training_error",
                    self,
                )
                break
        self.training_run_end()
        self.callback_handler.call_event(
            "on_training_run_end",
            self,
        )

    def _run_evaluation(self):
        """
        The method responsible for the orchestration of the high level steps which will be executed during an evaluation run.
        """
        self.evaluation_run_start()
        self.callback_handler.call_event(
            "on_evaluation_run_start",
            self,
        )
        try:
            self._run_eval_epoch(self._eval_dataloader, is_training=False)
        except StopTrainingError as e:
            self._accelerator.print(e)
            self.callback_handler.call_event(
                "on_stop_training_error",
                self,
            )
        self.evaluation_run_end()
        self.callback_handler.call_event(
            "on_evaluation_run_end",
            self,
        )

    def _run_train_epoch(self, train_dl):
        """
        The method responsible for the behaviour of each training epoch.

        :param train_dl: the dataloader to be used during training
        """
        self.train_epoch_start()
        self._loss_tracker.reset()
        self.callback_handler.call_event(
            "on_train_epoch_start",
            self,
        )

        for step, batch in enumerate(train_dl):
            self.callback_handler.call_event(
                "on_train_step_start",
                self,
            )
            batch_output = self.calculate_train_batch_loss(batch)
            self._loss_tracker.update(
                self.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            if self.run_config.gradient_accumulation_steps > 1:
                batch_output["loss"] /= self.run_config.gradient_accumulation_steps

            self.callback_handler.call_event(
                "on_train_step_end", self, batch_output=batch_output, batch=batch
            )
            self.backward_step(batch_output["loss"])

            if self.run_config.gradient_clip_value is not None:
                self._clip_gradients()

            if (step + 1) % self.run_config.gradient_accumulation_steps == 0 or (
                step + 1 == len(train_dl)
            ):
                self.optimizer_step()
                if (
                    self.scheduler is not None
                    and not self._accelerator.optimizer_step_was_skipped
                ):
                    self.scheduler_step()
                self.optimizer_zero_grad()

        self.train_epoch_end()
        self.run_history.update_metric("train_loss_epoch", self._loss_tracker.average)
        self.callback_handler.call_event(
            "on_train_epoch_end",
            self,
        )

    def _clip_gradients(self):
        """
        Clip the gradients of the model's parameters that fall outside of the threshold specified in :meth:`~Trainer.train`.

        By default, this clips the gradients using :meth:`accelerate.Accelerator.clip_grad_value_`
        """
        self._accelerator.clip_grad_value_(
            self.model.parameters(), clip_value=self.run_config.gradient_clip_value
        )

    def _run_eval_epoch(self, valid_dl, is_training: bool = True):
        """
        The method responsible for the behaviour of each evaluation epoch.

        :param valid_dl: the dataloader to be used during evaluation
        :param is_training: signals whether the evaluation is being run as part of a training run
        """
        self.eval_epoch_start()
        self._loss_tracker.reset()
        self.callback_handler.call_event(
            "on_eval_epoch_start",
            self,
        )

        for batch in valid_dl:
            self.callback_handler.call_event(
                "on_eval_step_start",
                self,
            )
            batch_output = self.calculate_eval_batch_loss(batch)
            self._loss_tracker.update(
                self.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            self.callback_handler.call_event(
                "on_eval_step_end", self, batch_output=batch_output, batch=batch
            )
        self.eval_epoch_end()
        metric_name = "eval_loss_epoch" if is_training else "evaluation_loss"
        self.run_history.update_metric(metric_name, self._loss_tracker.average)
        self.callback_handler.call_event(
            "on_eval_epoch_end",
            self,
        )

    def gather(self, tensor):
        """
        Gather the values in `tensor` across all processes and concatenate them on the first dimension. This can be
        useful to regroup the predictions from all processes when doing evaluation.

        .. Note:: This gather happens in all processes.

        :param tensor: (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`) The tensors to gather across all processes.
        :return: The gathered tensor(s) (:obj:`torch.Tensor`, or a nested tuple/list/dictionary of :obj:`torch.Tensor`). The first dimension of the result is `num_processes` multiplied by the first dimension of the input tensors.
        """
        return self._accelerator.gather(tensor)

    def print(self, *args, **kwargs):
        """
        Use in replacement of ``print()`` to only print once per node.
        """
        if self._accelerator is not None:
            self._accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def save_checkpoint(
        self, save_path, checkpoint_kwargs=None, save_optimizer=True, save_per_node=True
    ):
        """
        Save the model, optimizer and specified args as a checkpoint file.

        :param save_path: the path where to save the checkpoint, this should end in '.pt'
        :param checkpoint_kwargs: additional objects to include in the checkpoint
        :param save_optimizer: flag to indicate whether to include the optimizer in the checkpoint
        :param save_per_node: flag to indicate whether to save the checkpoint once per machine, if False, the checkpoint will only be saved from the world process zero. This is True by default.
        """
        # TODO: add save method for run history?

        checkpoint = {
            "model_state_dict": self._accelerator.unwrap_model(self.model).state_dict(),
        }

        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if checkpoint_kwargs is not None:
            checkpoint.update(checkpoint_kwargs)

        self._accelerator.wait_for_everyone()

        if save_per_node:

            self._accelerator.save(
                checkpoint,
                save_path,
            )
        else:

            if self.run_config.is_world_process_zero:
                self._accelerator.save(
                    checkpoint,
                    save_path,
                )

    def load_checkpoint(self, checkpoint_path, load_optimizer=True):
        """
        Load the model and optimizer from a checkpoint file.

        :param checkpoint_path: the path of the checkpoint file to load
        :param load_optimizer: flag to indicate whether to load the optimizer if it is included in the checkpoint
        """
        self._accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            if self.optimizer is None:
                raise ValueError(
                    "You are trying to load an optimizer from a checkpoint, but no optimizer"
                    "has been set in the Trainer. Either pass the correct optimizer instance when"
                    "creating the trainer, or specify load_optimizer=False when loading the checkpoint."
                )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
