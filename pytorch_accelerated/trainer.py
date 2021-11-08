import math
import os
from enum import Enum
from functools import partial
from typing import Iterable

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from pytorch_accelerated.callbacks import (
    CallbackHandler,
    PrintMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    StopTrainingError,
    ProgressBarCallback,
)
from pytorch_accelerated.tracking import RunHistory, InMemoryRunHistory, LossTracker

DEFAULT_CALLBACKS = (
    TerminateOnNaNCallback,
    PrintProgressCallback,
    ProgressBarCallback,
    PrintMetricsCallback,
)


class TrainerPlaceholderValues(Enum):
    NUM_EPOCHS = 'trainer.run_config["num_epochs"]'
    NUM_UPDATE_STEPS_PER_EPOCH = 'trainer.run_config["num_update_steps_per_epoch"]'
    TRAIN_DATALOADER_LEN = "len(trainer._train_dataloader)"
    EVAL_DATALOADER_LEN = "len(trainer._eval_dataloader)"

    @classmethod
    def placeholder_set(cls):
        return {placeholder for placeholder in cls}


def replace_trainer_placeholder_values(trainer, instance):
    "If the instance is partial and contains keywords, will replace these, returning a new function"

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
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        callbacks=DEFAULT_CALLBACKS,
        run_history=None,
    ):
        """

        :param model:
        :param loss_func:
        :param optimizer:
        :param callbacks:
        :param run_history:
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
        self._accelerator = Accelerator()
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
        self.run_config = None

        self.callback_handler.call_event("on_init_end", self)

    def create_train_dataloader(
        self, batch_size: int, train_dl_kwargs: dict
    ) -> Iterable:
        """
        Create a dataloader to be used during training. This is initialised with the train_dataset and collate function which have been passed to the Trainer.

        If no arguments are passed, the default arguments are:
        {
           \n"shuffle": True,
            \n"pin_memory": True if torch.cuda.is_available() else False,
            \n"batch_size": batch_size,
            \n"num_workers": max(os.cpu_count() // torch.cuda.device_count(), 1),
        }

        Note that if batch size is included in train_dl_kwargs, this takes precedence over the batch_size argument.

        :param batch_size: the batch size to use per device
        :param train_dl_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see torch.utils.data.DataLoader

        :return: an instance of torch.utils.data.DataLoader
        """
        default_train_dl_kwargs = {
            "shuffle": True,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(os.cpu_count() // torch.cuda.device_count(), 1),
        }

        if train_dl_kwargs is not None:
            default_train_dl_kwargs.update(train_dl_kwargs)

        self._train_dl_kwargs = default_train_dl_kwargs

        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collate_fn,
            **self._train_dl_kwargs
        )

    def create_eval_dataloader(self, batch_size: int, eval_dl_kwargs: dict) -> Iterable:
        """
        Create a dataloader to be used during evaluation. This is initialised with the eval_dataset and collate function which have been passed to the Trainer.

        If no arguments are passed, the default arguments are:
        {
           \n"shuffle": False,
            \n"pin_memory": True if torch.cuda.is_available() else False,
            \n"batch_size": batch_size,
            \n"num_workers": max(os.cpu_count() // torch.cuda.device_count(), 1),
        }

        Note that if batch size is included in eval_dl_kwargs, this takes precedence over the batch_size argument.

        :param batch_size: the batch size to use per device
        :param eval_dl_kwargs: a dictionary of keyword arguments to pass to the dataloader constructor, for details see torch.utils.data.DataLoader

        :return: an instance of torch.utils.data.DataLoader
        """
        default_eval_dl_kwargs = {
            "shuffle": False,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": batch_size,
            "num_workers": max(os.cpu_count() // torch.cuda.device_count(), 1),
        }

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
        Create a learning rate scheduler based on the create_scheduler_fn function which has been passed to the Trainer.
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

        The default behaviour of this method is to call `self.model.train()`

        """
        self.model.train()

    def calculate_train_batch_loss(self, batch) -> dict:
        """
        Calculates the training loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the `on_train_step_end` callback.

        :param batch: the output of the train dataloader
        :return: A dictionary containing the training loss, model outputs and batch size. Can include any keys, but must include the keys 'loss', 'model_outputs' and 'batch_size'
        """
        xb, yb = batch

        model_outputs = self.model(xb)
        loss = self.loss_func(model_outputs, yb)

        return {
            "loss": loss,
            "model_outputs": model_outputs,
            "batch_size": xb.size(0),
        }

    def backward_step(self, loss):
        """
        Use the accelerator to perform the backward pass on the calculated value of the loss returned by `calculate_train_batch_loss`.
        If gradient accumulation is enabled, this loss has been scaled by 1 / accumulation steps.

        :param loss: The loss tensor returned by calculate_train_batch_loss.
        """
        self._accelerator.backward(loss)

    def optimizer_step(self):
        """
        Performs a single optimization step and updates the parameters which have been passed to self.optimizer.
        """
        self.optimizer.step()

    def scheduler_step(self):
        """
        Performs a single scheduler step if self.scheduler has been assigned.

        """
        if self.scheduler is not None:
            self.scheduler.step()

    def optimizer_zero_grad(self):
        """
        Sets the gradients of all optimized torch.Tensor s to zero.
        """
        self.optimizer.zero_grad()

    def train_epoch_end(self):
        """
        This method is called at the end of each training epoch
        """
        pass

    def eval_epoch_start(self):
        """
        This method is called at the start of an evaluation epoch.

        The default behaviour of this method is to call `self.model.eval()`
        """
        self.model.eval()

    def calculate_eval_batch_loss(self, batch) -> dict:
        """
        Calculates the evaluation loss and return this along with the batch size and model outputs.
        Any additional values returned will be available in the `on_eval_step_end` callback.

        :param batch: the output of the eval dataloader
        :return: A dictionary containing the evaluation loss, model outputs and batch size. Can include any keys, but must include the keys 'loss', 'model_outputs' and 'batch_size'
        """
        with torch.no_grad():
            xb, yb = batch
            model_outputs = self.model(xb)
            val_loss = self.loss_func(model_outputs, yb)

        return {
            "loss": val_loss,
            "model_outputs": model_outputs,
            "batch_size": xb.size(0),
        }

    def eval_epoch_end(self):
        """
        This method is called at the end of an evaluation epoch.
        """
        pass

    def training_run_end(self):
        """
        This method is called at the end of a training run.
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
        create_scheduler_fn=None,
        train_dataloader_kwargs: dict = None,
        eval_dataloader_kwargs: dict = None,
        reset_run_history=True,
        collate_fn=None,
    ):
        """
        Start a training run. If an evaluation dataset is provided, this routine will include both training and evaluation epochs.

        TODO: As the internal optimizer needs to be prepared prior to training, in order to use a learning rate scheduler,

        :param train_dataset: the dataset to use during training epochs
        :param num_epochs: the number of epochs
        :param eval_dataset: the dataset to use during evaluation epochs
        :param per_device_batch_size: the batch size to use per device
        :param max_num_train_steps: the maximum number of steps to train for. If provided, this will override num_epochs
        :param gradient_accumulation_steps: accumulate grads to the specified number of steps to simulate a bigger batch size. By default, this is set to 1
        :param create_scheduler_fn: a function which accepts an optimizer as an argument and returns a learning rate scheduler
        :param train_dataloader_kwargs: : a dictionary of keyword arguments to pass to the  train dataloader constructor, for details see torch.utils.data.DataLoader
        :param eval_dataloader_kwargs: a dictionary of keyword arguments to pass to the eval dataloader constructor, for details see torch.utils.data.DataLoader
        :param reset_run_history: reset any run history saved by the trainer from previous training runs
        :param collate_fn: the collate function to be used by the train and eval dataloaders
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.create_scheduler_fn = create_scheduler_fn
        self.collate_fn = collate_fn

        if reset_run_history:
            self.run_history.reset()

        self._prepare_model_and_optimizer()

        self._train_dataloader = self.create_train_dataloader(
            batch_size=per_device_batch_size, train_dl_kwargs=train_dataloader_kwargs
        )

        if self.eval_dataset is not None:
            self._eval_dataloader = self.create_eval_dataloader(
                batch_size=per_device_batch_size, eval_dl_kwargs=eval_dataloader_kwargs
            )

        self._prepare_dataloaders()

        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
            per_device_batch_size=per_device_batch_size,
        )

        if self.create_scheduler_fn is not None:
            self.scheduler = self.create_scheduler()

        self.callback_handler.call_event(
            "on_train_run_begin",
            self,
        )
        self._run_training()
        self.callback_handler.call_event(
            "on_train_run_end",
            self,
        )

    def _prepare_model_and_optimizer(self):
        self._accelerator.free_memory()
        (self.model, self.optimizer,) = self._accelerator.prepare(
            self.model,
            self.optimizer,
        )

    def _prepare_dataloaders(self):
        self._train_dataloader = self._accelerator.prepare(self._train_dataloader)
        if self._eval_dataloader is not None:
            self._eval_dataloader = self._accelerator.prepare(self._eval_dataloader)

    def _create_run_config(
        self,
        per_device_batch_size,
        num_epochs,
        gradient_accumulation_steps,
        max_num_train_steps,
    ):

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

        num_update_steps_per_epoch = math.ceil(
            len(self._train_dataloader) / gradient_accumulation_steps
        )

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
            "fp16": self._accelerator.use_fp16,
        }

        return config

    def _run_training(self):
        self.training_run_start()
        for epoch in range(self.run_config["num_epochs"]):
            try:
                self._run_train_epoch(self._train_dataloader)
                if self.eval_dataset is not None:
                    self._run_eval_epoch(self._eval_dataloader)
                self.run_history.increment_epoch()
            except StopTrainingError as e:
                self._accelerator.print(e)
                self.callback_handler.call_event(
                    "on_stop_training_error",
                    self,
                )
                break
        self.training_run_end()

    def _run_train_epoch(self, train_dl):
        self.train_epoch_start()
        self._loss_tracker.reset()
        self.callback_handler.call_event(
            "on_train_epoch_begin",
            self,
        )

        for step, batch in enumerate(train_dl):
            self.callback_handler.call_event(
                "on_train_step_begin",
                self,
            )
            batch_output = self.calculate_train_batch_loss(batch)
            self._loss_tracker.update(
                self._accelerator.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            if self.run_config["gradient_accumulation_steps"] > 1:
                batch_output["loss"] /= self.run_config["gradient_accumulation_steps"]

            self.callback_handler.call_event(
                "on_train_step_end", self, batch_output=batch_output, batch=batch
            )
            self.backward_step(batch_output["loss"])

            if (step + 1) % self.run_config["gradient_accumulation_steps"] == 0 or (
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

    def _run_eval_epoch(self, valid_dl):
        self.eval_epoch_start()
        self._loss_tracker.reset()
        self.callback_handler.call_event(
            "on_eval_epoch_begin",
            self,
        )

        for batch in valid_dl:
            self.callback_handler.call_event(
                "on_eval_step_begin",
                self,
            )
            batch_output = self.calculate_eval_batch_loss(batch)
            self._loss_tracker.update(
                self._accelerator.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            self.callback_handler.call_event(
                "on_eval_step_end", self, batch_output=batch_output, batch=batch
            )
        self.eval_epoch_end()
        self.run_history.update_metric("eval_loss_epoch", self._loss_tracker.average)
        self.callback_handler.call_event(
            "on_eval_epoch_end",
            self,
        )

    def print(self, *args, **kwargs):
        """
        Use in replacement of print() to only print once per node.
        """
        if self._accelerator is not None:
            self._accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def save_model(self, save_dir, checkpoint_kwargs=None, save_optimizer=True):
        """

        :param save_dir:
        :param checkpoint_kwargs:
        :param save_optimizer:
        :return:
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

        self._accelerator.save(
            checkpoint,
            save_dir,
        )

    def load_checkpoint(self, checkpoint_dir):
        """

        :param checkpoint_dir:
        :return:
        """
        self._accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_dir, map_location="cpu")
        self._accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
