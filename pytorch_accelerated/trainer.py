import math
import os
from enum import Enum
from functools import partial

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from pytorch_accelerated.callbacks import (
    CallbackHandler,
    PrintMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    StopTrainingError,
)
from pytorch_accelerated.tracking import RunHistory, InMemoryRunHistory, LossTracker

DEFAULT_CALLBACKS = (
    TerminateOnNaNCallback,
    PrintMetricsCallback,
    PrintProgressCallback,
)

class TrainerPlaceholderValues(Enum):
    NUM_EPOCHS = 'trainer.run_config["num_epochs"]'
    NUM_UPDATE_STEPS_PER_EPOCH = 'trainer.run_config["num_update_steps_per_epoch"]'
    TRAIN_DATALOADER_LEN = 'len(trainer._train_dataloader)'
    EVAL_DATALOADER_LEN = 'len(trainer._eval_dataloader)'

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
        collate_fn=None,
        run_history=None,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler_type = None
        self.scheduler = None
        self.collate_fn = collate_fn
        self.train_dataset = None
        self.eval_dataset = None
        self._accelerator = Accelerator()
        self._train_dataloader = None
        self._train_dl_kwargs = None
        self._eval_dl_kwargs = None
        self._eval_dataloader = None
        self.run_config = None
        self._loss_tracker = LossTracker()
        self.run_history: RunHistory = (
            run_history if run_history is not None else InMemoryRunHistory()
        )

        self.callback_handler = CallbackHandler(
            callbacks,
        )
        self.callback_handler.call_event("on_init_end", self)

    def create_train_dataloader(self, per_device_batch_size, train_dl_kwargs):
        default_train_dl_kwargs = {
            "shuffle": True,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": per_device_batch_size,
            "num_workers": max(os.cpu_count()//torch.cuda.device_count(), 1)
        }

        if train_dl_kwargs is not None:
            default_train_dl_kwargs.update(train_dl_kwargs)

        self._train_dl_kwargs = default_train_dl_kwargs

        return DataLoader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **self._train_dl_kwargs
        )

    def create_eval_dataloader(self, per_device_batch_size, eval_dl_kwargs):
        default_eval_dl_kwargs = {
            "shuffle": False,
            "pin_memory": True if torch.cuda.is_available() else False,
            "batch_size": per_device_batch_size,
            "num_workers": max(os.cpu_count()//torch.cuda.device_count(), 1)
        }

        if eval_dl_kwargs is not None:
            default_eval_dl_kwargs.update(eval_dl_kwargs)

        self._eval_dl_kwargs = default_eval_dl_kwargs

        return DataLoader(
            dataset=self.eval_dataset, collate_fn=self.collate_fn, **self._eval_dl_kwargs
        )

    def create_scheduler(self, optimizer):
        scheduler_type = replace_trainer_placeholder_values(self, self.scheduler_type)
        return scheduler_type(optimizer)

    def training_run_start(self):
        pass

    def train_epoch_start(self):
        self.model.train()

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch

        model_outputs = self.model(xb)
        loss = self.loss_func(model_outputs, yb)

        return {
            "loss": loss,
            "model_outputs": self._accelerator.gather(model_outputs),
            "batch_size": xb.size(0),
        }

    def backward_step(self, loss):
        self._accelerator.backward(loss)

    def optimizer_step(self):
        self.optimizer.step()

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def train_epoch_end(
        self,
    ):
        pass

    def eval_epoch_start(self):
        self.model.eval()

    def calculate_eval_batch_loss(self, batch):
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
        pass

    def training_run_end(self):
        pass

    def train(
        self,
        train_dataset,
        num_epochs,
        eval_dataset=None,
        per_device_batch_size=8,
        max_num_train_steps=None,
        gradient_accumulation_steps=1,
        scheduler_type=None,
        train_dataloader_kwargs: dict = None,
        eval_dataloader_kwargs: dict = None,
        reset_run_history=True
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.scheduler_type = scheduler_type

        if reset_run_history:
            self.run_history.reset()

        self._prepare_model_and_optimizer()

        self._prepare_dataloaders(
            per_device_batch_size=per_device_batch_size,
            train_dl_kwargs=train_dataloader_kwargs,
            eval_dl_kwargs=eval_dataloader_kwargs,
        )

        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
        )

        if self.scheduler_type is not None:
            self.scheduler = self.create_scheduler(self.optimizer)


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

    def _prepare_dataloaders(
        self, per_device_batch_size, train_dl_kwargs=None, eval_dl_kwargs=None
    ):

        train_dataloader = self.create_train_dataloader(per_device_batch_size, train_dl_kwargs)
        self._train_dataloader = self._accelerator.prepare(train_dataloader)

        if self.eval_dataset is not None:
            eval_dataloader = self.create_eval_dataloader(per_device_batch_size, eval_dl_kwargs)
            self._eval_dataloader = self._accelerator.prepare(eval_dataloader)

    def _create_run_config(
        self,
        num_epochs,
        gradient_accumulation_steps,
        max_num_train_steps,
    ):

        train_per_device_batch_size = self._train_dl_kwargs['batch_size']

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
            # override num epochs
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
        # use SimpleNamespace or dotdict instead of dict?

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
        self.callback_handler.call_event(
            "on_train_epoch_begin",
            self,
        )
        self.train_epoch_start()
        self._loss_tracker.reset()

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
                if self.scheduler is not None and not self._accelerator.optimizer_step_was_skipped:
                    self.scheduler_step()
                self.optimizer_zero_grad()

        self.train_epoch_end()
        self.run_history.update_metric("train_loss_epoch", self._loss_tracker._average)
        self.callback_handler.call_event(
            "on_train_epoch_end",
            self,
        )

    def _run_eval_epoch(self, valid_dl):
        self.callback_handler.call_event(
            "on_eval_epoch_begin",
            self,
        )
        self.eval_epoch_start()
        self._loss_tracker.reset()

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
        self.run_history.update_metric("eval_loss_epoch", self._loss_tracker._average)
        self.callback_handler.call_event(
            "on_eval_epoch_end",
            self,
        )

    def print(self, *args, **kwargs):
        if self._accelerator is not None:
            self._accelerator.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def save_model(self, save_dir, checkpoint_kwargs=None, save_optimizer=True):
        # TODO: add save method for run history?

        checkpoint = {
                "model_state_dict": self._accelerator.unwrap_model(
                    self.model
                ).state_dict(),
            }

        if save_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if checkpoint_kwargs is not None:
            checkpoint.update(checkpoint_kwargs)

        self._accelerator.wait_for_everyone()

        self._accelerator.save(
            checkpoint,
            save_dir,)

    def load_checkpoint(self, checkpoint_dir):
        self._accelerator.wait_for_everyone()
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        self._accelerator.unwrap_model(
            self.model
        ).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        pass
