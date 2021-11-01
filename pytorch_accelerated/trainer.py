import math

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
from pytorch_accelerated.tracking import RunHistory, InMemoryRunHistory, AverageMeter

DEFAULT_CALLBACKS = (
    TerminateOnNaNCallback,
    PrintMetricsCallback,
    PrintProgressCallback,
)


class Trainer:
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        scheduler_type=None,
        callbacks=DEFAULT_CALLBACKS,
        collate_fn=None,
        run_history=None,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler = None
        self.collate_fn = collate_fn
        self.train_dataset = None
        self.eval_dataset = None
        self._accelerator = None
        self._train_dataloader = None
        self._eval_dataloader = None
        self.run_config = None
        self._loss_tracker = AverageMeter()
        self.run_history: RunHistory = (
            run_history if run_history is not None else InMemoryRunHistory()
        )

        self.callback_handler = CallbackHandler(
            callbacks,
        )
        self.callback_handler.call_event("on_init_end", self)

    def create_train_dataloader(self, **kwargs):
        return DataLoader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **kwargs
        )

    def create_eval_dataloader(self, **kwargs):
        return DataLoader(
            dataset=self.eval_dataset, collate_fn=self.collate_fn, **kwargs
        )

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

    def train_epoch_end(
        self,
    ):
        pass

    def eval_epoch_start(self):
        self.model.eval()

    def calculate_eval_batch_step(self, batch):
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

    def backward_step(self, loss):
        self._accelerator.backward(loss)

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizer_step(self):
        self.optimizer.step()

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def create_scheduler(self, optimizer):
        return self.scheduler_type(optimizer)

    def _create_run_config(
        self,
        num_epochs,
        per_device_batch_size,
        gradient_accumulation_steps,
        max_num_train_steps,
        fp16,
        train_dl_kwargs=None,
        eval_dl_kwargs=None,
    ):

        if train_dl_kwargs is not None:
            train_per_device_batch_size = train_dl_kwargs.get(
                "batch_size", per_device_batch_size
            )
        else:
            train_per_device_batch_size = per_device_batch_size

        if eval_dl_kwargs is not None:
            eval_per_device_batch_size = eval_dl_kwargs.get(
                "batch_size", per_device_batch_size
            )
        else:
            eval_per_device_batch_size = per_device_batch_size

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
            "eval_per_device_batch_size": eval_per_device_batch_size,
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
            "fp16": fp16,
        }
        # use SimpleNamespace or dotdict instead of dict?

        return config

    def train(
        self,
        train_dataset,
        num_epochs,
        eval_dataset=None,
        per_device_batch_size=8,
        max_num_train_steps=None,
        fp16=True,
        gradient_accumulation_steps=1,
        create_scheduler=True,
        train_dataloader_kwargs: dict = None,
        eval_dataloader_kwargs: dict = None,
    ):
        self._accelerator = Accelerator(fp16=fp16)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        (self.model, self.optimizer,) = self._accelerator.prepare(
            self.model,
            self.optimizer,
        )
        self._prepare_dataloaders(
            per_device_batch_size=per_device_batch_size,
            train_dl_kwargs=train_dataloader_kwargs,
            eval_dl_kwargs=eval_dataloader_kwargs,
        )

        # only create if doesn't exist
        if self.scheduler_type is not None and create_scheduler:
            self.scheduler = self.create_scheduler(self.optimizer)

        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
            fp16=fp16,
            train_dl_kwargs=train_dataloader_kwargs,
            eval_dl_kwargs=eval_dataloader_kwargs,
        )

        self.callback_handler.call_event(
            "on_train_run_begin",
            self,
        )
        self._run_training()
        self.callback_handler.call_event(
            "on_train_run_end",
            self,
        )

    def _prepare_dataloaders(
        self, per_device_batch_size, train_dl_kwargs=None, eval_dl_kwargs=None
    ):

        default_train_dl_kwargs = {
            "shuffle": True,
            "pin_memory": True if torch.cuda.is_available() else False,
        }

        if train_dl_kwargs is not None:
            default_train_dl_kwargs.update(train_dl_kwargs)

        if "batch_size" not in default_train_dl_kwargs:
            default_train_dl_kwargs["batch_size"] = per_device_batch_size

        default_eval_dl_kwargs = {
            "shuffle": False,
            "pin_memory": True if torch.cuda.is_available() else False,
        }

        if eval_dl_kwargs is not None:
            default_eval_dl_kwargs.update(eval_dl_kwargs)

        if "batch_size" not in default_train_dl_kwargs:
            default_train_dl_kwargs["batch_size"] = per_device_batch_size

        if "batch_size" not in default_eval_dl_kwargs:
            default_eval_dl_kwargs["batch_size"] = per_device_batch_size

        train_dataloader = self.create_train_dataloader(**default_train_dl_kwargs)
        self._train_dataloader = self._accelerator.prepare(train_dataloader)

        if self.eval_dataset is not None:
            eval_dataloader = self.create_eval_dataloader(**default_eval_dl_kwargs)
            self._eval_dataloader = self._accelerator.prepare(eval_dataloader)

    def _run_training(self):
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
                raise e

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
                if self.scheduler is not None:
                    self.scheduler_step()
                self.optimizer_zero_grad()

        self.train_epoch_end()
        self.run_history.update_metric("train_loss_epoch", self._loss_tracker.avg)
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
            batch_output = self.calculate_eval_batch_step(batch)
            self._loss_tracker.update(
                self._accelerator.gather(batch_output["loss"]).detach().mean().item(),
                batch_output["batch_size"],
            )
            self.callback_handler.call_event(
                "on_eval_step_end", self, batch_output=batch_output, batch=batch
            )
        self.eval_epoch_end()
        self.run_history.update_metric("eval_loss_epoch", self._loss_tracker.avg)
        self.callback_handler.call_event(
            "on_eval_epoch_end",
            self,
        )
