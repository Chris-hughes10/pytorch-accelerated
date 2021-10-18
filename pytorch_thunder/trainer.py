import math

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from pytorch_thunder.callbacks import (
    CallbackHandler,
    PrintMetricsCallback,
    PrintProgressCallback,
    TerminateOnNaNCallback,
    StopTrainingError,
)
from pytorch_thunder.tracking import RunHistory, InMemoryRunHistory

DEFAULT_CALLBACKS = (
    PrintMetricsCallback,
    TerminateOnNaNCallback,
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
        self.run_history: RunHistory = (
            run_history if run_history is not None else InMemoryRunHistory()
        )

        self.callback_handler = CallbackHandler(
            callbacks,
        )
        self.callback_handler.call_event("on_init_end", self)

    def create_train_dataloader(self, shuffle=True, batch_size=4, **kwargs):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def create_eval_dataloader(self, shuffle=False, batch_size=4, **kwargs):
        return DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def train_epoch_start(self):
        self.model.train()

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        loss = self.loss_func(self.model(xb), yb)

        return {"loss": loss}

    def train_epoch_end(self, train_batch_outputs):
        batch_losses = self._aggregate_losses(train_batch_outputs)
        losses = torch.cat(batch_losses)
        average_train_loss = losses.mean().item()
        self.run_history.update_metric("train_loss_epoch", average_train_loss)

    def _aggregate_losses(self, batch_outputs, move_to_cpu=False):
        losses = []
        for batch_output in batch_outputs:
            loss = self._accelerator.gather(batch_output["loss"]).detach()
            if move_to_cpu:
                loss = loss.cpu()
            if len(loss.shape) == 0:
                loss = loss[None]
            losses.append(loss)
        return losses

    def eval_epoch_start(self):
        self.model.eval()

    def calculate_eval_batch_step(self, batch):
        with torch.no_grad():
            xb, yb = batch
            val_loss = self.loss_func(self.model(xb), yb)

        return {
            "loss": val_loss,
            "batch_size": len(xb),
        }

    def eval_epoch_end(self, eval_batch_outputs):
        batch_losses = self._aggregate_losses(eval_batch_outputs)
        losses = torch.cat(batch_losses)
        average_eval_loss = losses.mean().item()
        self.run_history.update_metric("eval_loss_epoch", average_eval_loss)

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
    ):

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
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_batch_size": per_device_batch_size
            * self._accelerator.num_processes
            * gradient_accumulation_steps,
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
    ):
        self._accelerator = Accelerator(fp16=fp16)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        (self.model, self.optimizer,) = self._accelerator.prepare(
            self.model,
            self.optimizer,
        )
        self._prepare_dataloaders(per_device_batch_size=per_device_batch_size)

        # only create if doesn't exist
        if create_scheduler:
            self.scheduler = self.create_scheduler(self.optimizer)

        self.run_config = self._create_run_config(
            num_epochs=num_epochs,
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_num_train_steps=max_num_train_steps,
            fp16=fp16,
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

    def _prepare_dataloaders(self, per_device_batch_size):

        train_dataloader = self.create_train_dataloader(
            self.train_dataset,
            batch_size=per_device_batch_size,
        )

        self._train_dataloader = self._accelerator.prepare(train_dataloader)

        if self.eval_dataset is not None:

            eval_dataloader = self.create_eval_dataloader(
                self.eval_dataset,
                batch_size=per_device_batch_size,
            )

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

        train_batch_outputs = []
        for step, batch in enumerate(train_dl):
            self.callback_handler.call_event(
                "on_train_step_begin",
                self,
            )
            batch_output = self.calculate_train_batch_loss(batch)
            if self.run_config["gradient_accumulation_steps"] > 1:
                batch_output["loss"] /= self.run_config["gradient_accumulation_steps"]
            train_batch_outputs.append(batch_output)
            self.callback_handler.call_event(
                "on_train_step_end", self, batch_output=batch_output
            )
            self.backward_step(batch_output["loss"])

            if (step + 1) % self.run_config["gradient_accumulation_steps"] == 0 or (
                step + 1 == len(train_dl)
            ):
                self.optimizer_step()
                self.scheduler_step()
                self.optimizer_zero_grad()

        self.train_epoch_end(train_batch_outputs)
        self.callback_handler.call_event(
            "on_train_epoch_end", self, train_batch_outputs=train_batch_outputs
        )

    def _run_eval_epoch(self, valid_dl):
        self.callback_handler.call_event(
            "on_eval_epoch_begin",
            self,
        )
        self.eval_epoch_start()
        eval_batch_outputs = []
        for batch in valid_dl:
            self.callback_handler.call_event(
                "on_eval_step_begin",
                self,
            )
            batch_output = self.calculate_eval_batch_step(batch)
            eval_batch_outputs.append(batch_output)
            self.callback_handler.call_event(
                "on_eval_step_end", self, batch_output=batch_output
            )
        self.eval_epoch_end(eval_batch_outputs)
        self.callback_handler.call_event(
            "on_eval_epoch_end", self, eval_batch_outputs=eval_batch_outputs
        )
