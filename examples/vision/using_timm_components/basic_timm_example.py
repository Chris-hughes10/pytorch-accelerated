from functools import partial
from pathlib import Path

import torch
import timm.data
import timm.optim
import timm.scheduler
import timm.loss
from accelerate import notebook_launcher
from torchmetrics import Accuracy

from pytorch_accelerated.callbacks import (
    TrainerCallback,
    TerminateOnNaNCallback,
    PrintMetricsCallback,
    PrintProgressCallback,
    ProgressBarCallback,
)
from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS


class TimmTrainer(Trainer):
    def __init__(self, eval_loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_updates = None
        self.train_loss_func = kwargs['loss_func']
        self.eval_loss_func = eval_loss_func

    def create_train_dataloader(self, **kwargs):
        kwargs.pop("shuffle")

        return timm.data.create_loader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **kwargs
        )

    def create_eval_dataloader(self, **kwargs):
        kwargs.pop("shuffle")

        return timm.data.create_loader(
            dataset=self.eval_dataset, collate_fn=self.collate_fn, **kwargs
        )

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)
        self.loss_func = self.train_loss_func

    def eval_epoch_start(self):
        super().eval_epoch_start()
        self.loss_func = self.eval_loss_func

    def eval_epoch_end(self):
        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)


class AccuracyCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.accuracy = Accuracy(num_classes=num_classes)

    def on_train_run_begin(self, trainer, **kwargs):
        self.accuracy.to(trainer._eval_dataloader.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


def main():
    data_path = Path(r"/home/chris/notebooks/imagenette2/")

    train_path = data_path / "train"
    val_path = data_path / "val"

    batch_size = 32
    num_epochs = 10
    opt = "adamp"
    lr = 5e-3
    model = "resnetrs50"
    num_classes = len(list(train_path.iterdir()))
    pretrained = True

    args_train_split = "train"
    args_val_split = "val"

    # Create model
    model = timm.create_model(
        model,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Create datasets and config for timm dataloaders
    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)

    dataset_train = timm.data.create_dataset(
        "imagenette",
        root=data_path,
        split=args_train_split,
        is_training=True,
        batch_size=batch_size,
    )
    dataset_eval = timm.data.create_dataset(
        "imagenette",
        root=data_path,
        split=args_val_split,
        is_training=False,
        batch_size=batch_size,
    )

    train_dl_kwargs = {
        "input_size": data_config["input_size"],
        "is_training": True,
        "use_prefetcher": False,
        "mean": data_config["mean"],
        "std": data_config["std"],
        "interpolation": data_config["interpolation"],
        "num_workers": 0,
        "distributed": False,
        "pin_memory": True,
        "persistent_workers": False,
    }

    eval_dl_kwargs = {
        "input_size": data_config["input_size"],
        "is_training": False,
        "interpolation": data_config["interpolation"],
        "mean": data_config["mean"],
        "std": data_config["std"],
        "num_workers": 0,
        "distributed": False,
        "crop_pct": data_config["crop_pct"],
        "pin_memory": True,
        "use_prefetcher": False,
        "persistent_workers": False,
    }

    # Create optimizer and scheduler
    optimizer = timm.optim.create_optimizer_v2(
        model,
        opt,
        lr,
    )

    lr_scheduler_type = partial(timm.scheduler.CosineLRScheduler, t_initial=num_epochs)

    # Define loss functions
    train_loss_fn = timm.loss.LabelSmoothingCrossEntropy()
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    trainer = TimmTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=train_loss_fn,
        eval_loss_func=validate_loss_fn,
        scheduler_type=lr_scheduler_type,
        callbacks=(
            TerminateOnNaNCallback,
            AccuracyCallback(num_classes=num_classes),
            PrintMetricsCallback,
            PrintProgressCallback,
            ProgressBarCallback,
        ),
    )

    trainer.train(
        per_device_batch_size=64,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        num_epochs=num_epochs,
        train_dataloader_kwargs=train_dl_kwargs,
        eval_dataloader_kwargs=eval_dl_kwargs,
    )


if __name__ == "__main__":
    notebook_launcher(main, num_processes=2)