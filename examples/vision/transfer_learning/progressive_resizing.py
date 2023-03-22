# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a ResNet50d on the Imagenette Dataset using progressive resizing

# Note: this example requires installing the torchmetrics and timm packages
########################################################################
import argparse
import os
from collections import namedtuple
from functools import partial
from pathlib import Path
from func_to_script import script

import torch
from timm import create_model
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy
from torchvision import transforms, datasets

from pytorch_accelerated.callbacks import (
    TerminateOnNaNCallback,
    LogMetricsCallback,
    PrintProgressCallback,
    EarlyStoppingCallback,
    SaveBestModelCallback,
    TrainerCallback,
    ProgressBarCallback,
)

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues


class AccuracyCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def on_training_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer._eval_dataloader.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()


def create_transforms(train_image_size=224, val_image_size=224):
    # Data augmentation and normalization for training
    # Just normalization for validation
    return {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(train_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(int(round(1.15 * val_image_size))),
                transforms.CenterCrop(val_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }


DATA_PATH = (
    Path("/".join(Path(__file__).absolute().parts[:-3])) / "data/imagenette2-320"
)


@script
def main(data_dir: str = DATA_PATH):
    data_dir = Path(data_dir)
    num_classes = len(list((data_dir / "train").iterdir()))

    model = create_model("resnet50d", pretrained=False, num_classes=num_classes)

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01 / 25)

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            TerminateOnNaNCallback,
            AccuracyCallback(num_classes=num_classes),
            PrintProgressCallback,
            ProgressBarCallback,
            LogMetricsCallback,
            EarlyStoppingCallback(early_stopping_patience=2),
            SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
        ),
    )

    EpochConfig = namedtuple(
        "EpochConfig", ["num_epochs", "train_image_size", "eval_image_size", "lr"]
    )

    epoch_configs = [
        EpochConfig(num_epochs=2, train_image_size=64, eval_image_size=64, lr=0.01),
        EpochConfig(num_epochs=3, train_image_size=128, eval_image_size=128, lr=0.01),
        EpochConfig(num_epochs=6, train_image_size=224, eval_image_size=224, lr=0.001),
    ]

    for e_config in epoch_configs:
        trainer.print(f"Training with image size: {e_config.train_image_size}")

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x),
                create_transforms(
                    train_image_size=e_config.train_image_size,
                    val_image_size=e_config.eval_image_size,
                )[x],
            )
            for x in ["train", "val"]
        }

        # Here we use placeholders for the number of epochs and number of steps per epoch, so that the
        # trainer can inject those values later. This is especially key for the number of update steps
        # which will change depending on whether training is distributed or not
        lr_scheduler = partial(
            OneCycleLR,
            max_lr=e_config.lr,
            epochs=TrainerPlaceholderValues.NUM_EPOCHS,
            steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        )

        trainer.train(
            train_dataset=image_datasets["train"],
            eval_dataset=image_datasets["val"],
            num_epochs=e_config.num_epochs,
            create_scheduler_fn=lr_scheduler,
            per_device_batch_size=32,
            reset_run_history=False,
        )


if __name__ == "__main__":
    main()
