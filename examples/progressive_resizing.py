import os
from collections import namedtuple
from functools import partial
from pathlib import Path

import torch
from accelerate import notebook_launcher
from timm import create_model
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torchmetrics import Accuracy
from torchvision import transforms, datasets

from pytorch_accelerated.callbacks import (
    TerminateOnNaNCallback,
    PrintMetricsCallback,
    PrintProgressCallback,
    EarlyStoppingCallback,
    SaveBestModelCallback,
    TrainerCallback,
    ProgressBarCallback)
from pytorch_accelerated.trainer import Trainer


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


def main():
    # Create datasets
    # data_dir = (
    #     r"C:\Users\hughesc\OneDrive - Microsoft\Documents\toy_data\hymenoptera_data"
    # )

    data_dir = Path(r"/home/chris/notebooks/imagenette2/")
    data_dir = Path(r"/home/chris/notebooks/hymenoptera_data/")

    num_classes = 2

    # model = create_model(number_of_classes=2)
    model = create_model("resnet50d", pretrained=False, num_classes=num_classes)

    # Freezing the base model
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.get_classifier().parameters():
    #     param.requires_grad = True

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = partial(StepLR, step_size=7, gamma=0.1)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01/25)
    # lr_scheduler = partial(
    #     OneCycleLR,
    #     max_lr=0.01,
    #     epochs=8,
    #     steps_per_epoch=148
    # )

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler_type=lr_scheduler,
        callbacks=(
            TerminateOnNaNCallback,
            AccuracyCallback(num_classes=num_classes),
            PrintMetricsCallback,
            PrintProgressCallback,
            ProgressBarCallback(),
            EarlyStoppingCallback(early_stopping_patience=2),
            SaveBestModelCallback(watch_metric="accuracy", greater_is_better=True),
        ),
    )


    EpochConfig = namedtuple(
        "EpochConfig", ["num_epochs", "train_image_size", "eval_image_size"]
    )

    epoch_configs = [
        EpochConfig(num_epochs=3, train_image_size=64, eval_image_size=64),
        EpochConfig(num_epochs=3, train_image_size=128, eval_image_size=128),
        EpochConfig(num_epochs=3, train_image_size=224, eval_image_size=224),
    ]


    for e_config in epoch_configs:
        trainer.print("Starting phase")

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
        trainer.train(
            train_dataset=image_datasets["train"],
            eval_dataset=image_datasets["val"],
            num_epochs=e_config.num_epochs,
            per_device_batch_size=32,
        )


if __name__ == "__main__":
    # notebook_launcher(main, num_processes=2)
    main()