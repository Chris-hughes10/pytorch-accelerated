# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a model on the MNIST Dataset and demonstrates using
# a custom callback event

# Note: this example requires installing the torchvision package
########################################################################

import os

from accelerate import notebook_launcher
from torch import nn, optim
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_accelerated.callbacks import (
    TrainerCallback,
    TerminateOnNaNCallback,
    PrintProgressCallback,
    ProgressBarCallback,
    LogMetricsCallback,
)
from pytorch_accelerated.trainer import Trainer


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1))


class VerifyBatchCallback(TrainerCallback):
    def verify_train_batch(self, trainer, xb, yb):
        assert xb.shape[0] == trainer.run_config["train_per_device_batch_size"]
        assert xb.shape[1] == 1
        assert xb.shape[2] == 28
        assert xb.shape[3] == 28
        assert yb.shape[0] == trainer.run_config["train_per_device_batch_size"]


class TrainerWithCustomCallbackEvent(Trainer):
    def calculate_train_batch_loss(self, batch) -> dict:
        xb, yb = batch
        self.callback_handler.call_event(
            "verify_train_batch", trainer=self, xb=xb, yb=yb
        )
        return super().calculate_train_batch_loss(batch)


def main():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_dataset, validation_dataset = random_split(dataset, [55000, 5000])
    model = MNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    trainer = TrainerWithCustomCallbackEvent(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            VerifyBatchCallback,
            TerminateOnNaNCallback,
            PrintProgressCallback,
            ProgressBarCallback,
            LogMetricsCallback,
        ),
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        num_epochs=8,
        train_dataloader_kwargs={"num_workers": 0},
        per_device_batch_size=32,
    )


if __name__ == "__main__":
    notebook_launcher(main, num_processes=1)
    # main()
