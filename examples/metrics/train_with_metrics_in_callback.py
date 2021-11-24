# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a model on the MNIST Dataset

# This example demonstrates how a callback can be used to track metrics
# to avoid having to override the default trainer class
#
# Note, this example requires installing the torchmetrics package
########################################################################

import os

from torch import nn, optim
from torch.utils.data import random_split
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import TrainerCallback
from pytorch_accelerated.trainer import DEFAULT_CALLBACKS


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

    def forward(self, x):
        return self.main(x.view(x.shape[0], -1))


class ClassificationMetricsCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "precision": Precision(num_classes=num_classes),
                "recall": Recall(num_classes=num_classes),
            }
        )

    def _move_to_device(self, trainer):
        self.metrics.to(trainer.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.metrics.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        metrics = self.metrics.compute()
        trainer.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
        trainer.run_history.update_metric("precision", metrics["precision"].cpu())
        trainer.run_history.update_metric("recall", metrics["recall"].cpu())

        self.metrics.reset()


def main():
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    num_classes = len(dataset.class_to_idx)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [50000, 5000, 5000]
    )
    model = MNISTModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            ClassificationMetricsCallback(
                num_classes=num_classes,
            ),
            *DEFAULT_CALLBACKS,
        ),
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        num_epochs=2,
        per_device_batch_size=32,
    )

    trainer.evaluate(
        dataset=test_dataset,
        per_device_batch_size=64,
    )


if __name__ == "__main__":
    main()
