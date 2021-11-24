# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a model on the MNIST Dataset

# This example demonstrates how the default trainer class can be overridden
# so that we can record classification metrics
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


class TrainerWithMetrics(Trainer):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this will be moved to the correct device automatically by the
        # MoveModulesToDeviceCallback callback, which is used by default
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(num_classes=num_classes),
                "precision": Precision(num_classes=num_classes),
                "recall": Recall(num_classes=num_classes),
            }
        )

    def calculate_eval_batch_loss(self, batch):
        batch_output = super().calculate_eval_batch_loss(batch)
        preds = batch_output["model_outputs"].argmax(dim=-1)

        self.metrics.update(preds, batch[1])

        return batch_output

    def eval_epoch_end(self):
        metrics = self.metrics.compute()
        self.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
        self.run_history.update_metric("precision", metrics["precision"].cpu())
        self.run_history.update_metric("recall", metrics["recall"].cpu())

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

    trainer = TrainerWithMetrics(
        model=model, loss_func=loss_func, optimizer=optimizer, num_classes=num_classes
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
