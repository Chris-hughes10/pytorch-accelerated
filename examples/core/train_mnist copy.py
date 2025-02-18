# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a model on the MNIST Dataset

# Note: this example requires installing the torchvision package
########################################################################

from pathlib import Path

from torch import nn, optim
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import WSDCheckpointCallback
from pytorch_accelerated.schedulers.wsd_scheduler import WSDLrScheduler
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


DATA_PATH = Path("/".join(Path(__file__).absolute().parts[:-2])) / "data"


def main():
    dataset = MNIST(DATA_PATH, download=True, transform=transforms.ToTensor())
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
        callbacks=[*DEFAULT_CALLBACKS, WSDCheckpointCallback()],
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        num_epochs=2,
        per_device_batch_size=32,
        create_scheduler_fn=WSDLrScheduler.create_scheduler_fn(),
    )

    trainer.evaluate(
        dataset=test_dataset,
        per_device_batch_size=64,
    )


if __name__ == "__main__":
    main()
