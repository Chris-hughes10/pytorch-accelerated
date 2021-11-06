import os

from torch import nn, optim
from torch.utils.data import random_split
from torchvision import models, transforms
from torchvision.datasets import MNIST

from pytorch_accelerated.trainer import Trainer

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_dataset, validation_dataset = random_split(dataset, [55000, 5000])
model = models.resnet18()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
    )

trainer.train(
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        num_epochs=8,
        per_device_batch_size=32,
    )