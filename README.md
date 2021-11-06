# pytorch-accelerated

pytorch-accelerated is a lightweight library designed to accelerate the process of training PyTorch models across 
different hardware options by providing a minimal, but extensible training loop which is flexible enough to handle 
most use cases.

The key features are:
- Library providing a minimal, but extensible training loop for PyTorch which is flexible enough for most use cases
- Handles the device placement for you, fp16, multi GPU and distributed training with no code changes
- Focuses on using pure PyTorch components, interoperation with other libraries e.g. timm, transformers, torchmetrics
- Minimal learning curve
- Key aims are simplicity and transparency. Every part of the trainer is designed to be as simple and clear as possible,
making it easy to debug, and to understand exactly what is happening behind the scenes at each step

- Proudly built on top of huggingface accelerate

## Installation

Unfortunately at the moment, you have to clone the repo. Pip install coming soon!

## Quickstart

To get started, simply import and use the pytorch-accelerated `Trainer` ,as demonstrated in the following snippet,
and then launch training using the 
[accelerate CLI](https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script)
described below.

```python
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
```

To launch training using the [accelerate CLI](https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script)
, on your machine(s), run:

` accelerate config --config_file accelerate_config.yaml`

and answer the questions asked. This will generate a config file that will be used to properly set the default options when doing

`accelerate launch --config_file accelerate_config.yaml train.py [--training-args]`

*Note*: Using the [accelerate CLI](https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script) is completely optional, training can also be launched in the usual way using:

`python train.py` / `python -m torch.distributed ...`

depending on your infrastructure configuration, if you would like to maintain a more fine-grained control 
over the launch command.


## Who should use pytorch-accelerated
Users that are familiar with PyTorch but would like to avoid having to write the common training loop boilerplate
to focus on the interesting parts of the training loop

## Who should not use accelerate
- Users looking for an end-to-end solution where they don't have to specify a model, optimizer or loss function. pytorch-accelerate focuses only on the training loop, other concerns such 
as loading data or creating a model are left to the responsibility of the user
- Users who would like to write the training loop themselves -> use accelerate
- Users working on highly complex, custom use cases which do not fit the patterns of usual training loops. 
In highly specialised use-cases any high-level API becomes an overhead, and vanilla PyTorch is probably the best option

# Features

# Training Loop Class Definition
The core idea here, is that the entire training loop is abstracted behind a single interface, the ‘Trainer’ class. 
The trainer should be implemented in such a way that it provides (overridable) implementations of the things that 
rarely change after they have been defined – such as creating a data loader, or how a batch of data is fed to the model 
– whilst remaining decoupled from components such as the model and the dataset; the arguments specific to the training 
run are passed as arguments to the train function.

- overriding the trainer is encouraged, think of the default implementation as a set of 'sensible defaults'

# Acknowledgements
- fastai, pytorch-lightning, Huggingface 
- taken inspiration from all of these frameworks
