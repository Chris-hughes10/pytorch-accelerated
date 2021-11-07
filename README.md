# pytorch-accelerated

`pytorch-accelerated` is a lightweight library designed to accelerate the process of training PyTorch models across 
different hardware options by providing a minimal, but extensible training loop - encapsulated in a single `Trainer` 
object - which is flexible enough to handle the majority of use cases.

The key features are:
- A simple and contained, but easily customisable, training loop which should out of the box in straightforward cases;
 behaviour can be customised using inheritance and/or callbacks
- Handles device placement, mixed-precision, multi GPU and distributed training with no code changes
- Uses pure PyTorch components, with no additional modifications or wrappers, and easily interoperates
 with other popular libraries such as [timm](https://github.com/rwightman/pytorch-image-models), 
 [transformers](https://huggingface.co/transformers/) and [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/)
- A small, streamlined API ensures that there is a minimal learning curve for existing PyTorch users 

Significant effort has been taken to ensure that every part of the `Trainer` is as clear and simple as possible, 
making it easy to customise, debug and understand exactly what is going on behind the scenes at each step. 
In the spirit of Python, nothing is hidden and most of the behaviour of the trainer is contained in a singe class!

`pytorch-accelerated` is proudly and transparently built on top of 
[huggingface accelerate](https://github.com/huggingface/accelerate), which is responsible for the 
movement of data between devices and launching of training configurations. When customizing the trainer, or launching
training, users are encouraged to consult the [accelerate documentation](https://huggingface.co/docs/accelerate/) to understand all available options;
 accelerate provides convenient functions for operations such gathering tensors and gradient clipping, 
usage of which can be seen in the `pytorch-accelerated` examples folder! 

## Installation

Unfortunately at the moment, you have to clone the repo. Pip install coming soon!

## Quickstart

To get started, simply import and use the pytorch-accelerated `Trainer` ,as demonstrated in the following snippet,
and then launch training using the 
[accelerate CLI](https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script)
described below.

```python
# examples/vision/train_mnist.py
import os

from torch import nn, optim
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

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
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input):
        return self.main(input.view(input.shape[0], -1))

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_dataset, validation_dataset = random_split(dataset, [55000, 5000])
model = MNISTModel()
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

depending on your infrastructure configuration, for users who would like to maintain a more fine-grained control 
over the launch command.

More complex training examples can be seen in the examples folder [here](TODO).

## Usage

### Who should use pytorch-accelerated
- Users that are familiar with PyTorch but would like to avoid having to write the common training loop boilerplate
to focus on the interesting parts of the training loop
- Users who like, and are comfortable with, selecting their own models, loss functions and optimizers

### Who should not use pytorch-accelerated
- Users looking for an end-to-end solution where they don't have to specify a model, optimizer or loss function. pytorch-accelerate focuses only on the training loop, other concerns such 
as loading data or creating a model are left to the responsibility of the user
- Users who would like to write the training loop themselves -> use accelerate
- Users working on highly complex, custom use cases which do not fit the patterns of usual training loops. 
In highly specialised use-cases any high-level API becomes an overhead, and vanilla PyTorch is probably the best option

##The Trainer Class

The core idea behind `pytorch-accelerated`, is that the entire training loop is abstracted behind a single interface - the `Trainer` class. 

The trainer has been implemented such that it provides (overridable) implementations of the parts of training that 
rarely change after they have been defined – such as creating a data loader, or how a batch of data is fed to the model 
– whilst remaining decoupled from components that are likely to change, such as the model, dataset, loss function and 
optimizer.

### Overridable hooks

Whilst the trainer should work out of the box in straightforward use cases, subclassing the trainer and overriding
its methods is intended and encouraged - think of the base implementation as a set of 'sensible defaults'!

Methods which are prefixed with a verb such as *create* or *calculate* expect a value to be returned,
all other methods are used to set internal state (e.g. `optimizer.step()`)

#### Setup methods
- `create_train_dataloader`
- `create_eval_dataloader`
- `create_scheduler`

#### Training loop methods
- `training_run_start`
- `training_run_end`

##### train epoch
- `train_epoch_start`
- `calculate_train_batch_loss`
- `backward_step`
- `optimizer_step`
- `scheduler_step`
- `optimizer_zero_grad`
- `train_epoch_end`

##### evaluation epoch
- `eval_epoch_start`
- `calculate_eval_batch_loss`
- `eval_epoch_end`

### Callbacks

In addition to overridable hooks, the trainer also includes a callback system. It is recommended that callbacks are used
to contain 'infrastructure' code, which is not essential to the operation of the training loop, such as logging, but
this decision is left to the judgement of the user based on the specific use case.

Callbacks are executed sequentially, so if a callback is used to modify state, such as updating a metric, it is the 
responsibility of the user to ensure that this callback is placed before any callback which will read this state 
(i.e. for logging purposes).

**Note**: callbacks are called **after** their corresponding hooks, e.g., the method `train_epoch_end` is called
 *before* the callback `on_train_epoch_end`. This is done to support the pattern of updating the trainer's state in a
 method before reading this state in a callback.
 
 The available callbacks are:
- `on_init_end`
- `on_train_run_begin`
- `on_train_run_end`
- `on_stop_training_error`

#### train epoch
- `on_train_epoch_begin`
- `on_train_step_begin`
- `on_train_step_end`
- `on_train_epoch_end`

#### eval epoch
- `on_eval_epoch_begin`
- `on_eval_step_begin`
- `on_eval_step_end`
- `on_eval_epoch_end`

### Trainer Internals

In pseudocode, the execution of the training loop can be depicted as:
```python

train_dl = create_train_dataloader()
eval_dl = create_eval_dataloader()
scheduler = create_scheduler()

training_run_start()
on_train_run_begin()

for epoch in num_epochs:
    train_epoch_start()
    on_train_epoch_begin()
    for batch in train_dl:
        on_train_step_begin()
        batch_output = calculate_train_batch_loss(batch)
        on_train_step_end(batch, batch_output)
        backward_step(batch_output["loss"])
        optimizer_step()
        scheduler_step()
        optimizer_zero_grad()
    train_epoch_end()
    on_train_epoch_end()
    
    eval_epoch_start()
    on_eval_epoch_begin()
    for batch in eval_dl:
        on_eval_step_begin()
        batch_output = calculate_eval_batch_loss(batch)
        on_eval_step_end(batch, batch_output)
    eval_epoch_end()
    on_eval_epoch_end()
    
training_run_end()
```

The best way to understand how the trainer works internally is by examining the source code for the `train` method;
significant care has gone into making the internal methods as clean and clear as possible.







# Acknowledgements
- fastai, pytorch-lightning, Huggingface 
- taken inspiration from all of these frameworks
