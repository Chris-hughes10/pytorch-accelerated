# pytorch-accelerated

`pytorch-accelerated` is a lightweight library designed to accelerate the process of training PyTorch models
 by providing a minimal, but extensible training loop - encapsulated in a single `Trainer` 
object - which is flexible enough to handle the majority of use cases, and capable of utilizing different hardware
 options with no code changes required.
   
The key features are:
- A simple and contained, but easily customisable, training loop, which should work out of the box in straightforward cases;
 behaviour can be customised using inheritance and/or callbacks.
- Handles device placement, mixed-precision, multi GPU and distributed training with no code changes.
- Uses pure PyTorch components, with no additional modifications or wrappers, and easily interoperates
 with other popular libraries such as [timm](https://github.com/rwightman/pytorch-image-models), 
 [transformers](https://huggingface.co/transformers/) and [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/).
- A small, streamlined API ensures that there is a minimal learning curve for existing PyTorch users.

Significant effort has been taken to ensure that every part of the library - both internal and external components - is as clear and simple as possible, 
making it easy to customise, debug and understand exactly what is going on behind the scenes at each step; most of the 
behaviour of the trainer is contained in a single class! 
In the spirit of Python, nothing is hidden and everything is accessible.

`pytorch-accelerated` is proudly and transparently built on top of 
[Hugging Face Accelerate](https://github.com/huggingface/accelerate), which is responsible for the 
movement of data between devices and launching of training configurations. When customizing the trainer, or launching
training, users are encouraged to consult the [Accelerate documentation](https://huggingface.co/docs/accelerate/) 
to understand all available options; Accelerate provides convenient functions for operations such gathering tensors 
and gradient clipping, usage of which can be seen in the `pytorch-accelerated` 
[examples](https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/examples) folder! 

## Installation

As this package is still in the alpha stage of development, it is not yet available on pypi.
 
For the moment, please pip install from the main branch using the following command:
```
pip install git+https://github.com/Chris-hughes10/pytorch-accelerated.git
```

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

More complex training examples can be seen in the examples folder 
[here](https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/examples).

## Usage

### Who is pytorch-accelerated aimed at?

- Users that are familiar with PyTorch but would like to avoid having to write the common training loop boilerplate
to focus on the interesting parts of the training loop.
- Users who like, and are comfortable with, selecting and creating their own models, loss functions, optimizers and datasets.
- Users who value a simple and streamlined feature set, where the behaviour is easy to debug, understand, and reason about!

### Who should not use pytorch-accelerated?

- If you are looking for an end-to-end solution, encompassing everything from loading data to inference,
  which helps you to select a model, optimizer or loss function, you would probably be better suited to
  [fastai](https://github.com/fastai/fastai). `pytorch-accelerate` focuses only on the training process, with all other
  concerns being left to the responsibility of the user.
- If you would like to write the entire training loop yourself, just without all of the device management headaches, 
you would probably be best suited to using [Accelerate](https://github.com/huggingface/accelerate) directly! Whilst it
is possible to customize every part of the `Trainer`, the training loop is fundamentally broken up into a number of 
different methods that you would have to override. But, before you go, is writing those `for` loops really important 
enough to warrant starting from scratch *again* 😉.
- If you are working on a custom, highly complex, use case which does not fit the patterns of usual training loops
and want to squeeze out every last bit of performance on your chosen hardware, you are probably best off sticking
 with vanilla PyTorch; any high-level API becomes an overhead in highly specialized cases!

## Navigation
1. [The Trainer Class](#the-trainer-class)
2. [Training a model](#training-a-model)
    - [Using Learning Rate Schedulers](#using-learning-rate-schedulers) 
3. [Customizing Trainer Behaviour](#customizing-trainer-behaviour) 
    - [Overridable methods](#overridable-methods)
    - [Callbacks](#callbacks)
4. [Trainer Internals](#trainer-internals)
5. [Acknowledgements](#acknowledgements)

## The Trainer Class

The core idea behind `pytorch-accelerated`, is that the entire training loop is abstracted behind a single interface - 
the `Trainer` class. 

The `Trainer` is designed to encapsulate an entire training loop for a specific task, bringing together the model,
loss function and optimizer, and providing a specification of the behaviour to execute of each step of the training 
process.

The `Trainer` has been implemented such that it provides (overridable) implementations of the parts of training that 
rarely change after they have been defined – such as creating a data loader, or how a batch of data is fed to the model 
– whilst remaining decoupled from components that are likely to change, such as the model, dataset, loss function and 
optimizer.

### Training a model

After initialising a trainer instance with a model, loss function and optimizer like so:
```python
from pytorch_accelerated.trainer import Trainer

trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
    )
```
all that is left to do is to call the `train` method with a subset of the following parameters:
- `train_dataset`: the dataset object to use during training epochs (*required*)
- `num_epochs`: the number of epochs to train for (*required*)
- `eval_dataset`: the dataset to use during evaluation epochs, if this is not provided, evaluation is skipped.
- `per_device_batch_size`: the batch size to use per device
- `max_num_train_steps`: the maximum number of steps to train for. If provided, this will override `num_epochs`
- `gradient_accumulation_steps`: accumulate grads to the specified number of steps to simulate a bigger batch size. By default, this is set to 1
- `gradient_clip_value`: if specified, the gradients of the model's parameters will be clipped to within the range [-`gradient_clip_value`, `gradient_clip_value`]
- `create_scheduler_fn`: a function which accepts an optimizer as an argument and returns a learning rate scheduler
- `train_dataloader_kwargs`: : a dictionary of keyword arguments to pass to the training dataloader constructor, for details see torch.utils.data.DataLoader
- `eval_dataloader_kwargs`: a dictionary of keyword arguments to pass to the evaluation dataloader constructor, for details see torch.utils.data.DataLoader
- `reset_run_history`: reset any run history saved by the trainer from previous training runs
- `collate_fn`: the collate function to be used by the training and evaluation dataloaders

#### Using learning rate schedulers

Note that, as the optimizer needs to be internally prepared prior to training, in order to use a learning rate scheduler,
a factory function must be provided to `create_scheduler_fn`. This must be a function which accepts the optimizer as a 
single parameter and returns an instance of a learning rate scheduler. 
Passing an instance of a learning rate scheduler will **not** work here.

A simple method of doing this is by using `functools.partial` like so: 
```
from functools import Partial

from torch.optim import lr_scheduler

create_scheduler_fn = partial(lr_scheduler.StepLR, step_size=7, gamma=0.1)
```

##### More complex cases 

Some learning rate schedulers require information such as the total number of steps that will take place during training.
As this information is not accessible prior to creating the training dataloader - which will be done as part of the 
`train` method - a placeholder value can be used in the cases, as demonstrated below:

```
from functools import Partial

from torch.optim.lr_scheduler import OneCycleLR

create_scheduler_fn = partial(
            OneCycleLR,
            max_lr=e_config.lr,
            epochs=TrainerPlaceholderValues.NUM_EPOCHS,
            steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        )
```
These placeholders will be replaced by the trainer with the correct values during training.

A list of the available placeholders are:
- `NUM_EPOCHS` 
- `NUM_UPDATE_STEPS_PER_EPOCH`
- `TRAIN_DATALOADER_LEN` 
- `EVAL_DATALOADER_LEN`

Alternatively, the same outcome could be achieved by overriding the `Trainer`'s `create_scheduler` method.

## Customizing Trainer behaviour

### Overridable methods

Whilst the `Trainer` should work out of the box in straightforward use cases, subclassing the trainer and overriding
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

**Note**: callbacks are called **after** their corresponding hooks, e.g., the callback`on_train_epoch_end` is called
 *after* the method `train_epoch_end`. This is done to support the pattern of updating the trainer's state in a
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

## Trainer Internals

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


## Acknowledgements

Many aspects behind the design and features of `pytorch-accelerated` were greatly inspired by a number of excellent 
libraries and frameworks such as [fastai](https://github.com/fastai/fastai), [timm](https://github.com/rwightman/pytorch-image-models), 
[PyTorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [Hugging Face Accelerate](https://github.com/huggingface/accelerate). Each of these tools 
have made an enormous impact on both this library and the machine learning community, and their influence can not be 
stated enough!

`pytorch-accelerate` has taken only inspiration from these tools, and all of the functionality contained has been implemented
 from scratch in a way that benefits this library. The only exceptions to this are some of the scripts in the 
 [examples](https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/examples)
 folder in which existing resources were taken and modified in order to showcase the features of `pytorch-accelerated`;
 these cases are clearly marked, with acknowledgement being given to the original authors.
 
