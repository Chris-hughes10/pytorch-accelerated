import os
from functools import partial
from typing import Optional, Generator

import torch
from accelerate import notebook_launcher
from accelerate.utils import set_seed
from torch import nn, optim
from torch.optim import lr_scheduler, Optimizer
from torchvision import transforms, datasets, models

from pytorch_thunder.trainer import Trainer

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module, train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(
    module: torch.nn.Module, n: Optional[int] = None, train_bn: bool = True
) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: torch.nn.Module, train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(
    module: torch.nn.Module,
    optimizer: Optimizer,
    lr: Optional[float] = None,
    train_bn: bool = True,
):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]["lr"] if lr is None else float(lr)
    optimizer.add_param_group(
        {
            "params": filter_params(module=module, train_bn=train_bn),
            "lr": params_lr / 10.0,
        }
    )


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


def create_model(number_of_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)

    return model


def main():
    set_seed(42)

    # Create datasets
    data_dir = (
        r"C:\Users\hughesc\OneDrive - Microsoft\Documents\toy_data\hymenoptera_data"
    )
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            create_transforms(train_image_size=150, val_image_size=150)[x],
        )
        for x in ["train", "val"]
    }

    model = create_model(number_of_classes=len(image_datasets["train"].classes))

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = partial(lr_scheduler.StepLR, step_size=7, gamma=0.1)

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler_type=exp_lr_scheduler,
    )

    trainer.train(
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        num_epochs=3,
        per_device_batch_size=4,
        fp16=True,
    )

    resized_image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            create_transforms(train_image_size=225, val_image_size=250)[x],
        )
        for x in ["train", "val"]
    }

    trainer.train(
        train_dataset=resized_image_datasets["train"],
        eval_dataset=resized_image_datasets["val"],
        num_epochs=8,
        per_device_batch_size=4,
        fp16=True,
    )


if __name__ == "__main__":
    notebook_launcher(main, num_processes=1)
