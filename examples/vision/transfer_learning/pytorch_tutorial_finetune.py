# Modifications Copyright © 2021 Chris Hughes
########################################################################
# This is an accelerated example of the PyTorch "Transfer Learning for Computer Vision Tutorial"
# written by Sasank Chilamkurthy, available here:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html,
# which demonstrates fine-tuning a ResNet18 model to classify ants and bees.
#
# Note: this example requires installing the torchvision package
########################################################################
import os
from functools import partial
from pathlib import Path

from func_to_script import script
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

from pytorch_accelerated.trainer import Trainer, TrainerPlaceholderValues

DATA_PATH = (
    Path("/".join(Path(__file__).absolute().parts[:-3])) / "data/hymenoptera_data"
)


@script
def main(data_dir: str = DATA_PATH):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    # Create model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(image_datasets["train"].classes))

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # The Trainer calls schedulers after every step, not every epoch as StepLr expects
    # To use this as intended, we need to represent the step size as the number of iterations
    exp_lr_scheduler = partial(
        lr_scheduler.StepLR,
        step_size=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH * 7,
        gamma=0.1,
    )

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
    )

    trainer.train(
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        num_epochs=8,
        per_device_batch_size=4,
        create_scheduler_fn=exp_lr_scheduler,
    )


if __name__ == "__main__":
    main()
