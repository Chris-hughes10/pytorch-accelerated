########################################################################
# This is an accelerated example of the PyTorch "Transfer Learning for Computer Vision Tutorial"
# written by Sasank Chilamkurthy, available here:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html,
# which demonstrates fine-tuning a ResNet18 model to classify ants and bees.
#
# This example demonstrates how a callback can be used to track metrics
# to avoid having to override the default trainer class
########################################################################
import argparse
import os
from functools import partial

from accelerate.utils import set_seed
from torch import nn, optim
from torch.optim import lr_scheduler
from torchmetrics import ConfusionMatrix, Accuracy
from torchvision import transforms, datasets, models

from pytorch_accelerated.callbacks import (
    TrainerCallback,
    PrintMetricsCallback,
    TerminateOnNaNCallback,
    PrintProgressCallback,
)
from pytorch_accelerated.trainer import Trainer


class ClassificationMetricsCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.cm_metrics = ConfusionMatrix(num_classes=num_classes)
        self.accuracy = Accuracy(num_classes=num_classes)

    def on_train_run_begin(self, trainer, **kwargs):
        self.cm_metrics.to(trainer._eval_dataloader.device)
        self.accuracy.to(trainer._eval_dataloader.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.cm_metrics.update(preds, batch[1])
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric(
            "confusion_matrix", self.cm_metrics.compute().cpu()
        )
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())

        self.cm_metrics.reset()
        self.accuracy.reset()


def main(data_dir):
    set_seed(42)

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
    num_classes = len(image_datasets["train"].classes)

    # Create model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define loss function
    loss_func = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = partial(lr_scheduler.StepLR, step_size=7, gamma=0.1)

    cm_callback = ClassificationMetricsCallback(
        num_classes=num_classes,
    )

    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            cm_callback,
            TerminateOnNaNCallback,
            PrintMetricsCallback,
            PrintProgressCallback,
        ),
    )

    trainer.train(
        train_dataset=image_datasets["train"],
        eval_dataset=image_datasets["val"],
        num_epochs=8,
        per_device_batch_size=4,
        create_scheduler_fn=exp_lr_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple examples of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    main(args.data_dir)
