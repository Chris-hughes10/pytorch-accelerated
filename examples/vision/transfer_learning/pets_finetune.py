# Modifications Copyright (C) 2021 Chris Hughes
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
########################################################################
# This is an accelerated example of training a ResNet50 on the Oxford-IIT Pet Dataset and
# was adapted from an example produced by HuggingFace available here:
# https://github.com/huggingface/accelerate/blob/main/examples/cv_example.py
#
# Note: this example requires installing the torchvision, torchmetrics and timm packages
########################################################################
import os
from pathlib import Path
import re
from functools import partial

import numpy as np
import PIL
import torch
import torchmetrics
from func_to_script import script
from timm import create_model
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from pytorch_accelerated.callbacks import TrainerCallback
from pytorch_accelerated.finetuning import ModelFreezer
from pytorch_accelerated.trainer import (
    DEFAULT_CALLBACKS,
    Trainer,
    TrainerPlaceholderValues,
)


class ClassificationMetricsCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.metrics = torchmetrics.MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": torchmetrics.Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
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


def extract_label(fname):
    # Function to get the label from the filename
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]


class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return image, label


def create_datasets(
    file_names, label_to_id, image_size, normalization_mean, normalization_std
):
    random_perm = np.random.permutation(len(file_names))
    cut = int(0.8 * len(file_names))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training we use a simple RandomResizedCrop
    train_tfm = Compose(
        [
            RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            ToTensor(),
            Normalize(normalization_mean, normalization_std),
        ]
    )
    train_dataset = PetsDataset(
        [file_names[i] for i in train_split],
        image_transform=train_tfm,
        label_to_id=label_to_id,
    )

    # For evaluation, we use a deterministic Resize
    eval_tfm = Compose(
        [
            Resize(image_size),
            ToTensor(),
            Normalize(normalization_mean, normalization_std),
        ]
    )
    eval_dataset = PetsDataset(
        [file_names[i] for i in eval_split],
        image_transform=eval_tfm,
        label_to_id=label_to_id,
    )

    return train_dataset, eval_dataset


DATA_PATH = Path("/".join(Path(__file__).absolute().parts[:-3])) / "data/pets/images"


@script
def main(
    data_dir: str = DATA_PATH,
    lr: float = 3e-2,
    batch_size: int = 64,
    image_size: int = 224,
):
    image_size = (image_size, image_size)

    # Grab all the image filenames
    file_names = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".jpg")
    ]

    # Build the label correspondences
    all_labels = [extract_label(fname) for fname in file_names]
    id_to_label = list(set(all_labels))
    id_to_label.sort()
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}

    # Instantiate the model
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    num_in_features = model.get_classifier().in_features
    model.fc = nn.Sequential(
        nn.Linear(num_in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(label_to_id)),
    )

    # Freeze model backbone
    freezer = ModelFreezer(model, freeze_batch_norms=False)
    freezer.freeze()

    model_config = model.default_cfg
    normalization_mean = model_config["mean"]
    normalization_std = model_config["std"]

    train_dataset, eval_dataset = create_datasets(
        file_names, label_to_id, image_size, normalization_mean, normalization_std
    )

    # Define a loss function
    loss_func = torch.nn.functional.cross_entropy

    # Instantiate optimizer and scheduler type
    # Only pass unfrozen parameters to optimizer
    optimizer = torch.optim.Adam(
        params=freezer.get_trainable_parameters(),
        lr=lr / 25,
    )

    lr_scheduler = partial(
        OneCycleLR,
        max_lr=lr,
        epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
    )

    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=[
            ClassificationMetricsCallback(num_classes=len(id_to_label)),
            *DEFAULT_CALLBACKS,
        ],
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=2,
        per_device_batch_size=batch_size,
        create_scheduler_fn=lr_scheduler,
    )

    # Unfreeze backbone and pass parameters to optimizer
    param_groups = freezer.unfreeze()

    for idx, param_group in param_groups.items():
        param_group["lr"] = lr / 1000
        optimizer.add_param_group(param_group)

    lr_scheduler = partial(
        OneCycleLR,
        max_lr=lr / 100,
        epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        steps_per_epoch=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=6,
        per_device_batch_size=batch_size,
        create_scheduler_fn=lr_scheduler,
        reset_run_history=False,
    )


if __name__ == "__main__":
    main()
