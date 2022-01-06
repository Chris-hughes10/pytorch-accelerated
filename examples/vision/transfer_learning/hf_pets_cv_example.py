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

import argparse
import os
import re
from functools import partial

import PIL
import numpy as np
import torch
from timm import create_model
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from pytorch_accelerated.callbacks import (
    TerminateOnNaNCallback,
    LogMetricsCallback,
    PrintProgressCallback,
    ProgressBarCallback,
)
from pytorch_accelerated.trainer import Trainer


class PetsTrainer(Trainer):
    def training_run_start(self):
        config = self._accelerator.unwrap_model(self.model).default_cfg

        self.mean = torch.tensor(config["mean"])[None, :, None, None].to(self.device)
        self.std = torch.tensor(config["std"])[None, :, None, None].to(self.device)

    def calculate_train_batch_loss(self, batch):
        inputs = (batch["image"] - self.mean) / self.std

        return super().calculate_train_batch_loss((inputs, batch["label"]))

    def eval_epoch_start(self):
        super().eval_epoch_start()
        self.accurate = 0
        self.num_elems = 0

    def calculate_eval_batch_loss(self, batch):
        inputs = (batch["image"] - self.mean) / self.std

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, batch["label"])
        predictions = outputs.argmax(dim=-1)
        accurate_preds = self.gather(predictions) == self.gather(batch["label"])
        self.num_elems += accurate_preds.shape[0]
        self.accurate += accurate_preds.long().sum()

        return {
            "loss": loss,
            "model_outputs": outputs,
            "batch_size": inputs.size(0),
        }

    def eval_epoch_end(self):
        super().eval_epoch_end()
        self.run_history.update_metric(
            "accuracy", round(100 * self.accurate.item() / self.num_elems, 2)
        )

    def create_scheduler(self):
        return self.create_scheduler_fn(
            optimizer=self.optimizer, steps_per_epoch=len(self._train_dataloader)
        )


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
        return {"image": image, "label": label}


def training_function(data_dir, config):

    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    image_size = config["image_size"]
    if not isinstance(image_size, (list, tuple)):
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

    # Set the seed before splitting the data.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Split our filenames between train and validation
    random_perm = np.random.permutation(len(file_names))
    cut = int(0.8 * len(file_names))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # For training we use a simple RandomResizedCrop
    train_tfm = Compose([RandomResizedCrop(image_size, scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset(
        [file_names[i] for i in train_split],
        image_transform=train_tfm,
        label_to_id=label_to_id,
    )

    # For evaluation, we use a deterministic Resize
    eval_tfm = Compose([Resize(image_size), ToTensor()])
    eval_dataset = PetsDataset(
        [file_names[i] for i in eval_split],
        image_transform=eval_tfm,
        label_to_id=label_to_id,
    )

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # Define a loss function
    loss_func = torch.nn.functional.cross_entropy

    # Instantiate optimizer and scheduler type
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr / 25)
    lr_scheduler = partial(
        OneCycleLR,
        max_lr=lr,
        epochs=num_epochs,
    )

    trainer = PetsTrainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        callbacks=(
            TerminateOnNaNCallback,
            PrintProgressCallback,
            ProgressBarCallback,
            LogMetricsCallback,
        ),
    )

    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
        create_scheduler_fn=lr_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple examples of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    config = {
        "lr": 3e-2,
        "num_epochs": 3,
        "seed": 42,
        "batch_size": 64,
        "image_size": 224,
    }

    training_function(args.data_dir, config)
