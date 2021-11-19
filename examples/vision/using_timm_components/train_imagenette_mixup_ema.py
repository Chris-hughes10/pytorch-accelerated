# Copyright © 2021 Chris Hughes
########################################################################
# This example trains a ResNet-RS50 on the Imagenette Dataset using components from the torchvision, timm and
# torchmetrics libraries.
# This example demonstrates how the trainer can be extended to incorporate techniques such as mixup and modelEMA
# into a training run.
#
# Note: this example requires installing the torchvision, torchmetrics and timm packages
# ONLY WORKS DISTRIBUTED due to sync batchnorm
########################################################################

import argparse
import math
from functools import partial
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config, Mixup, rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.models import create_model
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from torch import nn
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import transforms, InterpolationMode

from pytorch_accelerated.trainer import Trainer


def train_transforms(img_size, data_mean, data_std, rand_augment_config_str):
    if isinstance(img_size, (tuple, list)):
        img_size_min = min(img_size)
    else:
        img_size_min = img_size

    aa_params = dict(
        translate_const=int(img_size_min * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in data_mean]),
    )

    # Manually override the interpolation mode to prevent warning being printed every epoch
    rrc = RandomResizedCropAndInterpolation(img_size)
    rrc.interpolation = InterpolationMode.BILINEAR

    pre_tensor_transforms = [
        rrc,
        transforms.RandomHorizontalFlip(p=0.5),
        rand_augment_transform(rand_augment_config_str, aa_params),
    ]
    post_tensor_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(data_mean), std=torch.tensor(data_std)),
    ]

    return transforms.Compose([*pre_tensor_transforms, *post_tensor_transforms])


def eval_transforms(img_size, crop_pct, data_mean, data_std):
    scale_size = int(math.floor(img_size[0] / crop_pct))

    return transforms.Compose(
        [
            transforms.Resize(scale_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(data_mean), std=torch.tensor(data_std)
            ),
        ]
    )


# Taken from timm master branch - not yet released
class BinaryCrossEntropy(nn.Module):
    """BCE with optional one-hot from dense targets, label smoothing, thresholding
    NOTE for experiments comparing CE to BCE /w label smoothing, may remove
    """

    def __init__(
        self,
        smoothing=0.1,
        target_threshold=None,
        weight=None,
        reduction: str = "mean",
        pos_weight=None,
    ):
        super(BinaryCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1.0 - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (target.size()[0], num_classes),
                off_value,
                device=x.device,
                dtype=x.dtype,
            ).scatter_(1, target, on_value)
        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        return F.binary_cross_entropy_with_logits(
            x, target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction
        )


class MixupTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_fn, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = mixup_fn
        self.accuracy = Accuracy(num_classes=num_classes)
        self.ema_accuracy = Accuracy(num_classes=num_classes)
        self.ema_model = None

    def training_run_start(self):
        # Model EMA requires the model without a DDP wrapper and before sync batchnorm conversion
        self.ema_model = ModelEmaV2(
            self._accelerator.unwrap_model(self.model), decay=0.5
        )
        if self.run_config.is_distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def train_epoch_end(
        self,
    ):
        self.ema_model.update(self.model)
        self.ema_model.eval()

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            preds = outputs.argmax(-1)
            self.accuracy.update(preds, yb)

            ema_model_preds = self.ema_model.module(xb).argmax(-1)
            self.ema_accuracy.update(ema_model_preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()
        if self.scheduler is not None:
            # timm scheduler must be manually updated per epoch
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.run_history.update_metric(
            "ema_model_accuracy", self.ema_accuracy.compute().cpu()
        )
        self.accuracy.reset()
        self.ema_accuracy.reset()

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)


def main(data_path):
    # Set training arguments, hardcoded here for clarity
    model = "resnetrs50"
    pretrained = False
    drop_path = 0.05
    img_size = (224, 224)
    lr = 5e-3
    smoothing = 0.1
    mixup = 0.2
    cutmix = 1.0
    batch_size = 16
    bce_target_thresh = 0.2
    num_epochs = 10

    data_path = Path(data_path)
    train_path = data_path / "train"
    val_path = data_path / "val"
    num_classes = len(list(train_path.iterdir()))

    # Create model using timm
    model = create_model(
        model, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path
    )

    # Load data config associated with the model to use in data augmentation pipeline
    data_config = resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]

    # Create datasets using PyTorch factory function
    train_dataset = datasets.ImageFolder(
        train_path,
        train_transforms(
            img_size=img_size,
            data_mean=data_mean,
            data_std=data_std,
            rand_augment_config_str="rand-m7-mstd0.5-inc1",
        ),
    )

    eval_dataset = datasets.ImageFolder(
        val_path,
        eval_transforms(
            img_size=img_size, data_mean=data_mean, data_std=data_std, crop_pct=0.95
        ),
    )

    # Create mixup function
    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )
    mixup_fn = Mixup(**mixup_args)

    # Create PyTorch optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Create higher order function to create timm scheduler
    lr_scheduler_type = partial(timm.scheduler.CosineLRScheduler, t_initial=num_epochs)

    # As we are using mixup, we can use BCE during training and CE for evaluation
    train_loss_fn = BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    # Create trainer and start training
    trainer = MixupTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=train_loss_fn,
        eval_loss_fn=validate_loss_fn,
        mixup_fn=mixup_fn,
        num_classes=num_classes,
    )

    trainer.train(
        per_device_batch_size=batch_size,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        create_scheduler_fn=lr_scheduler_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--data_dir", required=True, help="The data folder on disk.")
    args = parser.parse_args()
    main(args.data_dir)
