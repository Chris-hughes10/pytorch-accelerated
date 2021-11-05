# label smoothing - timm

# drop path - timm

# dropout

# rand augment - timm

# optimizer SGDR (warm restarts) + Cosine scheduling
# or LAMB cosine scheduler

# architecture - resnet RS

# loss - BCE loss

# mixup and cutmix

# repeated augmentation + stochastic depth : tend to improve the results at convergence, but they slow
# down the training in the early stages

# sync batchnorm
import math
from functools import partial
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from accelerate import notebook_launcher
from timm.data import resolve_data_config, Mixup, rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.models import create_model
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from torch import nn
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import transforms

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

    pre_tensor_transforms = [
        RandomResizedCropAndInterpolation(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        rand_augment_transform(rand_augment_config_str, aa_params)
    ]
    post_tensor_transforms = [transforms.ToTensor(),
                              transforms.Normalize(
                                  mean=torch.tensor(data_mean),
                                  std=torch.tensor(data_std))]

    return transforms.Compose([*pre_tensor_transforms, *post_tensor_transforms])


def eval_transforms(img_size, crop_pct, data_mean, data_std):
    scale_size = int(math.floor(img_size[0] / crop_pct))

    return transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(data_mean),
            std=torch.tensor(data_std))
    ])


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


class TimmTrainer(Trainer):
    def __init__(self, eval_loss_fn, mixup_fn, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_loss_fn = eval_loss_fn
        self.num_updates = None
        self.mixup_fn = mixup_fn
        self.accuracy = Accuracy(num_classes=num_classes)
        self.ema_model = None

    def training_run_start(self):
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        self.ema_model = ModelEmaV2(self._accelerator.unwrap_model(self.model))

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)
        self.accuracy.to(self._eval_dataloader.device)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def train_epoch_end(
            self,
    ):
        self.ema_model.update(self.model)

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            xb, yb = batch
            outputs = self.model(xb)
            val_loss = self.eval_loss_fn(outputs, yb)
            preds = outputs.argmax(-1)

            self.accuracy.update(preds, yb)

        return {"loss": val_loss, "model_outputs": outputs, "batch_size": xb.size(0)}

    def eval_epoch_end(self):
        super().eval_epoch_end()
        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        self.run_history.update_metric("accuracy", self.accuracy.compute().cpu())
        self.accuracy.reset()

        print(f"lr: {self.optimizer.param_groups[0]['lr']}")

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)


def main():
    data_path = Path(r"/home/chris/notebooks/imagenette2/")
    data_path = Path(r"C:\Users\hughesc\OneDrive - Microsoft\Documents\toy_data\hymenoptera_data")
    train_path = data_path / 'train'
    val_path = data_path / 'val'

    num_classes = len(list(train_path.iterdir()))

    model = "resnetrs50"
    pretrained = True
    drop_path = 0.05

    model = create_model(
        model, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path
    )

    # Freezing the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    data_config = resolve_data_config({}, model=model, verbose=True)

    img_size = (224, 224)

    data_mean = data_config['mean']
    data_std = data_config['std']

    train_dataset = datasets.ImageFolder(train_path, train_transforms(img_size=img_size,
                                                                      data_mean=data_mean,
                                                                      data_std=data_std,
                                                                      rand_augment_config_str="rand-m7-mstd0.5-inc1"))

    eval_dataset = datasets.ImageFolder(val_path, eval_transforms(img_size=img_size,
                                                                  data_mean=data_mean,
                                                                  data_std=data_std,
                                                                  crop_pct=0.95))
    smoothing = 0.1
    mixup = 0.2
    cutmix = 1.0

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )

    mixup_fn = Mixup(**mixup_args)

    batch_size = 16
    lr = 5e-3

    bce_target_thresh = 0.2

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    num_epochs = 10

    # timm scheduler
    lr_scheduler_type = partial(timm.scheduler.CosineLRScheduler, t_initial=num_epochs)

    train_loss_fn = BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    trainer = TimmTrainer(
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
        scheduler_type=lr_scheduler_type,
    )


if __name__ == '__main__':
    notebook_launcher(main, num_processes=2)
