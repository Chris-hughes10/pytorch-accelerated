from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import notebook_launcher
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup
from timm.models import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch import nn
from torchmetrics import ConfusionMatrix

from pytorch_thunder.trainer import Trainer


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
        self.cm_metrics = ConfusionMatrix(num_classes=num_classes)
        self.cm_metrics_dist = ConfusionMatrix(
            num_classes=num_classes, dist_sync_on_step=True
        )

    def create_train_dataloader(self, **kwargs):

        kwargs.pop("shuffle")

        return create_loader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **kwargs
        )

    def create_eval_dataloader(self, **kwargs):

        kwargs.pop("shuffle")

        return create_loader(
            dataset=self.train_dataset, collate_fn=self.collate_fn, **kwargs
        )

    def train_epoch_start(self):
        super().train_epoch_start()
        self.num_updates = self.run_history.current_epoch * len(self._train_dataloader)

        self.cm_metrics.to(self._eval_dataloader.device)
        self.cm_metrics_dist.to(self._eval_dataloader.device)

    def calculate_train_batch_loss(self, batch):
        xb, yb = batch
        mixup_xb, mixup_yb = self.mixup_fn(xb, yb)
        return super().calculate_train_batch_loss((mixup_xb, mixup_yb))

    def calculate_eval_batch_step(self, batch):
        with torch.no_grad():
            xb, yb = batch
            preds = self.model(xb)
            val_loss = self.eval_loss_fn(preds, yb)

            self.cm_metrics_dist.update(preds, yb)
            self.cm_metrics.update(
                self._accelerator.gather(preds), self._accelerator.gather(yb)
            )

        return {
            "loss": val_loss,
        }

    def eval_epoch_end(self):
        super().eval_epoch_end()
        if self.scheduler is not None:
            self.scheduler.step(self.run_history.current_epoch + 1)

        cm = self.cm_metrics.compute()
        cm_dist = self.cm_metrics_dist.compute()

        print(f"Confusion matrix: {cm}")
        print(f"Confusion matrix dist: {cm_dist}")

    def scheduler_step(self):
        self.num_updates += 1
        if self.scheduler is not None:
            self.scheduler.step_update(num_updates=self.num_updates)


def main():
    # data_path = Path(r"C:\Users\hughesc\Documents\imagenette2-320\imagenette2-320")

    data_path = Path(r"/home/chris/notebooks/hymenoptera_data/")

    train_path = data_path / "train"
    val_path = data_path / "val"

    #####
    aa = "rand-m7-mstd0.5-inc1"
    # batch_size = 2048
    batch_size = 16
    # opt = 'lamb'
    opt = "adamp"
    lr = 5e-3
    smoothing = 0.1
    drop_path = 0.05
    aug_repeats = 3  # what does this do?
    hflip = 0.5
    mixup = 0.2
    cutmix = 1.0
    color_jitter = 0
    weight_decay = 0.02
    crop_pct = 0.95
    bce_loss = True
    bce_target_thresh = 0.2
    # model = "resnet-rs"
    model = "resnet50"
    num_classes = len(list(train_path.iterdir()))
    pretrained = True

    args_train_split = "train"
    args_val_split = "val"

    # augs hflip, random resized crop

    mixup_args = dict(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes,
    )

    mixup_fn = Mixup(**mixup_args)

    model = create_model(
        model, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path
    )

    data_config = resolve_data_config({}, model=model, verbose=True)
    train_interpolation = data_config["interpolation"]

    dataset_train = create_dataset(
        "imagenette",
        root=data_path,
        split=args_train_split,
        is_training=True,
        batch_size=batch_size,
    )
    dataset_eval = create_dataset(
        "imagenette",
        root=data_path,
        split=args_val_split,
        is_training=False,
        batch_size=batch_size,
    )

    train_dl_kwargs = {
        "input_size": data_config["input_size"],
        "is_training": True,
        "auto_augment": aa,
        # "num_aug_repeats": aug_repeats,
        "hflip": hflip,
        "color_jitter": color_jitter,
        "num_aug_splits": 0,
        "interpolation": train_interpolation,
        "mean": data_config["mean"],
        "std": data_config["std"],
        "num_workers": 1,
        "distributed": False,
        "use_prefetcher": False,
    }

    eval_dl_kwargs = {
        "input_size": data_config["input_size"],
        "is_training": False,
        "interpolation": data_config["interpolation"],
        "mean": data_config["mean"],
        "std": data_config["std"],
        "num_workers": 1,
        "distributed": False,
        # "crop_pct": data_config["crop_pct"],
        "crop_pct": crop_pct,
        "pin_memory": True,
        "use_prefetcher": False,
    }

    optimizer = create_optimizer_v2(
        model,
        opt,
        lr,
        weight_decay,
    )

    num_epochs = 10

    lr_scheduler_type = partial(CosineLRScheduler, t_initial=num_epochs)

    train_loss_fn = BinaryCrossEntropy(
        target_threshold=bce_target_thresh, smoothing=smoothing
    )
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    trainer = TimmTrainer(
        model=model,
        optimizer=optimizer,
        loss_func=train_loss_fn,
        eval_loss_fn=validate_loss_fn,
        scheduler_type=lr_scheduler_type,
        mixup_fn=mixup_fn,
        num_classes=num_classes,
    )

    trainer.train(
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        num_epochs=num_epochs,
        train_dataloader_kwargs=train_dl_kwargs,
        eval_dataloader_kwargs=eval_dl_kwargs,
    )


if __name__ == "__main__":
    notebook_launcher(main, num_processes=1)
