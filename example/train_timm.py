from pathlib import Path

import torch
from accelerate import notebook_launcher

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler, StepLRScheduler

from pytorch_thunder.trainer import Trainer


class TimmTrainer(Trainer):

    def __init__(self, eval_loss_fn, *args, **kwargs):
        self.eval_loss_fn = eval_loss_fn
        super().__init__(*args, **kwargs)


    def create_train_dataloader(self, shuffle=True, batch_size=8, **kwargs):
        args_scale = [0.08, 1.0]  # Random resize scale
        args_ratio = [3. / 4., 4. / 3.]  # Random resize aspect ratio
        args_hflip = 0.5
        args_vflip = 0.
        args_color_jitter = 0.4
        args_aug_repeats = 0
        args_aug_splits = 0

        args_aa = 'rand-m9-mstd0.5'  # auto augment policy
        args_remode = 'pixel'  # random erase mode
        args_reprob = 0.2

        data_config = resolve_data_config({}, model=self.model, verbose=True)
        train_interpolation = data_config['interpolation']

        return create_loader(
            self.train_dataset,
            input_size=data_config['input_size'],
            batch_size=batch_size,
            is_training=True,
            re_prob=args_reprob,
            re_mode=args_remode,
            scale=args_scale,
            ratio=args_ratio,
            hflip=args_hflip,
            vflip=args_vflip,
            color_jitter=args_color_jitter,
            auto_augment=args_aa,
            num_aug_splits=0,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=8,
            distributed=False,
            collate_fn=None,
            pin_memory=True,
            use_prefetcher=False
        )

    def create_eval_dataloader(self, shuffle=False, batch_size=8, **kwargs):
        data_config = resolve_data_config({}, model=self.model, verbose=True)

        return create_loader(
            self.eval_dataset,
            input_size=data_config['input_size'],
            batch_size=batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=8,
            distributed=False,
            crop_pct=data_config['crop_pct'],
            pin_memory=True,
            use_prefetcher=False
        )

    def calculate_eval_batch_step(self, batch):
        with torch.no_grad():
            xb, yb = batch
            val_loss = self.eval_loss_fn(self.model(xb), yb)

        return {
            "loss": val_loss,
        }


def main():
    data_path = Path('/home/chris/notebooks/imagenette2')
    train_path = data_path / 'train'
    val_path = data_path / 'val'

    # set arguments - TODO cleanup
    args_model = 'efficientnet_b0'
    args_num_classes = len(list(train_path.iterdir()))
    args_pretrained = True
    args_drop = 0.0
    args_drop_path = None
    args_drop_block = None
    args_bn_momentum = None
    args_bn_eps = None

    args_train_split = 'train'
    args_val_split = 'val'

    args_batch_size = 8
    args_sched = 'step'
    args_epochs = 20
    args_decay_epochs = 2.4
    args_decay_rate = 0.97
    args_opt = 'rmsproptf'
    args_opt_eps = .001
    args_workers = 8
    args_warmup_lr = 1e-6
    args_weight_decay = 1e-5
    args_drop = 0.2
    args_drop_connect = 0.2

    args_lr = 0.001



    model = create_model(
        args_model,
        pretrained=args_pretrained,
        num_classes=args_num_classes,
        bn_momentum=args_bn_momentum,
        bn_eps=args_bn_eps,
    )

    data_config = resolve_data_config({}, model=model, verbose=True)
    train_interpolation = data_config['interpolation']

    optimizer = create_optimizer_v2(model, args_opt, args_lr,
                                    args_weight_decay, args_opt_eps
                                    )

    lr_scheduler = StepLRScheduler(
        optimizer,
        decay_t=args_decay_epochs,
        decay_rate=args_decay_rate,
        warmup_lr_init=args_warmup_lr,
    )

    num_epochs = args_epochs

    dataset_train = create_dataset(
        'imagenette',
        root=data_path, split=args_train_split, is_training=True,
        batch_size=args_batch_size)
    dataset_eval = create_dataset(
        'imagenette', root=data_path, split=args_val_split, is_training=False, batch_size=args_batch_size)

    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    trainer = TimmTrainer(model=model, optimizer=optimizer, loss_func=train_loss_fn, eval_loss_fn=validate_loss_fn)

    trainer.train(train_dataset=dataset_train, eval_dataset=dataset_eval, num_epochs=10)


if __name__ == '__main__':
    notebook_launcher(main, num_processes=2)