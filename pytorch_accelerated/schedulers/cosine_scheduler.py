# inspired by ideas from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/cosine_lr.py

import math
from functools import partial

import torch

from pytorch_accelerated import TrainerPlaceholderValues
from pytorch_accelerated.schedulers.scheduler_base import StatefulSchedulerBase


class CosineLrScheduler(StatefulSchedulerBase):
    """
    Cosine decay. without restarts
    This is described in the paper https://arxiv.org/abs/1608.03983.

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909, explain
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_num_epochs: int,
        num_update_steps_per_epoch: int,
        k_decay=1.0,
        lr_min: float = 0.0,
        min_lr_ratio=None,
        num_warmup_epochs: int = 0,
        warmup_starting_lr=0,
        num_cooldown_epochs=0,
    ):
        """

        :param optimizer:
        :param total_num_epochs:
        :param num_update_steps_per_epoch:
        :param k_decay:
        :param lr_min:
        :param min_lr_ratio:
        :param num_warmup_epochs:
        :param warmup_starting_lr:
        :param num_cooldown_epochs:
        """

        super().__init__(optimizer)
        assert total_num_epochs > 0 and num_update_steps_per_epoch > 0
        assert lr_min >= 0
        self.total_iterations = total_num_epochs * num_update_steps_per_epoch
        self.lr_min_ratio = min_lr_ratio
        self.lr_min = lr_min
        self.warmup_iterations = num_warmup_epochs * num_update_steps_per_epoch
        self.warmup_lr_init = warmup_starting_lr
        self.k_decay = k_decay
        self.num_cooldown_iterations = num_cooldown_epochs * num_update_steps_per_epoch
        if self.warmup_iterations:
            self._update_param_groups(self.warmup_lr_init)

    def get_updated_values(self, num_updates: int):
        if num_updates < self.warmup_iterations:
            # increase lr linearly
            lrs = [
                self.warmup_lr_init
                + num_updates * ((lr - self.warmup_lr_init) / self.warmup_iterations)
                for lr in self.base_lr_values
            ]
        elif (num_updates - self.warmup_iterations) < (
            self.total_iterations - self.num_cooldown_iterations
        ):
            num_updates = num_updates - self.warmup_iterations
            total_cosine_iterations = self.total_iterations - (
                self.warmup_iterations + self.num_cooldown_iterations
            )

            lrs = [
                (
                    self.lr_min_ratio * lr_max
                    if self.lr_min_ratio is not None
                    else self.lr_min
                )
                + 0.5
                * (
                    lr_max
                    - (
                        self.lr_min_ratio * lr_max
                        if self.lr_min_ratio is not None
                        else self.lr_min
                    )
                )
                * (
                    1
                    + math.cos(
                        math.pi
                        * num_updates ** self.k_decay
                        / total_cosine_iterations ** self.k_decay
                    )
                )
                for lr_max in self.base_lr_values
            ]

        else:
            # cooldown
            lrs = [
                self.lr_min_ratio * base_lr
                if self.lr_min_ratio is not None
                else self.lr_min
                for base_lr in self.base_lr_values
            ]

        return lrs

    @classmethod
    def create_scheduler(
        cls,
        total_num_epochs: int = TrainerPlaceholderValues.NUM_EPOCHS,
        num_iterations_per_epoch: int = TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        k_decay=1.0,
        lr_min: float = 0.0,
        min_lr_ratio=None,
        warmup_lr_init=0,
        num_warmup_epochs: int = 0,
    ):
        return partial(
            cls,
            total_num_epochs=total_num_epochs,
            num_iterations_per_epoch=num_iterations_per_epoch,
            lr_min=lr_min,
            min_lr_ratio=min_lr_ratio,
            k_decay=k_decay,
            num_warmup_epochs=num_warmup_epochs,
            warmup_lr_init=warmup_lr_init,
        )
