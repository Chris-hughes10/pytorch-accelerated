# Modifications Copyright © 2022 Chris Hughes
# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/cosine_lr.py

import math
from functools import partial
from typing import Callable

import torch

from pytorch_accelerated import TrainerPlaceholderValues
from pytorch_accelerated.schedulers.scheduler_base import StatefulSchedulerBase


class CosineLrScheduler(StatefulSchedulerBase):
    """
    A stateful Cosine annealing learning rate scheduler, as described in `this paper <https://arxiv.org/abs/1608.03983>`_,
    but without restarts.

    This scheduler differs from the PyTorch's :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` as it provides options
    to add warmup and cooldown epochs. Additionally, the annealing rate can be modified by adjusting the k-decay
    parameter, for which the rate of change of the learning rate is changed by its k-th order derivative,
    as described in `here <https://arxiv.org/abs/2004.05909>`_.

    If warmup epochs are specified, the learning rate will increase in constant increments from the ``warmup_starting_lr``
    provided until the learning rate specified in the parameter group is reached.

    If cooldown epochs are specified, the learning rate will be fixed at the minimum lr value given. This behaviour will
    continue if the scheduler is called after the training cycle has completed.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_num_epochs: int,
        num_update_steps_per_epoch: int,
        k_decay=1.0,
        lr_min: float = 1e-6,
        min_lr_ratio=None,
        num_warmup_epochs: int = 0,
        warmup_starting_lr=1e-6,
        warmup_starting_lr_ratio=None,
        num_cooldown_epochs=0,
    ):
        """
        Create a new ConsineLrScheduler object which can be used to modify the learning rate in an optimizer's parameter
        groups.

        :param optimizer: a PyTorch optimizer containing one or more parameter groups
        :param total_num_epochs: the total number of training epochs, inclusive of any warmup and cooldown epochs
        :param num_update_steps_per_epoch: the number of optimizer updates that take place per epoch
        :param k_decay: adjusts the rate of annealing. Higher values will maintain a higher lr for longer
        :param lr_min: the minimum value that the learning rate should decay to for all parameter groups. This will be held fixed during cooldown epochs
        :param min_lr_ratio: this can be used to represent the minimum lr for each parameter group as a ratio of its maximum lr. If set, this will take precedence over ``lr_min``
        :param num_warmup_epochs: the number of epochs to gradually increase the lr until it reaches the maximum value
        :param warmup_starting_lr: the starting lr that will be used for all parameter groups at the beginning of training if ``num_warmup_epochs`` is greater than 0
        :param warmup_starting_lr_ratio: this can be used to represent the warmup starting lr for each parameter group as a ratio of its maximum lr. If set, this will take precedence over ``warmup_starting_lr``
        :param num_cooldown_epochs: the number of epochs to hold the lr at its minimum value
        """

        super().__init__(optimizer)
        assert total_num_epochs > 0 and num_update_steps_per_epoch > 0
        assert lr_min >= 0
        self.total_iterations = total_num_epochs * num_update_steps_per_epoch
        self.lr_min_ratio = min_lr_ratio
        self.lr_min = lr_min
        self.warmup_iterations = num_warmup_epochs * num_update_steps_per_epoch
        self.warmup_lr_init = warmup_starting_lr
        self.warmup_lr_ratio = warmup_starting_lr_ratio
        self.k_decay = k_decay
        self.num_cooldown_iterations = num_cooldown_epochs * num_update_steps_per_epoch
        if self.warmup_iterations:
            self._update_param_groups(self.warmup_lr_init)

    def get_updated_values(self, num_updates: int):
        """
        Calculate the learning rate for a particular step given the number of previous updates.

        If warmup epochs are specified, the learning rate will increase in constant increments from the ``warmup_starting_lr``
        provided until the learning rate specified in the parameter group is reached.

        If cooldown epochs are specified, the learning rate will be fixed at the minimum lr value given. This behaviour will
        continue if the scheduler is called after the training cycle has completed.

        Between any warmup or cooldown epochs, the cosine annealing strategy will be used.

        :param num_updates: the number of previous updates
        :return: the learning rates with which to update each parameter group
        """
        if num_updates < self.warmup_iterations:
            # increase lr linearly
            lrs = [
                (
                    self.warmup_lr_ratio * lr
                    if self.warmup_lr_ratio is not None
                    else self.warmup_lr_init
                )
                + num_updates * ((lr - self.warmup_lr_init) / self.warmup_iterations)
                for lr in self.base_lr_values
            ]
        elif num_updates < (
            self.total_iterations
            - (self.warmup_iterations + self.num_cooldown_iterations)
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
                        * num_updates**self.k_decay
                        / total_cosine_iterations**self.k_decay
                    )
                )
                for lr_max in self.base_lr_values
            ]

        else:
            # cooldown
            lrs = [
                (
                    self.lr_min_ratio * base_lr
                    if self.lr_min_ratio is not None
                    else self.lr_min
                )
                for base_lr in self.base_lr_values
            ]

        return lrs

    @classmethod
    def create_scheduler_fn(
        cls,
        total_num_epochs: int = TrainerPlaceholderValues.NUM_EPOCHS,
        num_update_steps_per_epoch: int = TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        k_decay=1.0,
        lr_min: float = 1e-6,
        min_lr_ratio=None,
        num_warmup_epochs: int = 0,
        warmup_starting_lr=1e-6,
        warmup_starting_lr_ratio=None,
        num_cooldown_epochs=0,
    ) -> Callable:
        """
        An alternative constructor which returns a function that accepts an optimizer and creates an instance of
        ``CosineLrScheduler``. This is primarily intended to be used with the :class:`~pytorch_accelerated.trainer.Trainer`
        as illustrated below::

            trainer.train(
            train_dataset=train_dataset,
            num_epochs=num_epochs,
            per_device_batch_size=batch_size,
            create_scheduler_fn=CosineLrScheduler.create_scheduler_fn(num_warmup_epochs=5),
            )

        By default, the ``total_num_epochs`` and ``num_iterations_per_epoch`` arguments will be set by the
        :class:`~pytorch_accelerated.trainer.Trainer` with the correct values at runtime.

        :param total_num_epochs: the total number of training epochs, inclusive of any warmup and cooldown epochs
        :param num_update_steps_per_epoch: the number of optimizer updates that take place per epoch
        :param k_decay: adjusts the rate of annealing. Higher values will maintain a higher lr for longer
        :param lr_min: the minimum value that the learning rate should decay to for all parameter groups. This will be held fixed during cooldown epochs
        :param min_lr_ratio: this can be used to represent the minimum lr for each parameter group as a ratio of its maximum lr. If set, this will take precedence over ``lr_min``
        :param num_warmup_epochs: the number of epochs to gradually increase the lr until it reaches the maximum value
        :param warmup_starting_lr: the starting lr that will be used for all parameter groups at the beginning of training if ``num_warmup_epochs`` is greater than 0
        :param warmup_starting_lr_ratio: this can be used to represent the warmup starting lr for each parameter group as a ratio of its maximum lr. If set, this will take precedence over ``warmup_starting_lr``
        :param num_cooldown_epochs: the number of epochs to hold the lr at its minimum value
        :return: a function which accepts an optimizer as an argument and returns an instance of :class:`CosineLrScheduler`
        """
        return partial(
            cls,
            total_num_epochs=total_num_epochs,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            lr_min=lr_min,
            min_lr_ratio=min_lr_ratio,
            k_decay=k_decay,
            num_warmup_epochs=num_warmup_epochs,
            warmup_starting_lr=warmup_starting_lr,
            warmup_starting_lr_ratio=warmup_starting_lr_ratio,
            num_cooldown_epochs=num_cooldown_epochs,
        )

    def state_dict(self):
        current_state = {
            "total_iterations": self.total_iterations,
            "lr_min_ratio": self.lr_min_ratio,
            "lr_min": self.lr_min,
            "warmup_iterations": self.warmup_iterations,
            "warmup_lr_init": self.warmup_lr_init,
            "warmup_lr_ratio": self.warmup_lr_ratio,
            "k_decay": self.k_decay,
            "num_cooldown_iterations": self.num_cooldown_iterations,
            "num_updates": self._num_updates,
        }

        return current_state

    def load_state_dict(self, state_dict: dict):
        self.total_iterations = state_dict["total_iterations"]
        self.lr_min_ratio = state_dict["lr_min_ratio"]
        self.lr_min = state_dict["lr_min"]
        self.warmup_iterations = state_dict["warmup_iterations"]
        self.warmup_lr_init = state_dict["warmup_lr_init"]
        self.warmup_lr_ratio = state_dict["warmup_lr_ratio"]
        self.k_decay = state_dict["k_decay"]
        self.num_cooldown_iterations = state_dict["num_cooldown_iterations"]
        self._num_updates = state_dict["num_updates"]
