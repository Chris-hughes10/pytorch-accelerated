# Modifications Copyright © 2022 Chris Hughes
# SchedulerBase adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py

from abc import ABC, abstractmethod
from numbers import Number
from typing import Union, Iterable

import torch


class SchedulerBase(ABC):
    """
    A parameter scheduler base class that can be used to update any field within an optimizer's parameter groups.
    The most common use case for this is learning rate scheduling.

    Unlike PyTorch's schedulers, which can be called at different points in the training loop depending on the
    implementation, this class is intended to be consistently called at the end of each optimizer update.

    As this class is stateless by default, it expects that the number of updates is explicitly provided,
    as illustrated below::

        for current_epoch, epoch in enumerate(num_epochs):
            num_updates = current_epoch * num_update_steps_per_epoch
            for batch in train_dataloader:
                xb, yb = batch
                predictions = model(xb)
                loss = loss_func(predictions, yb)

                loss.backward()
                optimizer.step()

                num_updates +=1
                scheduler.step_update(num_updates)
    """

    def __init__(self, optimizer: torch.optim.Optimizer, param_group_field: str = "lr"):
        """
        Create a new instance of a parameter scheduler.

        :param optimizer: a PyTorch optimizer
        :param param_group_field: the field in the optimizer's parameter groups corresponding to the parameter to be scheduled
        """

        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        self._store_initial_lrs()
        self.base_lr_values = [
            group[self._initial_param_group_field]
            for group in self.optimizer.param_groups
        ]
        self._update_param_groups(self.base_lr_values)

    def _store_initial_lrs(self):
        """
        Store the initial value of the scheduled parameter for each parameter group.
        """
        for i, group in enumerate(self.optimizer.param_groups):
            if self.param_group_field not in group:
                raise KeyError(
                    f"{self.param_group_field} missing from param_groups[{i}]"
                )
            group.setdefault(
                self._initial_param_group_field, group[self.param_group_field]
            )

    @abstractmethod
    def get_updated_values(
        self, num_updates: int
    ) -> Union[None, Number, Iterable[Number]]:
        """
        Calculate updated values for the scheduled parameter.

        If a single value is returned, all parameter groups will be updated with this value.

        To update each parameter group with a different value, an iterable collection,
        containing an updated value for each parameter group, should be returned.

        If None is returned, the parameter groups will not be updated.

        :param num_updates: the number of optimizer updates
        :return: the updated values of the scheduled parameter. This should be either a single value, or an iterable collection containing a value for each parameter group.

        """
        pass

    def step_update(self, num_updates: int):
        """
        Calculate the updated value of the scheduled parameter and update the optimizer's parameter groups.

        :param num_updates: the number of optimizer updates

        """
        values = self.get_updated_values(num_updates)
        if values is not None:
            self._update_param_groups(values)

    def _update_param_groups(self, values):
        """
        Update the scheduled parameter with the given values in all of the optimizer's parameter groups.

        :param values: the updated values of the scheduled parameter
        """
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def state_dict(self):
        """
        Get the state dict for the scheduler, containing all attributes except the optimizer,
        which should be saved separately.

        :return: the scheduler's state dict
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """
        Updates the attributes of the given scheduler from the given state dict.

        :param state_dict: the state dict to be loaded
        """
        self.__dict__.update(state_dict)


class StatefulSchedulerBase(SchedulerBase, ABC):
    """
    A stateful parameter scheduler base class that can be used to update any field within an optimizer's parameter groups.
    The most common use case for this is learning rate scheduling.

    Unlike PyTorch's schedulers, which can be called at different points in the training loop depending on the
    implementation, this class is intended to be consistently called at the end of each optimizer update.

    This class is responsible for maintaining the number of updates, incrementing an internal count each time that
    the scheduler step is calculated.

    The usage of this class is illustrated below::

        for current_epoch, epoch in enumerate(num_epochs):
            for batch in train_dataloader:
                xb, yb = batch
                predictions = model(xb)
                loss = loss_func(predictions, yb)

                loss.backward()
                optimizer.step()

                scheduler.step()
    """

    def __init__(self, optimizer, param_group_field: str = "lr"):
        """
        Create a new instance of a stateful parameter scheduler.

        :param optimizer: a PyTorch optimizer
        :param param_group_field: the field in the optimizer's parameter groups corresponding to the parameter to be scheduled
        """
        super().__init__(optimizer=optimizer, param_group_field=param_group_field)
        self._num_updates = -1

    def step(self):
        """
        Calculate the updated value of the scheduled parameter and update the optimizer's parameter groups.
        """
        self._num_updates += 1
        self.step_update(self._num_updates)

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, scheduler_state_dict: dict):
        raise NotImplementedError
