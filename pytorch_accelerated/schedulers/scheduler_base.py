# inspired by ideas from https://github.com/rwightman/pytorch-image-models/blob/master/timm/scheduler/scheduler.py

from abc import ABC, abstractmethod

import torch


class SchedulerBase(ABC):
    """Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, param_group_field: str = "lr"
    ) -> None:

        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        self._store_initial_lrs()
        self.base_lr_values = [
            group[self._initial_param_group_field]
            for group in self.optimizer.param_groups
        ]
        self.update_param_groups(self.base_lr_values)

    def _store_initial_lrs(self):
        for i, group in enumerate(self.optimizer.param_groups):
            if self.param_group_field not in group:
                raise KeyError(
                    f"{self.param_group_field} missing from param_groups[{i}]"
                )
            group.setdefault(
                self._initial_param_group_field, group[self.param_group_field]
            )

    @abstractmethod
    def get_updated_lrs(self, current_iteration_number: int):
        pass

    def step_update(self, current_iteration_number: int):
        """
        To be called after each optimizer step
        """
        values = self.get_updated_lrs(current_iteration_number)
        if values is not None:
            self.update_param_groups(values)

    def update_param_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class StatefulSchedulerBase(SchedulerBase, ABC):
    def __init__(self, optimizer):
        super().__init__(optimizer=optimizer)
        self.current_iteration_number = 0

    def step(self):

        self.step_update(self.current_iteration_number)
        self.current_iteration_number += 1
