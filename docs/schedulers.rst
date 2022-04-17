.. _schedulers:

Schedulers
***********

PyTorch-accelerated provides some scheduler implementations which can be used in any PyTorch training loop. However,
unlike PyTorch’s native schedulers - which can be called at different points in the training loop - all
Pytorch-accelerated schedulers expect to be called after **each optimizer update**.


Implemented Schedulers
=======================
.. autoclass:: pytorch_accelerated.schedulers.cosine_scheduler.CosineLrScheduler
    :show-inheritance:
    :members:

    .. automethod:: __init__


Base Schedulers
=======================
PyTorch-accelerated provides base classes for two types of schedulers.


Stateful Schedulers
-------------------------

Stateful schedulers maintain an internal count corresponding to how many times the scheduler's
:meth:`~pytorch_accelerated.schedulers.scheduler_base.StatefulSchedulerBase.step` method has beeen called.
As these schedulers have the same interface as the native PyTorch schedulers, these are supported by the
:class:`~pytorch_accelerated.trainer.Trainer` by default.

.. autoclass:: pytorch_accelerated.schedulers.scheduler_base.StatefulSchedulerBase
    :members:

    .. automethod:: __init__


Stateless Schedulers
-------------------------

These schedulers maintain no internal state about the current training run, and therefore require that the current
number of updates is explicitly provided when called. To use a stateless scheduler with the
:class:`~pytorch_accelerated.trainer.Trainer`, this would require subclassing the
:class:`~pytorch_accelerated.trainer.Trainer` and overriding the
:meth:`~pytorch_accelerated.trainer.Trainer.scheduler_step` method.


.. autoclass:: pytorch_accelerated.schedulers.scheduler_base.SchedulerBase
    :members:

    .. automethod:: __init__


Creating New Schedulers
========================

Whilst schedulers are usually used to schedule learning rates, the scheduler base classes in PyTorch-accelerated can be
used to schedule any parameter in an optimizer's parameter group.

To create a new scheduler, in most cases, all that is required is to subclass one of the base classes and override the
:meth:`~pytorch_accelerated.schedulers.scheduler_base.SchedulerBase.get_updated_values` method.

Example: Creating a simple milestone lr scheduler
---------------------------------------------------
Here is an example of how we can implement a scheduler to adjust the learning rate for each parameter group
by a factor ``gamma`` each time an epoch milestone is reached::

    from pytorch_accelerated.schedulers import StatefulSchedulerBase

    class MilestoneLrScheduler(StatefulSchedulerBase):
        def __init__(
            self, optimizer, gamma=0.5, epoch_milestones=(2, 4, 5), num_steps_per_epoch=100
        ):
            super().__init__(optimizer, param_group_field="lr")
            self.milestones = set(
                (num_steps_per_epoch * milestone for milestone in epoch_milestones)
            )
            self.gamma = gamma

        def get_updated_values(self, num_updates: int):
            if num_updates in self.milestones:
                lr_values = [
                    group[self.param_group_field] for group in self.optimizer.param_groups
                ]
                updated_lrs = [lr * self.gamma for lr in lr_values]
                return updated_lrs


Example: Scheduling weight decay
---------------------------------------------------

Here is an example of how we can define a scheduler to incrementally increase the amount of weight decay
by a factor ``gamma`` every ``n`` steps::

    from pytorch_accelerated.schedulers import StatefulSchedulerBase

    class StepWdScheduler(StatefulSchedulerBase):
        def __init__(self, optimizer, n=1000, gamma=1.1):
            super().__init__(optimizer, param_group_field="weight_decay")
            self.n = n
            self.gamma = gamma

        def get_updated_values(self, num_updates: int):
            if num_updates % self.n == 0 and num_updates > 0:
                wd_values = [
                    group[self.param_group_field] for group in self.optimizer.param_groups
                ]
                updated_wd_values = [wd * self.gamma for wd in wd_values]
                return updated_wd_values
