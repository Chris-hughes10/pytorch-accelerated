# Copyright © 2021 Chris Hughes
from dataclasses import dataclass, asdict
from numbers import Number
from typing import Union


@dataclass(frozen=True)
class TrainerRunConfig:
    """
    An immutable dataclass holding values representing the current state of the :class:`~pytorch_accelerated.trainer.Trainer`

    :param num_epochs: the number of epochs in the current training run
    :param train_per_device_batch_size: the device size per batch used during training epochs
    :param train_dl_kwargs: the arguments that have been used to create the training dataloader
    :param eval_per_device_batch_size: the device size per batch used during evaluation epochs
    :param eval_dl_kwargs: the arguments that have been used to create the evaluation dataloader
    :param gradient_accumulation_steps: the number of gradient accumulation steps which will be used during training
    :param  gradient_clip_value: the value used to determine the threshold to clip the gradients of the model's parameters
    :param train_total_batch_size: the total batch size used during training
    :param eval_total_batch_size: the total batch size used during evaluation
    :param num_update_steps_per_epoch: the number of steps per training epoch where the model's parameters will be updated
    :param max_num_train_steps: the maximum number of steps to train for, if present, this will take precedence over ``num_epochs``
    :param is_local_process_zero: ``True`` if the current process is the main process on the current node, ``False`` otherwise
    :param is_world_process_zero: ``True`` if the current process is the main process across all nodes, ``False`` otherwise
    :param is_distributed: ``True`` if the trainer is set up to perform distributed training, ``False`` otherwise
    :param using_fp16: ``True`` if the trainer is set up to use fp16, ``False`` otherwise


    """

    num_epochs: int
    train_per_device_batch_size: int
    train_dl_kwargs: dict
    eval_per_device_batch_size: int
    eval_dl_kwargs: dict
    gradient_accumulation_steps: int
    gradient_clip_value: Union[Number, None]
    train_total_batch_size: int
    eval_total_batch_size: int
    num_update_steps_per_epoch: int
    max_num_train_steps: Union[int, None]
    is_local_process_zero: bool
    is_world_process_zero: bool
    is_distributed: bool
    using_fp16: bool

    def to_dict(self):
        return asdict(self)
