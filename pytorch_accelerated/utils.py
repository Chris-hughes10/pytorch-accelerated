# Modifications Copyright (C) 2021 Chris Hughes
# Model EMA code adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py

import itertools
import os
from copy import deepcopy
from functools import wraps

import torch
from accelerate.state import PartialState
from accelerate.utils import wait_for_everyone
from torch import nn, Tensor

LIMIT_BATCHES_ENV_VAR = "PT_ACC_LIMIT_BATCHES"


class LimitBatches:
    """
    A context manager which can be used to limit the batches used within a :class:`~pytorch_accelerated.trainer.Trainer`.
    Any Trainer initialised within this context manager will contain the :class:`~pytorch_accelerated.callbacks.LimitBatchesCallback`
    callback. To remove this behaviour, a new trainer must be created or this callback must be explicitly removed.

    This will be automatically applied by the trainer if the environment variable ``PT_ACC_LIMIT_BATCHES`` is set.
    """

    def __init__(self, num_batches: int):
        self.num_batches = num_batches

    def __enter__(self):
        os.environ[LIMIT_BATCHES_ENV_VAR] = str(self.num_batches)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del os.environ[LIMIT_BATCHES_ENV_VAR]


class DataLoaderSlice:
    """
    A class which can be used to slice a :class:`~torch.utils.data.DataLoader` to only return a certain number of batches.

    """

    def __init__(self, dl, slice_size):
        self.dl = dl
        self.slice_size = slice_size

    def __iter__(self):
        return itertools.islice(self.dl, self.slice_size)

    def __len__(self):
        return self.slice_size


def local_process_zero_only(func):
    """
    A decorator which can be used to ensure that the decorated function is only executed on the local main process
    during distributed training

    :param func: the function to be decorated
    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        state = PartialState()
        if state.local_process_index == 0:
            result = func(*args, **kwargs)
            wait_for_everyone()
            return result
        else:
            wait_for_everyone()

    return wrapper_func


def local_process_zero_first(func):
    """
    A decorator which can be used to ensure that the decorated function is executed on the local main process first
    during distributed training

    :param func: the function to be decorated
    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        result = None
        state = PartialState()
        if state.local_process_index == 0:
            result = func(*args, **kwargs)
        wait_for_everyone()
        if state.local_process_index != 0:
            result = func(*args, **kwargs)
        return result

    return wrapper_func


def world_process_zero_only(func):
    """
    A decorator which can be used to ensure that the decorated function is only executed on the global main process
    during distributed training

    :param func: the function to be decorated

    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        state = PartialState()
        if state.process_index == 0:
            result = func(*args, **kwargs)
            wait_for_everyone()
            return result
        else:
            wait_for_everyone()

    return wrapper_func


class ModelEma(nn.Module):
    """
    Maintains a moving average of everything in the model state_dict (parameters and buffers), based on the ideas
    from https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.

    This class maintains a copy of the model that we are training. However,
    rather than updating all of the parameters of this model after every update step,
    we set these parameters using a linear combination of the existing parameter values and the updated values

    .. Note:: It is important to note that this class is sensitive to where it is initialised.
        During distributed training, it should be applied before before the conversion to :class:`~torch.nn.SyncBatchNorm`
        takes place and before the :class:`torch.nn.parallel.DistributedDataParallel` wrapper is used!
    """

    def __init__(self, model, decay=0.9999):
        """
        Create an instance of :class:`torch.nn.Module` to maintain an exponential moving average of the weights of
        the given model.

        This is done using the following formula:

        `updated_EMA_model_weights = decay * EMA_model_weights + (1. — decay) * updated_model_weights`

        where the decay is a parameter that we set.

        :param model: the subclass of :class: `torch.nn.Module` that we are training. This is the model that will be updated in our training loop as normal.
        :param decay: the amount of decay to use, which determines how much of the previous state will be maintained. The TensorFlow documentation suggests that reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999

        """
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        self.decay = decay

    def update_fn(self, ema_model_weights, updated_model_weights):
        return (
            self.decay * ema_model_weights + (1.0 - self.decay) * updated_model_weights
        )

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                updated_v = update_fn(ema_v, model_v)
                ema_v.copy_(updated_v)

    def update(self, model):
        self._update(model, update_fn=self.update_fn)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def worker_init_fn(worker_id):
    """
    Function that is used to set the random seed in each dataloader worker.
    This differs from the default by using the current random seed, which should be different in each process,
    rather than the initial random seed.
    """
    return torch.seed() + worker_id


def remove_padding(padded_tensor: Tensor, pad_value):
    """
    A utility function to remove padding tokens from a tensor. This can be useful after applying padding
    in order to gather a tensor.

    :param padded_tensor: the tensor containing padding
    :param pad_value: the padding token to be removed
    :return: a new tensor with padding tokens removed
    """
    padding_mask = padded_tensor == pad_value

    while padding_mask.ndim > 1:
        padding_mask = torch.all(padding_mask, dim=-1)

    result = padded_tensor[~padding_mask, ...]

    return result
