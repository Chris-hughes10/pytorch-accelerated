import os
from functools import wraps

from accelerate.state import AcceleratorState
from accelerate.utils import wait_for_everyone

LIMIT_BATCHES_ENV_VAR = "PT_ACC_LIMIT_BATCHES"


class LimitBatches:
    """
    A context manager which can be used to limit the batches used within a :class:`~pytorch_accelerated.trainer.Trainer`.
    Any Trainer initialised within this context manager will contain the :class:`~pytorch_accelerated.callbacks.LimitBatchesCallback`
    callback. To remove this behaviour, a new trainer must be created or this callback must be explicitly removed.
    """

    def __init__(self, num_batches: int):
        self.num_batches = num_batches

    def __enter__(self):
        os.environ[LIMIT_BATCHES_ENV_VAR] = str(self.num_batches)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del os.environ[LIMIT_BATCHES_ENV_VAR]


def limit_trainer_batches(func, *, num_batches):
    """
    A decorator which can be used to ensure that the decorated function is only executed on the local main process
    during distributed training

    :param func: the function to be decorated
    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        os.environ[LIMIT_BATCHES_ENV_VAR] = num_batches
        result = func(*args, **kwargs)
        del os.environ[LIMIT_BATCHES_ENV_VAR]

        return result

    return wrapper_func


def local_process_zero_only(func):
    """
    A decorator which can be used to ensure that the decorated function is only executed on the local main process
    during distributed training

    :param func: the function to be decorated
    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        state = AcceleratorState(_from_accelerator=True)
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
        state = AcceleratorState(_from_accelerator=True)
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
        state = AcceleratorState(_from_accelerator=True)
        if state.process_index == 0:
            result = func(*args, **kwargs)
            wait_for_everyone()
            return result
        else:
            wait_for_everyone()

    return wrapper_func
