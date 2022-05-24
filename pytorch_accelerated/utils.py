from functools import wraps

from accelerate.state import AcceleratorState
from accelerate.utils import wait_for_everyone


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
