from unittest.mock import Mock
import pytest

import torch
from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import LimitBatchesCallback
from pytorch_accelerated.utils import (
    LimitBatches,
    local_process_zero_first,
    local_process_zero_only,
    remove_padding,
    world_process_zero_only,
)

PAD_VALUE = -1


def test_can_limit_batches_with_manager():
    limit_batches_num = 4
    with LimitBatches(limit_batches_num):
        limited_trainer = Trainer(model=Mock(), loss_func=Mock(), optimizer=Mock())
    callback = limited_trainer.callback_handler.callbacks[0]

    assert isinstance(callback, LimitBatchesCallback)
    assert callback.num_batches == limit_batches_num


def test_limit_batches_not_present_outside_manager():
    with LimitBatches(2):
        pass

    trainer = Trainer(model=Mock(), loss_func=Mock(), optimizer=Mock())

    assert LimitBatchesCallback not in {
        callback.__class__ for callback in trainer.callback_handler.callbacks
    }


def test_can_remove_padding_1d():
    t = torch.tensor([0, 1, 2, 3, PAD_VALUE, PAD_VALUE])
    expected_t = torch.tensor([0, 1, 2, 3])

    actual_t = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual_t).all()


def test_can_remove_padding_2d():
    t = torch.tensor([[0, 1], [1, 2], [PAD_VALUE, PAD_VALUE], [PAD_VALUE, PAD_VALUE]])

    expected_t = torch.tensor([[0, 1], [1, 2]])

    actual_t = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual_t).all()


def test_can_remove_padding_3d():
    t = torch.tensor(
        [[[0, 1], [1, 2]], [[PAD_VALUE, PAD_VALUE], [PAD_VALUE, PAD_VALUE]]]
    )

    expected_t = torch.tensor([[[0, 1], [1, 2]]])

    actual_t = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual_t).all()


@pytest.mark.parametrize("shape", [(3, 4), (3, 4, 2), (3, 4, 2, 5)])
def test_can_remove_padding(shape):
    batch_size, *_ = shape

    t = torch.ones(*shape)
    t[-1, ...] = PAD_VALUE

    expected_t = torch.ones(batch_size - 1, *shape[1:])

    actual_t = remove_padding(t, PAD_VALUE)

    assert torch.eq(expected_t, actual_t).all()


@pytest.mark.parametrize(
    "process_decorator",
    [world_process_zero_only, local_process_zero_only, local_process_zero_first],
)
def test_process_decorators_are_transparent_without_multi_gpu(process_decorator):
    @process_decorator
    def fun():
        return 42

    result = fun()

    assert result == 42
