from unittest.mock import Mock

from pytorch_accelerated import Trainer
from pytorch_accelerated.callbacks import LimitBatchesCallback
from pytorch_accelerated.utils import LimitBatches


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
