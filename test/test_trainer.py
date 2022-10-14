from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, Mock, call

import torch
from pytest import fixture
from pytorch_accelerated.utils import worker_init_fn
from torch import optim, nn

from pytorch_accelerated.trainer import Trainer


class DummyTrainer(Trainer):
    def __init__(
        self,
        train_dl_mock=MagicMock(),
        eval_dl_mock=MagicMock,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_dl_mock = train_dl_mock
        self.eval_dl_mock = eval_dl_mock

    def calculate_train_batch_loss(self, batch):
        return {
            "loss": torch.tensor(1.0),
            "model_outputs": MagicMock(),
            "batch_size": 1,
        }

    def create_train_dataloader(self, batch_size, train_dl_kwargs):
        return self.train_dl_mock

    def create_eval_dataloader(self, batch_size, eval_dl_kwargs):
        return self.eval_dl_mock

    def calculate_eval_batch_loss(self, batch):
        return {
            "loss": torch.tensor(1.0),
            "model_outputs": MagicMock(),
            "batch_size": 1,
        }

    def _prepare_model_and_optimizer(self):
        pass


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


@fixture
def model():
    return SimpleModel(10, 1)


@fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


def test_skip_eval_if_not_present(mocker):
    trainer = DummyTrainer(model=Mock(), optimizer=Mock(), loss_func=Mock())
    mocked_train_epoch: MagicMock = mocker.patch.object(trainer, "_run_train_epoch")
    mocked_eval_epoch = mocker.patch.object(trainer, "_run_eval_epoch")

    trainer.train(train_dataset=Mock(), num_epochs=2)

    mocked_train_epoch.assert_has_calls(
        [call(trainer._train_dataloader), call(trainer._train_dataloader)]
    )
    mocked_eval_epoch.assert_not_called()


def test_skip_scheduler_step_if_not_present(mocker):
    trainer = DummyTrainer(model=Mock(), optimizer=Mock(), loss_func=Mock())
    scheduler_step = mocker.patch.object(trainer, "scheduler_step")

    trainer.train(train_dataset=Mock(), num_epochs=1)

    scheduler_step.assert_not_called()


def test_can_override_train_dataloader_kwargs(mocker):
    trainer = Trainer(model=Mock, optimizer=Mock(), loss_func=Mock())
    dl_constructor: MagicMock = mocker.patch("pytorch_accelerated.trainer.DataLoader")
    train_dataset = Mock()
    collate_fn = Mock()
    trainer.train_dataset = train_dataset
    trainer.collate_fn = collate_fn
    override_dl_kwargs = {"batch_size": 100, "pin_memory": False, "num_workers": 0}
    expected_dl_kwargs = {
        "shuffle": True,
        "batch_size": 100,
        "pin_memory": False,
        "num_workers": 0,
        "worker_init_fn": worker_init_fn,
    }

    trainer.create_train_dataloader(10, override_dl_kwargs)

    dl_constructor.assert_called_with(
        dataset=train_dataset, collate_fn=collate_fn, **expected_dl_kwargs
    )


def test_can_override_eval_dataloader_kwargs(mocker):
    trainer = Trainer(model=Mock, optimizer=Mock(), loss_func=Mock())
    dl_constructor: MagicMock = mocker.patch("pytorch_accelerated.trainer.DataLoader")
    eval_dataset = Mock()
    collate_fn = Mock()
    trainer.eval_dataset = eval_dataset
    trainer.collate_fn = collate_fn
    override_dl_kwargs = {"batch_size": 100, "pin_memory": False, "num_workers": 0}
    expected_dl_kwargs = {
        "shuffle": False,
        "batch_size": 100,
        "pin_memory": False,
        "num_workers": 0,
        "worker_init_fn": worker_init_fn,
    }

    trainer.create_eval_dataloader(10, override_dl_kwargs)

    dl_constructor.assert_called_with(
        dataset=eval_dataset, collate_fn=collate_fn, **expected_dl_kwargs
    )


def test_model_is_in_correct_mode():
    pass


def test_gradient_accumulation():
    pass


def test_can_create_scheduler():
    pass


def test_can_reset_run_history():
    pass
