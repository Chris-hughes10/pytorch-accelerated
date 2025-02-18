from pathlib import Path
from tempfile import TemporaryFile, TemporaryDirectory
from unittest.mock import MagicMock, Mock, call
import pytest

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


class DummyDataloader:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


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
    mocked_train_epoch: MagicMock = mocker.patch.object(
        trainer, "_run_train_epoch", return_value=False
    )
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


def test_check_eval_batch_size_is_transparent_on_single_process():
    batch_size = 8
    n_samples = batch_size - 1

    class FakeRunConfig:
        eval_total_batch_size = batch_size
        is_distributed = False

    trainer = Trainer("irrelevant", "irrelevant", "irrelevant")
    trainer.eval_dataset = list(range(n_samples))
    trainer.run_config = FakeRunConfig()

    trainer._check_eval_batch_size()


def test_check_eval_batch_size_is_transparent_with_full_batches_for_all_processes():
    per_device_batch_size = 8
    n_processes = 4
    n_full_batches = 2

    n_samples_full_batches = per_device_batch_size * n_processes * n_full_batches
    # All batches are exactly full
    n_samples_last_batch = 0
    n_samples = n_samples_full_batches + n_samples_last_batch

    class FakeRunConfig:
        eval_per_device_batch_size = per_device_batch_size
        eval_total_batch_size = per_device_batch_size * n_processes
        num_processes = n_processes
        is_distributed = True

    trainer = Trainer("irrelevant", "irrelevant", "irrelevant")
    trainer.eval_dataset = list(range(n_samples))
    trainer.run_config = FakeRunConfig()

    with pytest.warns(None) as record:
        trainer._check_eval_batch_size()

    assert len(record) == 0


def test_check_eval_batch_size_raises_batch_size_bigger_than_dataset():
    batch_size = 8

    class FakeRunConfig:
        eval_total_batch_size = batch_size
        is_distributed = True

    trainer = Trainer("irrelevant", "irrelevant", "irrelevant")
    trainer.eval_dataset = list(range(batch_size - 1))
    trainer.run_config = FakeRunConfig()

    with pytest.raises(ValueError):
        trainer._check_eval_batch_size()


def test_check_eval_batch_size_raises_empty_node_on_last_batch():
    per_device_batch_size = 8
    n_processes = 4
    n_full_batches = 2

    n_samples_full_batches = per_device_batch_size * n_processes * n_full_batches
    # One process will have no samples, another one less samples than the batch size
    n_samples_last_batch = per_device_batch_size * (n_processes - 1) - 1
    n_samples = n_samples_full_batches + n_samples_last_batch

    class FakeRunConfig:
        eval_per_device_batch_size = per_device_batch_size
        eval_total_batch_size = per_device_batch_size * n_processes
        num_processes = n_processes
        is_distributed = True

    trainer = Trainer("irrelevant", "irrelevant", "irrelevant")
    trainer.eval_dataset = list(range(n_samples))
    trainer.run_config = FakeRunConfig()

    with pytest.warns(UserWarning):
        trainer._check_eval_batch_size()


def test_check_eval_batch_size_warns_padding_is_needed():
    per_device_batch_size = 8
    n_processes = 4
    n_full_batches = 2

    n_samples_full_batches = per_device_batch_size * n_processes * n_full_batches
    # One process will have just one sample
    n_samples_last_batch = per_device_batch_size * (n_processes - 1) + 1
    n_samples = n_samples_full_batches + n_samples_last_batch

    class FakeRunConfig:
        eval_per_device_batch_size = per_device_batch_size
        eval_total_batch_size = per_device_batch_size * n_processes
        num_processes = n_processes
        is_distributed = True

    trainer = Trainer("irrelevant", "irrelevant", "irrelevant")
    trainer.eval_dataset = list(range(n_samples))
    trainer.run_config = FakeRunConfig()

    with pytest.warns(UserWarning):
        trainer._check_eval_batch_size()

def test_run_config_per_process():
    # Simulate a scenario where each GPU gets 1250 samples,
    # resulting in 20 batches per epoch when using a per-device batch size of 64.
    # With 300 epochs, max_num_train_steps should be 300 * 20 = 6000.
    dummy_train_dl = DummyDataloader(20)
    trainer = Trainer(model=Mock(), optimizer=Mock(), loss_func=Mock())
    
    # Replace the accelerator with a dummy that has the necessary attributes.
    dummy_accelerator = Mock()
    dummy_accelerator.num_processes = 4
    dummy_accelerator.is_main_process = True
    dummy_accelerator.is_local_main_process = True
    dummy_accelerator.optimizer_step_was_skipped = False
    trainer._accelerator = dummy_accelerator
    
    # Manually assign the dummy dataloader.
    trainer._train_dataloader = dummy_train_dl
    trainer._train_dl_kwargs = {"batch_size": 64}
    
    # Create a run config without passing max_num_train_steps explicitly.
    run_config = trainer._create_run_config(
        per_device_batch_size=64,
        num_epochs=300,
        gradient_accumulation_steps=1,
        max_num_train_steps=None,
        gradient_clip_value=None,
    )
    
    # There are 20 update steps per epoch per process.
    assert run_config.num_update_steps_per_epoch == 20, f"Expected 20 update steps, got {run_config.num_update_steps_per_epoch}"
    # Total max updates per process should be 300 epochs * 20 = 6000.
    assert run_config.max_num_train_steps == 6000, f"Expected 6000 max steps, got {run_config.max_num_train_steps}"
    # The number of epochs should remain 300.
    assert run_config.num_epochs == 300, f"Expected 300 epochs, got {run_config.num_epochs}"