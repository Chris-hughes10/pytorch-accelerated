import torch

from pytorch_accelerated.schedulers.scheduler_base import (
    SchedulerBase,
    StatefulSchedulerBase,
)


class EqualIterationScheduler(SchedulerBase):
    def get_updated_values(self, num_updates: int):
        return num_updates


class StatefulEqualIterationScheduler(StatefulSchedulerBase):
    def get_updated_values(self, num_updates: int):
        return num_updates

    def state_dict(self):
        pass

    def load_state_dict(self, scheduler_state_dict):
        pass


def create_model_and_optimizer(lr_1, lr_2):
    model = torch.nn.Linear(2, 1)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        [
            {
                "params": params[0],
            },
            {"params": params[1], "lr": lr_2},
        ],
        lr=lr_1,
    )
    return model, optimizer


def test_stores_initial_lr_on_creation():
    lr_1 = 0.01
    lr_2 = 0.001
    expected_lrs = [lr_1, lr_2]
    model, optimizer = create_model_and_optimizer(lr_1, lr_2)

    scheduler = EqualIterationScheduler(optimizer)

    for expected_lr, param_group in zip(expected_lrs, optimizer.param_groups):
        assert "initial_lr" in param_group
        assert param_group["initial_lr"] == expected_lr


def test_can_update_parameter_groups():
    lr_1 = 0.01
    lr_2 = 0.001
    expected_initial_lrs = [lr_1, lr_2]
    model, optimizer = create_model_and_optimizer(lr_1, lr_2)

    scheduler = EqualIterationScheduler(optimizer)
    lrs_before_update = [param_group["lr"] for param_group in optimizer.param_groups]
    scheduler.step_update(1)
    lrs_after_update = [param_group["lr"] for param_group in optimizer.param_groups]

    assert lrs_before_update == expected_initial_lrs
    assert lrs_after_update == [1, 1]


def test_stateful_scheduler_can_update_parameter_groups():
    lr_1 = 0.01
    lr_2 = 0.001
    expected_initial_lrs = [lr_1, lr_2]
    model, optimizer = create_model_and_optimizer(lr_1, lr_2)

    scheduler = StatefulEqualIterationScheduler(optimizer)
    lrs_before_update = [param_group["lr"] for param_group in optimizer.param_groups]
    scheduler.step()
    lrs_after_update_1 = [param_group["lr"] for param_group in optimizer.param_groups]
    scheduler.step()
    lrs_after_update_2 = [param_group["lr"] for param_group in optimizer.param_groups]

    assert lrs_before_update == expected_initial_lrs
    assert lrs_after_update_1 == [0, 0]
    assert lrs_after_update_2 == [1, 1]
