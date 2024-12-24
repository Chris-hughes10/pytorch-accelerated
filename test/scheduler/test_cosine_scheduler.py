import torch

from pytorch_accelerated.schedulers.cosine_scheduler import CosineLrScheduler


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


def collect_lrs_for_scheduler(scheduler, num_epochs, num_steps_per_epoch):
    group_1_lrs = []
    group_2_lrs = []

    for epoch in range(num_epochs):
        for i in range(num_steps_per_epoch):
            scheduler.step()

            group_1_lrs.append(scheduler.optimizer.param_groups[0]["lr"])
            group_2_lrs.append(scheduler.optimizer.param_groups[1]["lr"])

    return group_1_lrs, group_2_lrs


def test_lr_maxes_equal():
    num_epochs = 300
    num_steps_per_epoch = 10
    lr_1_max = 0.01
    lr_2_max = 0.002
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch
    )

    assert group_1_lrs[0] == lr_1_max
    assert max(group_1_lrs) == lr_1_max
    assert group_2_lrs[0] == lr_2_max
    assert max(group_2_lrs) == lr_2_max


def test_lr_mins_at_schedule_end():
    num_epochs = 1
    num_steps_per_epoch = 10
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min = 0.0001
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        lr_min=lr_min,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch + 1
    )

    assert group_1_lrs[-1] == lr_min
    assert min(group_1_lrs) == lr_min
    assert group_2_lrs[-1] == lr_min
    assert min(group_2_lrs) == lr_min


def test_lr_min_ratio_at_schedule_end():
    num_epochs = 1
    num_steps_per_epoch = 10
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min_ratio = 0.5
    expected_lr_1_min = lr_min_ratio * lr_1_max
    expected_lr_2_min = lr_min_ratio * lr_2_max
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        min_lr_ratio=lr_min_ratio,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch + 1
    )

    assert group_1_lrs[-1] == expected_lr_1_min
    assert min(group_1_lrs) == expected_lr_1_min
    assert group_2_lrs[-1] == expected_lr_2_min
    assert min(group_2_lrs) == expected_lr_2_min


def test_cooldown_epochs_at_lr_min():
    num_epochs = 10
    num_cooldown_epochs = 2
    num_steps_per_epoch = 10
    num_cooldown_steps = num_cooldown_epochs * num_steps_per_epoch
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min = 1e-6
    expected_cooldown_steps = [lr_min] * num_cooldown_steps
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        lr_min=lr_min,
        num_cooldown_epochs=num_cooldown_epochs,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch
    )

    assert group_1_lrs[-num_cooldown_steps:] == expected_cooldown_steps
    assert group_2_lrs[-num_cooldown_steps:] == expected_cooldown_steps


def test_warmup():
    num_epochs = 10
    num_warmup_epochs = 2
    num_steps_per_epoch = 10
    num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
    lr_1_max = 0.01
    lr_2_max = 0.002
    warmup_lr = 1e-6
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        num_warmup_epochs=num_warmup_epochs,
        warmup_starting_lr=warmup_lr,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch
    )

    assert group_1_lrs[0] == warmup_lr
    assert group_2_lrs[0] == warmup_lr
    assert group_1_lrs[num_warmup_steps] == max(group_1_lrs)
    assert group_2_lrs[num_warmup_steps] == max(group_2_lrs)


def test_warmup_from_ratio():
    num_epochs = 10
    num_warmup_epochs = 2
    num_steps_per_epoch = 10
    num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
    lr_1_max = 0.01
    lr_2_max = 0.002
    warmup_lr_ratio = 0.01
    starting_lr_1 = warmup_lr_ratio * lr_1_max
    starting_lr_2 = warmup_lr_ratio * lr_2_max
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        num_warmup_epochs=num_warmup_epochs,
        warmup_starting_lr_ratio=warmup_lr_ratio,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch
    )

    assert group_1_lrs[0] == starting_lr_1
    assert group_2_lrs[0] == starting_lr_2
    assert group_1_lrs[num_warmup_steps] == max(group_1_lrs)
    assert group_2_lrs[num_warmup_steps] == max(group_2_lrs)


def test_warmup_and_cooldown():
    num_epochs = 10
    num_warmup_epochs = 2
    num_cooldown_epochs = 2

    num_steps_per_epoch = 10
    num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
    num_cooldown_steps = num_cooldown_epochs * num_steps_per_epoch
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min = 1e-6
    expected_cooldown_steps = [lr_min] * num_cooldown_steps
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        num_warmup_epochs=num_warmup_epochs,
        warmup_starting_lr=lr_min,
        lr_min=lr_min,
        num_cooldown_epochs=2,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(
        scheduler, num_epochs, num_steps_per_epoch
    )

    assert group_1_lrs[0] == lr_min
    assert group_2_lrs[0] == lr_min
    assert group_1_lrs[num_warmup_steps] == max(group_1_lrs)
    assert group_2_lrs[num_warmup_steps] == max(group_2_lrs)
    assert group_1_lrs[-num_cooldown_steps:] == expected_cooldown_steps
    assert group_2_lrs[-num_cooldown_steps:] == expected_cooldown_steps


def test_can_restore_scheduler_state():
    num_epochs = 10
    num_warmup_epochs = 2
    num_cooldown_epochs = 2
    num_steps_per_epoch = 10
    num_warmup_steps = num_warmup_epochs * num_steps_per_epoch
    num_cooldown_steps = num_cooldown_epochs * num_steps_per_epoch
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min = 1e-6
    expected_cooldown_steps = [lr_min] * num_cooldown_steps
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        num_warmup_epochs=num_warmup_epochs,
        warmup_starting_lr=lr_min,
        lr_min=lr_min,
        num_cooldown_epochs=2,
    )
    scheduler.step()
    scheduler_state_dict = scheduler.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler.step()
    expected_next_lr_1 = scheduler.optimizer.param_groups[0]["lr"]
    expected_next_lr_2 = scheduler.optimizer.param_groups[1]["lr"]

    # Restore optimizer and create new scheduler
    optimizer.load_state_dict(optimizer_state_dict)
    new_scheduler = CosineLrScheduler(
        optimizer,
        total_num_epochs=num_epochs,
        num_update_steps_per_epoch=num_steps_per_epoch,
        num_warmup_epochs=num_warmup_epochs,
        warmup_starting_lr=lr_min,
        lr_min=lr_min,
        num_cooldown_epochs=2,
    )
    new_scheduler.load_state_dict(scheduler_state_dict)
    new_scheduler.step()
    actual_next_lr_1 = new_scheduler.optimizer.param_groups[0]["lr"]
    actual_next_lr_2 = new_scheduler.optimizer.param_groups[1]["lr"]

    assert expected_next_lr_1 == actual_next_lr_1
    assert expected_next_lr_2 == actual_next_lr_2
