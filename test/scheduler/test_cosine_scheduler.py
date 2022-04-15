import torch

from pytorch_accelerated.schedulers.cosine_scheduler import CosineScheduler


def create_model_and_optimizer(lr_1, lr_2):
    model = torch.nn.Linear(2, 1)
    params = list(model.parameters())
    optimizer = torch.optim.SGD([{'params': params[0],
                                  },
                                 {'params': params[1],
                                  'lr': lr_2
                                  }],
                                lr=lr_1)
    return model, optimizer


def collect_lrs_for_scheduler(scheduler,
                              num_epochs,
                              num_steps_per_epoch):
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
    scheduler = CosineScheduler(optimizer, total_num_epochs=num_epochs,
                                num_iterations_per_epoch=num_steps_per_epoch)

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler,
                                                         num_epochs,
                                                         num_steps_per_epoch)

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
    scheduler = CosineScheduler(optimizer, total_num_epochs=num_epochs,
                                num_iterations_per_epoch=num_steps_per_epoch,
                                lr_min=lr_min)

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler,
                                                         num_epochs,
                                                         num_steps_per_epoch + 1)

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
    scheduler = CosineScheduler(optimizer, total_num_epochs=num_epochs,
                                num_iterations_per_epoch=num_steps_per_epoch,
                                min_lr_ratio=lr_min_ratio)

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler,
                                                         num_epochs,
                                                         num_steps_per_epoch + 1)

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
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)
    scheduler = CosineScheduler(optimizer,
                                total_num_epochs=num_epochs,
                                num_iterations_per_epoch=num_steps_per_epoch,
                                lr_min=lr_min,
                                num_cooldown_epochs=num_cooldown_epochs)

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler,
                                                         num_epochs,
                                                         num_steps_per_epoch)

    assert group_1_lrs[-num_cooldown_steps:] == [lr_min] * num_cooldown_steps
    assert group_2_lrs[-num_cooldown_steps:] == [lr_min] * num_cooldown_steps


def test_warmup():
    pass


def test_warmup_and_cooldown():
    pass
