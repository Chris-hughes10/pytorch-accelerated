import math
import pytest
from pytorch_accelerated.schedulers.wsd_scheduler import WSDLrScheduler
from test.scheduler.test_cosine_scheduler import create_model_and_optimizer

def collect_lrs_for_scheduler(scheduler, num_steps):
    group_1_lrs = []
    group_2_lrs = []

    for i in range(num_steps):
        scheduler.step()
        group_1_lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        group_2_lrs.append(scheduler.optimizer.param_groups[1]["lr"])

    return group_1_lrs, group_2_lrs

def test_wsd_stable_phase():
    num_steps = 1000
    lr_1_max = 0.01
    lr_2_max = 0.002
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=2
    )
    
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)
    checkpoints = scheduler.get_checkpoint_steps()
    
    # Check middle of each stable phase
    for i in range(len(checkpoints)-1):
        period_start = 0 if i == 0 else checkpoints[i-1]
        period_end = checkpoints[i]
        stable_check_point = period_start + (period_end - period_start) // 2
        
        assert group_1_lrs[stable_check_point] == lr_1_max
        assert group_2_lrs[stable_check_point] == lr_2_max

def test_wsd_warmup():
    num_steps = 1000
    num_warmup_steps = 100
    lr_1_max = 0.01
    lr_2_max = 0.002
    warmup_lr = 1e-6
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=2,
        num_warmup_steps=num_warmup_steps,
        warmup_starting_lr=warmup_lr
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)

    # Check warmup start and end
    assert group_1_lrs[0] == warmup_lr
    assert group_2_lrs[0] == warmup_lr
    assert group_1_lrs[num_warmup_steps] == lr_1_max
    assert group_2_lrs[num_warmup_steps] == lr_2_max

def test_wsd_decay():
    num_steps = 1000
    decay_phase_ratio = 0.1
    lr_1_max = 0.01
    lr_2_max = 0.002
    lr_min = 1e-6
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=3,
        decay_phase_ratio=decay_phase_ratio,
        lr_min=lr_min,
        use_inverse_sqrt_decay=True
    )
    
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)
    checkpoints = scheduler.get_checkpoint_steps()
    
    for i, checkpoint in enumerate(checkpoints):
        period_start = 0 if i == 0 else checkpoints[i-1]
        period_length = checkpoint - period_start
        decay_steps = int(period_length * decay_phase_ratio)
        decay_start = checkpoint - decay_steps
        
        # Just before decay
        if decay_start > 0:
            assert group_1_lrs[decay_start-1] == pytest.approx(lr_1_max)
            assert group_2_lrs[decay_start-1] == pytest.approx(lr_2_max)
            
        # Middle of decay
        mid_decay = decay_start + decay_steps // 2
        steps_into_decay = (mid_decay - decay_start)
        relative_decay_step = steps_into_decay / decay_steps
        expected_scale = lr_min + (1 - lr_min) * (1 - math.sqrt(relative_decay_step))
        assert group_1_lrs[mid_decay] == pytest.approx(lr_1_max * expected_scale)
        
        # End of decay
        assert group_1_lrs[checkpoint-1] == pytest.approx(lr_1_max * lr_min)

def test_wsd_resume_from_checkpoint():
    num_steps = 1000
    lr_1_max = 0.01
    lr_2_max = 0.002
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(
        optimizer,
        num_steps=num_steps,
        num_checkpoints=2
    )
    scheduler.step()
    scheduler_state_dict = scheduler.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler.step()
    expected_next_lr_1 = scheduler.optimizer.param_groups[0]["lr"]
    expected_next_lr_2 = scheduler.optimizer.param_groups[1]["lr"]

    # Restore optimizer and create new scheduler
    optimizer.load_state_dict(optimizer_state_dict)
    new_scheduler = WSDLrScheduler(
        optimizer,
        num_steps=num_steps,
        num_checkpoints=2
    )
    new_scheduler.load_state_dict(scheduler_state_dict)
    new_scheduler.step()
    actual_next_lr_1 = new_scheduler.optimizer.param_groups[0]["lr"]
    actual_next_lr_2 = new_scheduler.optimizer.param_groups[1]["lr"]

    assert expected_next_lr_1 == actual_next_lr_1
    assert expected_next_lr_2 == actual_next_lr_2

def test_wsd_both_decay_formulas():
    num_steps = 1000
    lr_max = 0.01
    lr_min = 1e-6
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)  # Same LR for comparison

    # Test with paper's decay
    paper_scheduler = WSDLrScheduler(
        optimizer,
        num_steps=num_steps,
        num_checkpoints=2,
        use_inverse_sqrt_decay=False,
        lr_min=lr_min
    )
    paper_lrs, _ = collect_lrs_for_scheduler(paper_scheduler, num_steps)

    # Test with sqrt decay
    sqrt_scheduler = WSDLrScheduler(
        optimizer,
        num_steps=num_steps,
        num_checkpoints=2,
        use_inverse_sqrt_decay=True,
        lr_min=lr_min
    )
    sqrt_lrs, _ = collect_lrs_for_scheduler(sqrt_scheduler, num_steps)

    # Verify different decay behaviors
    assert paper_lrs != sqrt_lrs  # Decay patterns should differ
    assert paper_lrs[499] == pytest.approx(lr_min, rel=1e-5)  # Both should reach lr_min
    assert sqrt_lrs[499] == pytest.approx(lr_min, rel=1e-5)