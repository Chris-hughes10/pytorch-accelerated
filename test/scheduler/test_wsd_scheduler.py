import math
import pytest
from pytorch_accelerated.schedulers.wsd_scheduler import WSDLrScheduler
from test.scheduler.test_cosine_scheduler import create_model_and_optimizer


def collect_lrs_for_scheduler(scheduler, num_steps):
    group_1_lrs = []
    group_2_lrs = []

    for i in range(num_steps + 1):
        scheduler.step()
        group_1_lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        group_2_lrs.append(scheduler.optimizer.param_groups[1]["lr"])

    return group_1_lrs, group_2_lrs


def test_wsd_stable_phase():
    """
    Test the WSD scheduler's stable phase behavior.

    Verifies that learning rates remain constant at their maximum values during
    the stable phase between decay periods. Tests this using two parameter groups
    with different learning rates:
    - Group 1: 0.01
    - Group 2: 0.002

    Checks the middle of each stable phase to ensure learning rates haven't
    deviated from their maximum values.
    """
    num_steps = 1000
    lr_1_max = 0.01
    lr_2_max = 0.002
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(optimizer, total_steps=num_steps, num_checkpoints=2)

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)
    checkpoints = scheduler.get_checkpoint_steps()

    # Check middle of each stable phase
    for i in range(len(checkpoints) - 1):
        period_start = 0 if i == 0 else checkpoints[i - 1]
        period_end = checkpoints[i]
        stable_check_point = period_start + (period_end - period_start) // 2

        assert group_1_lrs[stable_check_point] == lr_1_max
        assert group_2_lrs[stable_check_point] == lr_2_max


def test_wsd_warmup():
    """
    Test the WSD scheduler's warmup phase behavior.

    Verifies that learning rates correctly:
    1. Start at the specified warmup_starting_lr (1e-6)
    2. Linearly increase during warmup
    3. Reach their target maximum values after warmup completion

    Tests this using two parameter groups with different target learning rates:
    - Group 1: 0.01
    - Group 2: 0.002
    """
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
        warmup_starting_lr=warmup_lr,
    )
    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)

    # Check warmup start and end
    assert group_1_lrs[0] == warmup_lr
    assert group_2_lrs[0] == warmup_lr
    assert group_1_lrs[num_warmup_steps] == lr_1_max
    assert group_2_lrs[num_warmup_steps] == lr_2_max


def calculate_scale(rel_step, lr_min):
    # Ensure rel_step is capped at 1.0 to match implementation
    rel_step = min(rel_step, 1.0)
    return lr_min + (1 - lr_min) * (1 - math.sqrt(rel_step))


def test_wsd_decay():
    """
    Test the WSD (Warmup-Stable-Decay) scheduler's decay phase behavior.

    This test verifies three key aspects of the decay phase for a scheduler
    configured with two parameter groups:

    1. Stable Phase
        - Confirms learning rates remain at their maximum values
          just before each decay phase begins

    2. Decay Phase Progression
        - Tests the sqrt decay formula at multiple points:
            * Start of decay phase
            * Middle of decay phase
            * End of decay phase
        - Verifies both parameter groups scale correctly from their
          respective maximum learning rates
        - Ensures learning rates decrease monotonically during decay

    3. Final Values
        - Confirms learning rates approach lr_min * max_lr at the
          end of each decay phase

    Test Configuration:
        - Uses 1000 total steps
        - 3 checkpoints (creating 3 decay phases)
        - 10% decay phase ratio
        - Two parameter groups with different max learning rates:
            * Group 1: 0.01
            * Group 2: 0.002
        - Minimum learning rate: 1e-6
        - Uses sqrt decay formula
    """
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
        use_inverse_sqrt_decay=True,
    )

    group_1_lrs, group_2_lrs = collect_lrs_for_scheduler(scheduler, num_steps)
    checkpoints = scheduler.get_checkpoint_steps()

    for i, checkpoint in enumerate(checkpoints):
        period_start = 0 if i == 0 else checkpoints[i - 1]
        period_length = checkpoint - period_start
        decay_steps = int(period_length * decay_phase_ratio)
        decay_start = checkpoint - decay_steps

        # Test stable phase learning rate
        if decay_start > 0:
            assert group_1_lrs[decay_start - 1] == pytest.approx(lr_1_max, rel=1e-5)
            assert group_2_lrs[decay_start - 1] == pytest.approx(lr_2_max, rel=1e-5)

        # Test several points during decay phase
        decay_points = [
            decay_start,  # Start of decay
            decay_start + decay_steps // 2,  # Middle of decay
            min(checkpoint - 1, num_steps - 1),  # End of decay (with bounds check)
        ]

        for step in decay_points:
            steps_into_decay = step - decay_start
            relative_decay_step = steps_into_decay / decay_steps
            expected_scale = calculate_scale(relative_decay_step, lr_min)

            # Test both parameter groups with appropriate tolerance
            assert group_1_lrs[step] == pytest.approx(
                lr_1_max * expected_scale, rel=1e-5
            ), f"Mismatch at step {step}, expected scale: {expected_scale}"
            assert group_2_lrs[step] == pytest.approx(
                lr_2_max * expected_scale, rel=1e-5
            ), f"Mismatch at step {step}, expected scale: {expected_scale}"

        # Verify decay phase learning rates are monotonically decreasing
        decay_phase_lrs = group_1_lrs[decay_start : min(checkpoint, num_steps)]
        assert all(
            x >= y for x, y in zip(decay_phase_lrs, decay_phase_lrs[1:])
        ), "Learning rates should be monotonically decreasing during decay"


def test_wsd_resume_from_checkpoint():
    """
    Test the WSD scheduler's ability to resume training from a checkpoint.

    Verifies that:
    1. Scheduler state can be saved and restored correctly
    2. Learning rates continue from the same point after restoration
    3. Both parameter groups maintain correct learning rates

    This ensures the scheduler can be used reliably in scenarios requiring
    training resumption or checkpointing.
    """
    num_steps = 1000
    lr_1_max = 0.01
    lr_2_max = 0.002
    model, optimizer = create_model_and_optimizer(lr_1_max, lr_2_max)

    scheduler = WSDLrScheduler(optimizer, total_steps=num_steps, num_checkpoints=2)
    scheduler.step()
    scheduler_state_dict = scheduler.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler.step()
    expected_next_lr_1 = scheduler.optimizer.param_groups[0]["lr"]
    expected_next_lr_2 = scheduler.optimizer.param_groups[1]["lr"]

    # Restore optimizer and create new scheduler
    optimizer.load_state_dict(optimizer_state_dict)
    new_scheduler = WSDLrScheduler(optimizer, total_steps=num_steps, num_checkpoints=2)
    new_scheduler.load_state_dict(scheduler_state_dict)
    new_scheduler.step()
    actual_next_lr_1 = new_scheduler.optimizer.param_groups[0]["lr"]
    actual_next_lr_2 = new_scheduler.optimizer.param_groups[1]["lr"]

    assert expected_next_lr_1 == actual_next_lr_1
    assert expected_next_lr_2 == actual_next_lr_2


@pytest.mark.parametrize(
    "lr_max,lr_min_ratio",
    [  # Changed lr_min to lr_min_ratio
        (1.0, 0.1),  # Decays to 0.1 of base lr
        (0.01, 0.1),  # Paper's typical values - decays to 0.001
        (0.001, 0.1),  # Decays to 0.0001
        (0.1, 0.01),  # Steeper decay - to 0.001
        (0.01, 0.01),  # Steeper decay - to 0.0001
    ],
)
def test_wsd_decay_formulas(lr_max, lr_min_ratio):
    """Test both decay formulas respect minimum lr and maintain monotonicity"""
    num_steps = 1000
    decay_phase_ratio = 0.1
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=2,
        decay_phase_ratio=decay_phase_ratio,
        lr_min=lr_min_ratio,  # This is now a ratio of base lr
    )

    lrs, _ = collect_lrs_for_scheduler(scheduler, num_steps)

    # Get decay phase for first checkpoint
    checkpoints = scheduler.get_checkpoint_steps()
    first_checkpoint = checkpoints[0]
    decay_steps = int(first_checkpoint * decay_phase_ratio)
    decay_start = first_checkpoint - decay_steps

    # Check minimum learning rate - multiply by base lr to get actual min
    expected_min_lr = lr_max * lr_min_ratio
    min_lr = min(lrs)
    assert (
        min_lr >= expected_min_lr
    ), f"Learning rate {min_lr} went below minimum {expected_min_lr}"

    # Verify monotonic decrease during decay
    decay_lrs = lrs[decay_start:first_checkpoint]
    assert all(
        x >= y for x, y in zip(decay_lrs, decay_lrs[1:])
    ), "Learning rate should decrease monotonically during decay phase"


def test_wsd_invalid_configurations():
    """
    Test that the scheduler correctly validates its configuration parameters.
    """
    num_steps = 1000
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    # Test invalid decay ratio
    with pytest.raises(AssertionError):
        WSDLrScheduler(optimizer, total_steps=num_steps, decay_phase_ratio=1.5)

    # Test invalid num_steps
    with pytest.raises(AssertionError):
        WSDLrScheduler(optimizer, total_steps=0)

    # Test invalid lr_min
    with pytest.raises(AssertionError):
        WSDLrScheduler(optimizer, total_steps=num_steps, lr_min=-0.1)


def test_wsd_checkpoint_spacing():
    """
    Test that checkpoint steps are correctly spaced using geometric progression.
    """
    num_steps = 1000
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    scheduler = WSDLrScheduler(optimizer, total_steps=num_steps, num_checkpoints=3)

    checkpoints = scheduler.get_checkpoint_steps()

    # Verify geometric progression (approximately)
    ratios = [c2 / c1 for c1, c2 in zip(checkpoints[:-1], checkpoints[1:])]
    assert all(
        abs(r1 - r2) < 0.1 for r1, r2 in zip(ratios[:-1], ratios[1:])
    ), "Checkpoint spacing should follow geometric progression"


def test_wsd_warmup_edge_cases():
    """
    Test edge cases in warmup behavior.

    Verifies:
    1. Zero warmup steps - should start directly at max learning rate
    2. Full steps warmup - should follow standard linear warmup pattern
    3. Learning rates should remain within bounds
    4. Warmup starting lr is respected
    """
    num_steps = 1000
    lr_max = 0.01
    warmup_starting_lr = 1e-8
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    # Test zero warmup steps
    scheduler = WSDLrScheduler(optimizer, total_steps=num_steps, num_warmup_steps=0)
    scheduler.step()
    assert scheduler.optimizer.param_groups[0]["lr"] == lr_max

    # Test warmup_steps = total_steps
    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_warmup_steps=num_steps,
        warmup_starting_lr=warmup_starting_lr,
    )
    group_1_lrs, _ = collect_lrs_for_scheduler(scheduler, num_steps)

    # Verify learning rate behavior
    assert group_1_lrs[0] == warmup_starting_lr  # Should start at warmup_starting_lr
    assert max(group_1_lrs) <= lr_max  # Should never exceed max

    # Verify warmup progression
    quarter_step = num_steps // 4
    assert group_1_lrs[quarter_step] > group_1_lrs[0]  # Should increase during warmup
    assert group_1_lrs[quarter_step] < lr_max  # But not reach max yet
