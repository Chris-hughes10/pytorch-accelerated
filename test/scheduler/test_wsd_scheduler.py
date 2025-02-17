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

def test_wsd_decay_info_single_checkpoint():
    """Test that get_decay_info returns correct decay information for single checkpoint case"""
    num_steps = 100
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=1,
        decay_phase_ratio=0.1
    )
    
    decay_info = scheduler.get_decay_info()
    assert len(decay_info) == 1, "Should have one period for single checkpoint"
    
    period = decay_info[0]
    assert period["period_start"] == 0, "Period should start at beginning"
    assert period["period_end"] == num_steps, "Period should end at total steps"
    assert period["decay_steps"] == 10, "Should have 10% decay steps"
    assert period["pre_decay_step"] == 90, "Decay should start at 90% through training"

def test_wsd_decay_info_multiple_checkpoints():
    """Test that get_decay_info returns correct decay information for multiple checkpoints"""
    num_steps = 100
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=3,
        decay_phase_ratio=0.1
    )
    
    decay_info = scheduler.get_decay_info()
    checkpoints = scheduler.get_checkpoint_steps()
    
    assert len(decay_info) == len(checkpoints), "Should have info for each checkpoint period"
    
    # Verify each period's information
    for i, period in enumerate(decay_info):
        period_start = 0 if i == 0 else checkpoints[i-1]
        period_end = checkpoints[i]
        period_length = period_end - period_start
        expected_decay_steps = int(period_length * 0.1)
        
        assert period["period_start"] == period_start, f"Period {i} should start at checkpoint {i-1}"
        assert period["period_end"] == period_end, f"Period {i} should end at checkpoint {i}"
        assert period["decay_steps"] == expected_decay_steps, f"Period {i} should have correct decay steps"
        assert period["pre_decay_step"] == period_end - expected_decay_steps, f"Period {i} should have correct pre-decay step"
        
        # Verify decay points are properly spaced
        if i > 0:
            prev_period = decay_info[i-1]
            assert prev_period["period_end"] == period["period_start"], \
                f"Period {i} should start where period {i-1} ends"
            assert prev_period["period_end"] < period["pre_decay_step"], \
                f"Decay phases should not overlap between periods {i-1} and {i}"


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

def test_wsd_pre_decay_continuation():
    """Test continuation from pre-decay behaves correctly with new training length"""
    num_steps = 100
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    # Initial training 
    initial_scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        num_checkpoints=1,
        decay_phase_ratio=0.1  # 10% decay phase
    )
    
    # With 1 checkpoint, decay happens at end of training
    decay_steps = int(num_steps * initial_scheduler.decay_phase_ratio)
    pre_decay_step = num_steps - decay_steps

    print(f"\nInitial training setup:")
    print(f"Total steps: {num_steps}")
    print(f"Decay steps: {decay_steps}")
    print(f"Pre-decay step: {pre_decay_step}")
    # Print initial training info
    
    initial_lrs, _ = collect_lrs_for_scheduler(initial_scheduler, pre_decay_step)
    print(f"Final initial lr: {initial_lrs[-1]}")
    
    # Save state at pre-decay point
    scheduler_state = initial_scheduler.state_dict()
    print(f"\nScheduler state at save:")
    print(f"Previous checkpoint step: {scheduler_state.get('previous_checkpoint_step')}")
    print(f"Decay phase ratio: {scheduler_state.get('decay_phase_ratio')}")
    
    # Continue with new shorter training run
    continue_steps = 50  # New total steps
    continued_scheduler = WSDLrScheduler(
        optimizer,
        total_steps=continue_steps,
        is_continuation_from_pre_decay=True
    )
    continued_scheduler.load_state_dict(scheduler_state)
    
    continued_lrs, _ = collect_lrs_for_scheduler(continued_scheduler, continue_steps)
    
    # Calculate when decay should start in new training
    new_decay_start = int(continue_steps * (1 - continued_scheduler.decay_phase_ratio))
    print(f"\nContinuation setup:")
    print(f"Total steps: {continue_steps}")
    print(f"New decay start: {new_decay_start}")
    print(f"Continuation first 5 lrs: {continued_lrs[:5]}")
    print(f"Expected lr: {lr_max}")

     # Find first deviation from lr_max
    for i, lr in enumerate(continued_lrs[:new_decay_start]):
        if abs(lr - lr_max) > 1e-10:  # Use small epsilon for float comparison
            print(f"\nFirst lr deviation at step {i}:")
            print(f"Expected: {lr_max}")
            print(f"Got: {lr}")
            break
    
    # Verify behavior
    # 1. Stable phase should maintain max lr
    assert all(lr == lr_max for lr in continued_lrs[:new_decay_start]), \
        "Should maintain stable lr until new decay point"
        
    # 2. Should start decay at correct point
    assert continued_lrs[new_decay_start + 1] < lr_max, \
        "Should start decay at calculated point"
        
    # 3. Should reach minimum by end
    min_lr = lr_max * continued_scheduler.lr_min
    assert continued_lrs[-1] <= min_lr * 1.1, \
        "Should decay to specified minimum"
    

def test_wsd_continuation_validation():
    """Test validation when attempting continuation without proper state"""
    num_steps = 1000
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    # Create scheduler in continuation mode without state
    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        is_continuation_from_pre_decay=True
    )
    
    # Should raise when trying to get lr values without state
    with pytest.raises(ValueError, match="no previous checkpoint step found"):
        scheduler.get_updated_values(0)
    
    # Should work after loading valid state
    valid_state = {
        'previous_checkpoint_step': 500,
        'total_steps': 1000,
        'num_warmup_steps': 0,
        'decay_phase_ratio': 0.1,
        'lr_min': 1e-6,
        'warmup_starting_lr': 1e-6,
        'use_inverse_sqrt_decay': True,
        'num_checkpoints': 2,
        'is_continuation_from_pre_decay': True,
        'checkpoint_steps': [250, 500, 750, 1000]
    }
    
    scheduler.load_state_dict(valid_state)
    # Should now work without error
    scheduler.get_updated_values(0)

def test_wsd_continuation_validation():
    """Test proper validation when attempting continuation without state"""
    num_steps = 1000
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)

    # Create scheduler in continuation mode
    scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        is_continuation_from_pre_decay=True
    )
    
    # Should raise error when getting lr values without state loaded
    with pytest.raises(ValueError, match="no previous checkpoint step found"):
        scheduler.get_updated_values(0)
        
    # Should work after loading valid state
    scheduler.previous_checkpoint_step = 500  # Simulate loaded state
    # Should not raise error
    scheduler.get_updated_values(0)


def test_wsd_continuation_state():
    """Test state handling during continuation"""
    num_steps = 1000
    lr_max = 0.01
    model, optimizer = create_model_and_optimizer(lr_max, lr_max)
    
    # Create initial scheduler and train to pre-decay
    initial_scheduler = WSDLrScheduler(
        optimizer,
        total_steps=num_steps,
        decay_phase_ratio=0.1
    )
    
    checkpoints = initial_scheduler.get_checkpoint_steps()
    first_checkpoint = checkpoints[0]
    decay_steps = int(first_checkpoint * initial_scheduler.decay_phase_ratio)
    pre_decay_step = first_checkpoint - decay_steps - 1
    
    # Train to pre-decay point
    _, _ = collect_lrs_for_scheduler(initial_scheduler, pre_decay_step)
    state = initial_scheduler.state_dict()
    
    # Create new scheduler with different initial settings
    continued_scheduler = WSDLrScheduler(
        optimizer,
        total_steps=500,  # Different steps
        decay_phase_ratio=0.2,  # Different ratio
        is_continuation_from_pre_decay=True
    )
    
    # Load state and verify it preserves original values
    continued_scheduler.load_state_dict(state)
    assert continued_scheduler.previous_checkpoint_step == pre_decay_step
    assert continued_scheduler.total_steps == 500  # Should keep new training length
    assert continued_scheduler.decay_phase_ratio == state['decay_phase_ratio']  # Should restore original ratio