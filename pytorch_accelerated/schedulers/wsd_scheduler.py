from functools import lru_cache
from typing import Callable, List
from pytorch_accelerated.schedulers.scheduler_base import StatefulSchedulerBase
import torch
import math

from pytorch_accelerated.trainer import TrainerPlaceholderValues


class WSDLrScheduler(StatefulSchedulerBase):
    """
    Implements the Warmup-Stable-Decay (WSD) Simplified learning rate schedule as described in
    'Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective'.

    The schedule has three phases:
    1. Warmup: Linear warmup from warmup_starting_lr to base learning rate
    2. Stable: Maintains constant high learning rate
    3. Decay: Rapidly decays learning rate before each checkpoint

    Checkpoints are spaced using geometric progression (e.g., 50B, 100B, 200B tokens).
    For each checkpoint:
    - The stable phase continues until decay_phase_ratio portion of steps remain
    - Then learning rate decays to lr_min * base_lr using selected decay formula

      Two decay formulas are provided:

    1. Inverse Proportional Decay (paper's formula):
        lr = 1 / (t * (1/lr_min - 1) + 1)
        - Derived from theoretical analysis on quadratic functions
        - Steeper initial decay, more gradual approach to lr_min
        - Optimal for quadratic loss landscapes

    2. Sqrt Decay:
        lr = lr_min + (1 - lr_min) * (1 - sqrt(t))
        - Similar to traditional cosine decay patterns
        - More gradual initial decay, consistent decay rate
        - May be more robust across different architectures

    Continuation Behavior:
    - Training can be continued from a pre-decay checkpoint
    - When continuing, scheduler starts a fresh stable phase with new total_steps
    - Decay phase ratio applies to new training length
    - No warmup is applied during continuation
    - State must be loaded via load_state_dict for continuation to work

    Example:
        Initial run (1000 steps, 0.1 decay ratio):
        - Steps 0-50: Optional warmup
        - Steps 50-900: Stable high learning rate
        - Steps 900-1000: Decay

        Continuation (500 new steps, 0.1 decay ratio):
        - Steps 0-450: Stable high learning rate
        - Steps 450-500: Decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        num_warmup_steps: int = 0,
        decay_phase_ratio: float = 0.1,
        lr_min: float = 1e-6,
        warmup_starting_lr: float = 1e-6,
        use_inverse_sqrt_decay: bool = True,
        num_checkpoints: int = 3,
        is_continuation_from_pre_decay: bool = False,
    ):
        """
        Create a new WSDLrScheduler object which can be used to modify the learning rate in an optimizer's parameter
        groups.

        :param optimizer: PyTorch optimizer
        :param total_steps: Total number of training steps
        :param num_warmup_steps: Number of warmup steps
        :param decay_phase_ratio: Fraction of steps to use for decay before each checkpoint
        :param lr_min: The minimum learning rate as a fraction of the base learning rate. For example, 0.1 means decay to 10% of the base learning rate.
        :param warmup_starting_lr: Starting learning rate for warmup
        :param use_inverse_sqrt_decay: Whether to use a more gradual sqrt decay
        :param num_checkpoints: Number of checkpoints to use
        :param is_continuation_from_pre_decay: If True, indicates this is a continuation run from pre-decay checkpoint

        .. Note::
         For continuation training:
            - State must be loaded via load_state_dict before training
            - New training segment starts fresh stable phase
            - Decay ratio applies to new total_steps
            - No warmup is applied
        """
        super().__init__(optimizer)

        assert total_steps > 0, "total_steps must be positive"
        assert 0 <= decay_phase_ratio <= 1, "decay_fraction must be between 0 and 1"
        assert lr_min >= 0, "lr_min must be non-negative"
        assert num_checkpoints > 0, "num_checkpoints must be positive"
        assert num_warmup_steps >= 0, "num_warmup_steps must be non-negative"
        assert (
            num_warmup_steps <= total_steps
        ), "num_warmup_steps cannot exceed total_steps"

        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.decay_phase_ratio = decay_phase_ratio
        self.lr_min = lr_min
        self.warmup_starting_lr = warmup_starting_lr
        self.use_inverse_sqrt_decay = use_inverse_sqrt_decay
        self.num_checkpoints = num_checkpoints
        self.checkpoint_steps = estimate_checkpoint_steps(
            self.total_steps, self.num_checkpoints
        )
        self.is_continuation_from_pre_decay = is_continuation_from_pre_decay
        self.previous_checkpoint_step = None
        self._get_checkpoint_info.cache_clear()

        if warmup_starting_lr > max(self.base_lr_values):
            raise ValueError(
                "warmup_starting_lr should not exceed maximum base learning rate"
            )

        # Validate decay ratio based on checkpoint spacing
        if num_checkpoints > 1:
            min_spacing = float("inf")
            for i in range(len(self.checkpoint_steps) - 1):
                spacing = self.checkpoint_steps[i + 1] - self.checkpoint_steps[i]
                min_spacing = min(min_spacing, spacing)

            # For geometric progression spacing, max safe ratio is 1/3
            max_safe_ratio = 1 / 3

            if decay_phase_ratio > max_safe_ratio:
                raise ValueError(
                    f"Decay phase ratio {decay_phase_ratio} is too large and may cause "
                    f"overlapping decay phases. Maximum safe ratio is {max_safe_ratio:.2f} "
                    f"with {num_checkpoints} checkpoints using geometric progression"
                )

        # Calculate and store decay information for each checkpoint period
        self.checkpoint_decay_info = self._calculate_decay_info()

    def get_checkpoint_steps(self) -> List[int]:
        """Return the list of steps at which checkpoints occur.
        Useful for training loop coordination."""
        return self.checkpoint_steps

    def get_current_phase(self, num_updates: int) -> str:
        if (
            not self.is_continuation_from_pre_decay
            and num_updates < self.num_warmup_steps
        ):
            return "warmup"
        for info in self.checkpoint_decay_info:
            if info["period_end"] >= num_updates:
                if num_updates >= info["pre_decay_step"]:
                    return "decay"
                return "stable"
        return "stable"

    def _calculate_decay_info(self) -> List[dict]:
        """Calculate decay information for each checkpoint period"""
        decay_info = []

        for i, checkpoint in enumerate(self.checkpoint_steps):
            period_start = 0 if i == 0 else self.checkpoint_steps[i - 1]
            period_length = checkpoint - period_start

            period_info = {
                "period_start": period_start,
                "period_end": checkpoint,
                "decay_steps": int(period_length * self.decay_phase_ratio),
                "pre_decay_step": checkpoint
                - int(period_length * self.decay_phase_ratio),
            }
            decay_info.append(period_info)

        return decay_info

    def get_decay_info(self) -> List[dict]:
        """Get information about decay phases for all checkpoint periods.

        Returns:
            List[dict]: List of dicts containing for each period:
                - period_start: Start of period
                - period_end: End of period (checkpoint)
                - decay_steps: Number of steps in decay phase
                - pre_decay_step: Step before decay phase starts
        """
        return self.checkpoint_decay_info

    @lru_cache(maxsize=1)
    def _get_checkpoint_info(self, num_updates):
        """Get information about the current checkpoint period."""
        if self.is_continuation_from_pre_decay:
            # Calculate new checkpoint steps based on continuation length
            checkpoints = estimate_checkpoint_steps(
                self.total_steps, self.num_checkpoints
            )

            # Find current checkpoint period
            current_checkpoint = 0
            for i, checkpoint in enumerate(checkpoints):
                if num_updates <= checkpoint:
                    current_checkpoint = i
                    break

            # Calculate period information
            period_start = (
                0 if current_checkpoint == 0 else checkpoints[current_checkpoint - 1]
            )
            period_end = checkpoints[current_checkpoint]
            period_length = period_end - period_start
            steps_into_period = num_updates - period_start

            return period_length, steps_into_period

        # Original non-continuation logic
        current_checkpoint = 0
        for i, checkpoint in enumerate(self.checkpoint_steps):
            if num_updates <= checkpoint:
                current_checkpoint = i
                break

        period_start = (
            0
            if current_checkpoint == 0
            else self.checkpoint_steps[current_checkpoint - 1]
        )
        period_end = self.checkpoint_steps[current_checkpoint]
        period_length = period_end - period_start
        steps_into_period = num_updates - period_start
        return period_length, steps_into_period

    def get_updated_values(self, num_updates: int):
        if (
            self.is_continuation_from_pre_decay
            and self.previous_checkpoint_step is None
        ):
            raise ValueError(
                "no previous checkpoint step found - state must be loaded before continuing training"
            )

        # Handle warmup phase - but skip warmup in continuation
        if (
            not self.is_continuation_from_pre_decay
            and num_updates < self.num_warmup_steps
        ):
            return [
                self.warmup_starting_lr
                + (num_updates / self.num_warmup_steps)
                * (base_lr - self.warmup_starting_lr)
                for base_lr in self.base_lr_values
            ]

        # Get period information
        total_period_steps, steps_into_period = self._get_checkpoint_info(num_updates)

        # Calculate decay phase
        decay_steps = int(total_period_steps * self.decay_phase_ratio)
        decay_start = total_period_steps - decay_steps

        # In continuation mode, we should maintain stable phase until new decay point
        if self.is_continuation_from_pre_decay:
            # During stable phase, return base learning rates
            if steps_into_period < decay_start:
                return self.base_lr_values
            # Track pre-decay step
            elif steps_into_period == decay_start - 1:
                self.previous_checkpoint_step = num_updates

        # For regular mode, track pre-decay step
        elif steps_into_period == decay_start - 1:
            self.previous_checkpoint_step = num_updates

        # Check if in decay phase
        if steps_into_period >= decay_start:
            relative_decay_step = min(
                (steps_into_period - decay_start) / decay_steps, 1.0
            )

            if self.use_inverse_sqrt_decay:
                # Sqrt decay formula
                return [
                    base_lr
                    * (
                        self.lr_min
                        + (1 - self.lr_min) * (1 - math.sqrt(relative_decay_step))
                    )
                    for base_lr in self.base_lr_values
                ]
            else:
                # Paper's inverse proportional decay
                return [
                    base_lr
                    / (
                        relative_decay_step * (1.0 / self.lr_min)
                        + (1.0 - relative_decay_step) * 1.0
                    )
                    for base_lr in self.base_lr_values
                ]

        # Stable phase
        return self.base_lr_values

    def state_dict(self):
        # Clear the cache before saving state
        self._get_checkpoint_info.cache_clear()

        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer" and not key.startswith("_")
        }

    def load_state_dict(self, state_dict: dict):
        # In continuation mode, preserve new total_steps
        if self.is_continuation_from_pre_decay:
            new_total_steps = self.total_steps
            self._get_checkpoint_info.cache_clear()
            self.__dict__.update(state_dict)
            self.total_steps = new_total_steps  # Restore new training length

            # Reset internal counters for fresh start
            self._step_count = 0  # Start counting from 0 for new training run
            self.checkpoint_steps = estimate_checkpoint_steps(
                self.total_steps, self.num_checkpoints
            )  # Recalculate checkpoints for new length
            self.checkpoint_decay_info = self._calculate_decay_info()

        else:
            # Standard state loading
            self._get_checkpoint_info.cache_clear()
            self.__dict__.update(state_dict)

    @classmethod
    def create_scheduler_fn(
        cls,
        total_num_epochs: int = TrainerPlaceholderValues.NUM_EPOCHS,
        num_update_steps_per_epoch: int = TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        decay_phase_ratio: float = 0.1,
        lr_min: float = 1e-6,
        warmup_starting_lr: float = 1e-6,
        use_inverse_sqrt_decay: bool = True,
        num_checkpoints: int = 3,
    ) -> Callable:
        """Creates a scheduler function that the trainer can use.

        Returns a function that accepts an optimizer and creates an instance of WSDScheduler.
        The trainer will replace TrainerPlaceholderValues with actual values at runtime.
        """

        def create_scheduler(optimizer):
            total_steps = total_num_epochs * num_update_steps_per_epoch
            num_warmup_steps = int(0.01 * total_steps)

            return cls(
                optimizer=optimizer,
                total_steps=total_steps,
                num_warmup_steps=num_warmup_steps,
                decay_phase_ratio=decay_phase_ratio,
                lr_min=lr_min,
                warmup_starting_lr=warmup_starting_lr,
                use_inverse_sqrt_decay=use_inverse_sqrt_decay,
                num_checkpoints=num_checkpoints,
            )

        return create_scheduler


def estimate_checkpoint_steps(total_steps: int, num_checkpoints: int = 3) -> List[int]:
    """
    Estimates reasonable checkpoint steps given total training steps.
    Uses a geometric progression similar to paper's 50B->100B->200B pattern.

    Args:
        total_steps: Total number of training steps
        num_checkpoints: Number of checkpoints desired
    """
    if num_checkpoints == 1:
        # Single checkpoint case - put checkpoint at end
        return [total_steps]

    # Multiple checkpoint case - use geometric progression
    # First checkpoint at ~25% of total steps
    first_checkpoint = total_steps // 4

    checkpoints = []
    for i in range(num_checkpoints):
        checkpoint = min(first_checkpoint * (2**i), total_steps)
        checkpoints.append(checkpoint)
        if checkpoint == total_steps:
            break

    return checkpoints
