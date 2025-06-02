from functools import lru_cache, partial
from typing import Callable, List
from pytorch_accelerated.schedulers.scheduler_base import StatefulSchedulerBase
import torch
import math

from pytorch_accelerated.trainer import TrainerPlaceholderValues


class WSDLrScheduler(StatefulSchedulerBase):
    """
    Implements the Warmup-Stable-Decay (WSD) Simplified learning rate schedule as described in
    `Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective <https://arxiv.org/abs/2410.05192>`_.

    The schedule has three phases:
        1. Warmup: Linear warmup from warmup_starting_lr to base learning rate
        2. Stable: Maintains constant high learning rate
        3. Decay: Rapidly decays learning rate before each checkpoint

    This scheduler is designed to create intermediate model checkpoints during training. Each checkpoint
    involves decaying the learning rate to get better model performance.

    Use multiple checkpoints (typically 2-3) if:
        - Training on large datasets (>100B tokens) where intermediate models are useful for development/testing
        - You want to evaluate model performance vs training data size (e.g., does your model need full training?)
        - You might need to continue training later but want flexibility about when to stop training

    The scheduler uses geometric progression to space checkpoints evenly on a log scale:
        - First checkpoint is placed at 25% of total steps
        - Each subsequent checkpoint is ~2x steps from previous checkpoint

    Examples:
          - 2 checkpoints for 100K steps: [50K, 100K]
          - 3 checkpoints for 200K steps: [50K, 100K, 200K]
          - 4 checkpoints for 200K steps: [25K, 50K, 100K, 200K]

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
        - Training can be continued from a pre-decay (WSD) or post-decay (WSD-S) checkpoint
        - When continuing, scheduler starts a fresh stable phase with new total_steps
        - Decay phase ratio applies to new training length
        - No warmup is applied during continuation
        - State must be loaded via load_state_dict for continuation to work

    Example:
        Initial run (1000 steps, 0.1 decay ratio):
            - Steps 0-50: Optional warmup
            - Steps 50-900: Stable high learning rate
            - Steps 900-1000: Decay to lr_min

        Continuation (500 new steps, 0.1 decay ratio):
            - Steps 0-450: Stable high learning rate
            - Steps 450-500: Decay to lr_min

    .. Note:: This scheduler is designed to be used with the :class:`~pytorch_accelerated.callbacks.WSDCheckpointCallback` class,
        which handles saving and loading checkpoints.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = None,
        num_update_steps_per_epoch: int = None,
        total_steps: int = None,
        num_warmup_steps: int = 0,
        decay_phase_ratio: float = 0.1,
        lr_min: float = 1e-6,
        warmup_starting_lr: float = 1e-6,
        use_inverse_sqrt_decay: bool = True,
        num_checkpoints: int = 1,
        is_continuation_from_checkpoint: bool = False,
    ):
        """
        Create a new WSDLrScheduler object which can be used to modify the learning rate in an optimizer's parameter
        groups.

        :param optimizer: PyTorch optimizer
        :param num_epochs: Total number of training epochs
        :param num_update_steps_per_epoch: The number of update steps per epoch per process
        :param num_warmup_steps: Number of warmup steps. If None is passed, this will be set to 10% of the total steps
        :param total_steps: Total number of training steps per process
        :param decay_phase_ratio: Fraction of steps to use for decay before each checkpoint
        :param lr_min: The minimum learning rate as a fraction of the base learning rate. For example, 0.1 means decay to 1% of the base learning rate.
        :param warmup_starting_lr: Starting learning rate for warmup
        :param use_inverse_sqrt_decay: Whether to use a more gradual sqrt decay
        :param num_checkpoints: Number of checkpoints to use
        :param is_continuation_from_checkpoint: If True, indicates this is a continuation run from a previous checkpoint. The scheduler will start a fresh stable phase with new total_steps.

        .. Note::
         For continuation of training:
            - State must be loaded via load_state_dict before training
            - New training segment starts fresh stable phase
            - Decay ratio applies to new total_steps
            - No warmup is applied
        """
        super().__init__(optimizer)

        if (
            num_epochs is not None
            and num_update_steps_per_epoch is not None
            and total_steps is not None
        ):
            raise ValueError(
                "Only num_epochs and num_update_steps_per_epoch, or total_steps should be provided"
            )

        if num_epochs is not None and num_update_steps_per_epoch is not None:
            total_steps = num_epochs * num_update_steps_per_epoch

        assert total_steps > 0, "total_steps must be positive"
        assert 0 <= decay_phase_ratio <= 1, "decay_fraction must be between 0 and 1"
        assert lr_min >= 0, "lr_min must be non-negative"
        assert num_checkpoints > 0, "num_checkpoints must be positive"

        if num_warmup_steps is None:
            num_warmup_steps = int(0.01 * total_steps)

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
        self.is_continuation_from_checkpoint = is_continuation_from_checkpoint
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

    def get_phase_info(self, num_updates: int) -> dict:
        """Return full information about the current phase given the training step."""
        # Use the existing cached decay info since checkpoint_steps and decay_info align
        for info in self.checkpoint_decay_info:
            if info["period_end"] >= num_updates:
                return info
        # Return final period info if num_updates exceeds all checkpoints
        return self.checkpoint_decay_info[-1]

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
        if self.is_continuation_from_checkpoint:
            # Calculate new checkpoint steps based on continuation length
            checkpoints = estimate_checkpoint_steps(
                self.total_steps, self.num_checkpoints
            )

            # Find current checkpoint period
            current_checkpoint = 0
            for i, checkpoint in enumerate(checkpoints):
                if num_updates < checkpoint:
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
            if num_updates < checkpoint:
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
        # Handle warmup phase - but skip warmup in continuation
        if (
            not self.is_continuation_from_checkpoint
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
        if self.is_continuation_from_checkpoint:
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
        if self.is_continuation_from_checkpoint:
            new_total_steps = self.total_steps
            self._get_checkpoint_info.cache_clear()
            self.__dict__.update(state_dict)
            self.total_steps = new_total_steps  # Restore new training length

            # Reset internal counters for a fresh stable phase
            self._step_count = 0  # Start counting from 0 for new training run
            # Clear any previous checkpoint state so we start in stable phase
            self.previous_checkpoint_step = None
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
        num_update_steps_per_epoch: int = TrainerPlaceholderValues.PER_PROCESS_NUM_UPDATE_STEPS_PER_EPOCH,
        num_warmup_epochs: int = None,
        decay_phase_ratio: float = 0.1,
        lr_min: float = 1e-6,
        warmup_starting_lr: float = 1e-6,
        use_inverse_sqrt_decay: bool = True,
        num_checkpoints: int = 1,
        is_continuation_from_checkpoint: bool = False,
    ) -> Callable:
        """
        An alternative constructor which returns a function that accepts an optimizer and creates an instance of
        ``WSDLrScheduler``. This is primarily intended to be used with the :class:`~pytorch_accelerated.trainer.Trainer`
        as illustrated below::


            trainer = Trainer(
            ...,
            callbacks=[
                WSDCheckpointCallback(
                    save_dir="checkpoints",
                    initial_checkpoint="checkpoint_45000_pre_decay.pt",
                    )
                ],)

            trainer.train(
            train_dataset=train_dataset,
            num_epochs=num_epochs,
            per_device_batch_size=batch_size,
            create_scheduler_fn=CosineLrScheduler.WSDLrScheduler(is_continuation_from_checkpoint=True),
            )

        By default, the ``total_num_epochs`` and ``num_iterations_per_epoch`` arguments will be set by the
        :class:`~pytorch_accelerated.trainer.Trainer` with the correct values at runtime.
        if the number of warmup epochs is not set, this will be set to 10% of the total steps
        """

        if num_warmup_epochs is not None:
            num_warmup_steps = (
                TrainerPlaceholderValues.PER_PROCESS_NUM_UPDATE_STEPS_PER_EPOCH
                * num_warmup_epochs
            )
        else:
            num_warmup_steps = None

        return partial(
            cls,
            num_epochs=total_num_epochs,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            num_warmup_steps=num_warmup_steps,
            decay_phase_ratio=decay_phase_ratio,
            lr_min=lr_min,
            warmup_starting_lr=warmup_starting_lr,
            use_inverse_sqrt_decay=use_inverse_sqrt_decay,
            num_checkpoints=num_checkpoints,
            is_continuation_from_checkpoint=is_continuation_from_checkpoint,
        )


def estimate_checkpoint_steps(total_steps: int, num_checkpoints: int = 3) -> List[int]:
    """
    Estimates reasonable checkpoint steps given total training steps.
    Uses a geometric progression similar to paper's 50B->100B->200B pattern.

    :param total_steps: Total number of training steps
    :param num_checkpoints: Number of checkpoints desired
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
