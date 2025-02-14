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

    Key features:
    - Can continue training from intermediate checkpoints without resetting
    - Uses geometric progression for checkpoint spacing (e.g., 50B, 100B, 200B tokens)
    
    This implementation provides two decay formulas.

    The paper's Inverse Proportional Decay:
        lr = 1 / (t * (1/lr_min - 1) + 1)

    - Derived from theoretical analysis on quadratic functions
    - Steeper initial decay, more gradual approach to lr_min
    - Optimal for quadratic loss landscapes
    
    A Sqrt Decay:
        lr = lr_min + (1 - lr_min) * (1 - sqrt(t))
        
    - Similar to traditional cosine decay patterns
    - More gradual initial decay, consistent decay rate
    - May be more robust across different architectures

    The decay fraction (default 0.1) determines what portion of steps before each
    checkpoint is used for decay. For example, with 50k steps and 0.1 decay_fraction,
    the final 5k steps would use decay.
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
        num_checkpoints: int = 3
        
    ):
        """
        Create a new WSDLrScheduler object which can be used to modify the learning rate in an optimizer's parameter
        groups.

        :param optimizer: PyTorch optimizer
        :param total_steps: Total number of training steps
        :param num_warmup_steps: Number of warmup steps
        :param decay_phase_ratio: Fraction of steps to use for decay before each checkpoint
        :param lr_min: Minimum learning rate to decay to
        :param warmup_starting_lr: Starting learning rate for warmup
        :param use_inverse_sqrt_decay: Whether to use a more gradual sqrt decay
        :param num_checkpoints: Number of checkpoints to use
        """
        super().__init__(optimizer)

        assert total_steps > 0, "total_steps must be positive"
        assert 0 <= decay_phase_ratio <= 1, "decay_fraction must be between 0 and 1"
        assert lr_min >= 0, "lr_min must be non-negative"
        assert num_checkpoints > 0, "num_checkpoints must be positive"

        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.decay_phase_ratio = decay_phase_ratio
        self.lr_min = lr_min
        self.warmup_starting_lr = warmup_starting_lr
        self.use_inverse_sqrt_decay = use_inverse_sqrt_decay
        self.num_checkpoints = num_checkpoints
        self.checkpoint_steps = estimate_checkpoint_steps(self.total_steps, self.num_checkpoints)
        self._get_checkpoint_info.cache_clear()

    def get_checkpoint_steps(self) -> List[int]:
        """Return the list of steps at which checkpoints occur.
        Useful for training loop coordination."""
        return self.checkpoint_steps

    @lru_cache(maxsize=1)
    def _get_checkpoint_info(self, num_updates):
        # Find current checkpoint period
        current_checkpoint = 0
        for i, checkpoint in enumerate(self.checkpoint_steps):
            if num_updates <= checkpoint:
                current_checkpoint = i
                break

        # Calculate period boundaries
        period_start = 0 if current_checkpoint == 0 else self.checkpoint_steps[current_checkpoint - 1]
        period_end = self.checkpoint_steps[current_checkpoint]
        total_period_steps = period_end - period_start
        steps_into_period = num_updates - period_start

        return total_period_steps, steps_into_period
        
    def get_updated_values(self, num_updates: int):
        # Handle warmup phase
        if num_updates < self.num_warmup_steps:
            return [
                self.warmup_starting_lr + (num_updates / self.num_warmup_steps) * (base_lr - self.warmup_starting_lr)
                for base_lr in self.base_lr_values
            ]

        total_period_steps, steps_into_period = self._get_checkpoint_info(num_updates)
        
       # Calculate decay phase
        decay_steps = int(total_period_steps * self.decay_phase_ratio)
        decay_start = total_period_steps - decay_steps

        # Check if in decay phase
        if steps_into_period >= decay_start:
            # relative_decay_step = (steps_into_period - decay_start) / decay_steps
            relative_decay_step = min((steps_into_period - decay_start) / decay_steps, 1.0)
            
            if self.use_inverse_sqrt_decay:
                lr_scale = self.lr_min + (1 - self.lr_min) * (1 - math.sqrt(relative_decay_step))
            else:
                # Paper's inverse proportional decay
                lr_scale = 1.0 / (relative_decay_step * (1.0 / self.lr_min - 1.0) + 1.0)
                
            return [base_lr * lr_scale for base_lr in self.base_lr_values]
        
        # Stable phase
        return self.base_lr_values

    def state_dict(self):
        # Clear the cache before saving state
        self._get_checkpoint_info.cache_clear()

        return {
            "num_updates": self._num_updates,
            "base_lr_values": self.base_lr_values,
        }

    def load_state_dict(self, state_dict: dict):
        # Clear the cache when loading state
        self._get_checkpoint_info.cache_clear()
        self._num_updates = state_dict["num_updates"]
        self.base_lr_values = state_dict["base_lr_values"]

    @classmethod
    def create_scheduler_fn(
        cls,
        total_num_epochs: int = TrainerPlaceholderValues.NUM_EPOCHS,
        num_update_steps_per_epoch: int = TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        num_warmup_steps: int = 0,
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
    # First checkpoint at ~25% of total steps
    first_checkpoint = total_steps // 4
    
    checkpoints = []
    for i in range(num_checkpoints):
        checkpoint = min(first_checkpoint * (2 ** i), total_steps)
        checkpoints.append(checkpoint)
        if checkpoint == total_steps:
            break
            
    return checkpoints