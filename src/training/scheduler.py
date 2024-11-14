import math
from typing import List, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """Learning rate scheduler with warmup and cosine decay"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs_after_warmup = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rates based on current step"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs_after_warmup
        ]

class LinearWarmupScheduler(_LRScheduler):
    """Linear learning rate warmup with optional plateau after warmup"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        plateau_steps: Optional[int] = None,
        min_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps or 0
        self.min_lr = min_lr
        self.base_lrs_after_warmup = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rates with linear warmup and optional plateau"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        if step < self.warmup_steps + self.plateau_steps:
            # Maintain learning rate during plateau
            return self.base_lrs_after_warmup
        
        if self.min_lr is not None:
            # Linear decay to min_lr after plateau
            remaining_steps = self.total_steps - (self.warmup_steps + self.plateau_steps)
            progress = (step - self.warmup_steps - self.plateau_steps) / remaining_steps
            progress = min(1.0, max(0.0, progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs_after_warmup
            ]
        
        return self.base_lrs_after_warmup

class CyclicalCosineScheduler(_LRScheduler):
    """Cyclical learning rate scheduler with cosine annealing"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        cycle_length: int,
        min_lr: float,
        warmup_steps: int = 0,
        cycle_multiplier: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.cycle_length = cycle_length
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cycle_multiplier = cycle_multiplier
        self.base_lrs_after_warmup = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rates with cyclical cosine annealing"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Calculate cycle position
        cycle_step = (step - self.warmup_steps) % self.cycle_length
        cycle_count = (step - self.warmup_steps) // self.cycle_length
        
        # Adjust cycle length for next cycle
        if cycle_count > 0:
            self.cycle_length = int(self.cycle_length * self.cycle_multiplier)
        
        # Cosine annealing within cycle
        progress = cycle_step / self.cycle_length
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs_after_warmup
        ]

class NoamScheduler(_LRScheduler):
    """Noam learning rate scheduler as described in 'Attention is All You Need'"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rates using the Noam scheme"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        step = max(1, self.last_epoch)
        scale = self.factor * (
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        ) * self.d_model ** (-0.5)
        
        return [base_lr * scale for base_lr in self.base_lrs]