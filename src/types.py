from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
import torch

class Operation(Enum):
    ADDITION = "+"
    SUBTRACTION = "-"
    MULTIPLICATION = "*"
    DIVISION = "/"

class ModelType(Enum):
    DEEP_NARROW = "deep_narrow"
    WIDE_SHALLOW = "wide_shallow"
    VANILLA = "vanilla"
    CUDA_OPTIMIZED = "cuda_optimized"

@dataclass
class TrainingConfig:
    """Configuration for training settings"""
    use_amp: bool = True
    gradient_clip: float = 1.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_cuda: bool = torch.cuda.is_available()

@dataclass
class BaseConfig:
    """Base configuration shared across all models"""
    project_name: str = "arithmetic-grokking"
    experiment_name: str = "baseline"
    modulo: int = 97
    dropout: float = 0.1
    weight_decay: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 512
    n_epochs: int = 1000
    train_frac: float = 0.8
    seed: int = 42
    sequence_length: int = 3
    warmup_steps: int = 100
    operations: Optional[List[Operation]] = None
    vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.operations is None:
            self.operations = list(Operation)
        if self.vocab_size is None:
            self.vocab_size = self.modulo + len(self.operations)