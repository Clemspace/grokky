import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

def shape_assert(tensor: torch.Tensor, expected_shape: Tuple[Optional[int], ...], name: str):
    """Utility function to check tensor shapes"""
    if not all(e is None or s == e for s, e in zip(tensor.shape, expected_shape)):
        raise ValueError(f"Shape mismatch in {name}: expected {expected_shape}, got {tensor.shape}")

class PositionalEncoding(nn.Module):
    """Positional encoding module for transformer models"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Save dimensions for validation
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")
        if d_model != self.d_model:
            raise ValueError(f"Input dimension {d_model} doesn't match expected {self.d_model}")
        
        return x + self.pe[:, :seq_len]

class TransformerLayerInitializer:
    """Utility class for initializing transformer layer weights"""
    
    @staticmethod
    def init_layer_weights(layer: nn.TransformerEncoderLayer):
        """Initialize transformer layer weights for better training"""
        # Initialize attention weights
        nn.init.normal_(layer.self_attn.in_proj_weight, mean=0, std=0.02)
        nn.init.normal_(layer.self_attn.out_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(layer.self_attn.in_proj_bias)
        nn.init.zeros_(layer.self_attn.out_proj.bias)
        
        # Initialize FFN weights
        nn.init.normal_(layer.linear1.weight, mean=0, std=0.02)
        nn.init.normal_(layer.linear2.weight, mean=0, std=0.02)
        nn.init.zeros_(layer.linear1.bias)
        nn.init.zeros_(layer.linear2.bias)

class WarmupCosineScheduler:
    """Warmup with cosine decay learning rate scheduler"""
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self, step_num: Optional[int] = None):
        """Update learning rate based on current step"""
        if step_num is not None:
            self.current_step = step_num
        
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        if step_num is None:
            self.current_step += 1
        
        return lr

class ModelOptimizer:
    """Utility class for model optimization settings"""
    
    @staticmethod
    def setup_cuda_optimizations():
        """Setup CUDA optimizations if available"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on Ampere GPUs
            torch.backends.cudnn.benchmark = True  # Optimize CUDA kernels