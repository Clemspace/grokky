import torch
import torch.nn as nn
import math
from typing import Optional, List
from dataclasses import dataclass

from .components import PositionalEncoding, TransformerLayerInitializer
from .config import ModelConfig, ModelType

class BaseTransformer(nn.Module):
    """Base transformer class with shared functionality"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embedding with scaled initialization
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Positional encoding and input processing
        self.pos_encoding = PositionalEncoding(config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        
        # Output head
        self.final_norm = nn.LayerNorm(config.d_model)
        self.final = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize output layer
        nn.init.normal_(self.final.weight, mean=0, std=0.02)
        nn.init.zeros_(self.final.bias)
    
    def _init_layer_weights(self, layer: nn.TransformerEncoderLayer):
        """Initialize transformer layer weights"""
        TransformerLayerInitializer.init_layer_weights(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to correct device if necessary
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        
        # Input normalization and dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Process through transformer layers (implemented by subclasses)
        x = self.transformer_forward(x)
        
        # Final prediction
        x = self.final_norm(x)
        x = self.final(x)
        
        return x
    
    def transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses"""
        raise NotImplementedError

class DeepNarrowTransformer(BaseTransformer):
    """Deep and narrow transformer architecture"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Deep stack of transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layers)
        ])
        
        # Initialize transformer weights
        for layer in self.transformer_layers:
            self._init_layer_weights(layer)
    
    def transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_layers:
            x = layer(x)
        return x

class WiderTransformer(BaseTransformer):
    """Wider but shallower transformer architecture"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Wider transformer with fewer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 8,  # Wider FFN
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layers)
        ])
        
        # Initialize transformer weights
        for layer in self.transformer_layers:
            self._init_layer_weights(layer)
    
    def transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_layers:
            x = layer(x)
        return x

class VanillaTransformer(BaseTransformer):
    """Standard transformer implementation"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Standard transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
    
    def transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

class CUDAOptimizedTransformer(BaseTransformer):
    """CUDA-optimized transformer with memory efficient attention"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Register causal mask as buffer for efficiency
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.sequence_length, config.sequence_length),
                diagonal=1
            ).bool(),
            persistent=False
        )
        
        # Memory efficient transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            ) for _ in range(config.n_layers)
        ])
        
        # Initialize transformer weights
        for layer in self.transformer_layers:
            self._init_layer_weights(layer)
    
    def transformer_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_layers:
            x = layer(x, src_mask=self.causal_mask[:x.size(1), :x.size(1)])
        return x

def create_model(config: ModelConfig) -> nn.Module:
    """Factory function for creating models"""
    model_map = {
        ModelType.DEEP_NARROW: DeepNarrowTransformer,
        ModelType.WIDE_SHALLOW: WiderTransformer,
        ModelType.VANILLA: VanillaTransformer,
        ModelType.CUDA_OPTIMIZED: CUDAOptimizedTransformer
    }
    
    model_class = model_map.get(config.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model = model_class(config)
    
    # Move model to appropriate device
    if config.training.use_cuda and torch.cuda.is_available():
        model = model.to(config.training.device)
    
    return model

# Model registry for experiment management
MODEL_REGISTRY = {
    "deep_narrow": {
        "class": DeepNarrowTransformer,
        "default_config": {
            "d_model": 256,
            "n_layers": 12,
            "n_heads": 8
        }
    },
    "wide_shallow": {
        "class": WiderTransformer,
        "default_config": {
            "d_model": 1024,
            "n_layers": 3,
            "n_heads": 16
        }
    },
    "vanilla": {
        "class": VanillaTransformer,
        "default_config": {
            "d_model": 256,
            "n_layers": 6,
            "n_heads": 8
        }
    },
    "cuda_optimized": {
        "class": CUDAOptimizedTransformer,
        "default_config": {
            "d_model": 256,
            "n_layers": 12,
            "n_heads": 8
        }
    }
}