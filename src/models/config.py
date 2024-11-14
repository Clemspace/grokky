from dataclasses import dataclass, field
from typing import Optional
from ..types import BaseConfig, ModelType, TrainingConfig

@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration"""
    model_type: ModelType = ModelType.DEEP_NARROW
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 8
    training: TrainingConfig = field(default_factory=TrainingConfig)

class ModelConfigs:
    """Factory for creating different model configurations"""
    
    @staticmethod
    def deep_narrow() -> ModelConfig:
        return ModelConfig(
            model_type=ModelType.DEEP_NARROW,
            experiment_name="deep_narrow_baseline",
            d_model=256,
            n_layers=12,
            n_heads=8
        )
    
    @staticmethod
    def wide_shallow() -> ModelConfig:
        return ModelConfig(
            model_type=ModelType.WIDE_SHALLOW,
            experiment_name="wide_shallow_baseline",
            d_model=1024,
            n_layers=3,
            n_heads=16
        )
    
    @staticmethod
    def vanilla() -> ModelConfig:
        return ModelConfig(
            model_type=ModelType.VANILLA,
            experiment_name="vanilla_baseline",
            d_model=256,
            n_layers=6,
            n_heads=8
        )
    
    @staticmethod
    def cuda_optimized() -> ModelConfig:
        config = ModelConfig(
            model_type=ModelType.CUDA_OPTIMIZED,
            experiment_name="cuda_optimized",
            d_model=256,
            n_layers=12,
            n_heads=8
        )
        config.training.use_amp = True
        return config

def create_model_config(
    model_type: ModelType,
    experiment_name: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """Factory function for creating model configs with custom parameters"""
    
    # Get base config for model type
    base_configs = {
        ModelType.DEEP_NARROW: ModelConfigs.deep_narrow,
        ModelType.WIDE_SHALLOW: ModelConfigs.wide_shallow,
        ModelType.VANILLA: ModelConfigs.vanilla,
        ModelType.CUDA_OPTIMIZED: ModelConfigs.cuda_optimized
    }
    
    config = base_configs[model_type]()
    
    # Update with custom parameters
    if experiment_name:
        config.experiment_name = experiment_name
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
    
    return config