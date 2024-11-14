# grokking_study.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from enum import Enum

class ModelArchitecture(Enum):
    DEEP_NARROW = "deep_narrow"
    WIDE_SHALLOW = "wide_shallow"
    VANILLA = "vanilla"

@dataclass
class GrokkingMetrics:
    """Tracks metrics relevant to grokking analysis"""
    train_accuracy: List[float] = field(default_factory=list)
    test_accuracy: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    generalization_gap: List[float] = field(default_factory=list)
    epochs_to_memorization: Optional[int] = None
    epochs_to_grokking: Optional[int] = None
    
    def update(self, train_acc: float, test_acc: float, train_loss: float, test_loss: float):
        self.train_accuracy.append(train_acc)
        self.test_accuracy.append(test_acc)
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)
        self.generalization_gap.append(train_acc - test_acc)

@dataclass
class GrokkingConfig:
    """Configuration for grokking experiments"""
    architecture: ModelArchitecture
    d_model: int
    n_layers: int
    n_heads: int
    learning_rate: float
    weight_decay: float
    n_epochs: int = 400
    batch_size: int = 512
    dropout: float = 0.1
    warmup_steps: int = 100
    # Grokking detection parameters
    memorization_threshold: float = 0.9
    grokking_threshold: float = 0.9
    sudden_improvement_threshold: float = 0.1

class GrokkingAnalyzer:
    """Analyzes training dynamics for signs of grokking"""
    
    def __init__(self, config: GrokkingConfig):
        self.config = config
        self.metrics = GrokkingMetrics()
    
    def detect_memorization_point(self) -> Optional[int]:
        """Detect when model achieves high training accuracy"""
        for epoch, train_acc in enumerate(self.metrics.train_accuracy):
            if train_acc >= self.config.memorization_threshold:
                return epoch
        return None
    
    def detect_grokking_point(self) -> Optional[int]:
        """Detect sudden improvement in test accuracy after memorization"""
        memorization_epoch = self.detect_memorization_point()
        if memorization_epoch is None:
            return None
            
        test_acc = self.metrics.test_accuracy
        for epoch in range(memorization_epoch + 1, len(test_acc)):
            if (test_acc[epoch] - test_acc[epoch-1] >= self.config.sudden_improvement_threshold and
                test_acc[epoch] >= self.config.grokking_threshold):
                return epoch
        return None
    
    def analyze_training_dynamics(self) -> Dict[str, any]:
        """Analyze training patterns and grokking behavior"""
        memorization_epoch = self.detect_memorization_point()
        grokking_epoch = self.detect_grokking_point()
        
        # Calculate key metrics
        final_gen_gap = self.metrics.generalization_gap[-1]
        avg_train_loss = np.mean(self.metrics.train_loss)
        avg_test_loss = np.mean(self.metrics.test_loss)
        
        return {
            "memorization_epoch": memorization_epoch,
            "grokking_epoch": grokking_epoch,
            "time_to_grok": grokking_epoch - memorization_epoch if all([memorization_epoch, grokking_epoch]) else None,
            "final_generalization_gap": final_gen_gap,
            "average_train_loss": avg_train_loss,
            "average_test_loss": avg_test_loss
        }

class GrokkingExperiment:
    """Runs and tracks grokking experiments for different architectures"""
    
    def __init__(self, models: Dict[str, nn.Module], config: GrokkingConfig):
        self.models = models
        self.config = config
        self.analyzers = {name: GrokkingAnalyzer(config) for name in models.keys()}
        self.initialize_wandb()
    
    def initialize_wandb(self):
        """Initialize W&B project for tracking experiments"""
        wandb.init(
            project="transformer-grokking-study",
            config=vars(self.config),
            name=f"comparative-study-{self.config.architecture.value}"
        )
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch and return metrics"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output[:, -1, :], y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = output[:, -1, :].argmax(dim=1)
            correct += (predictions == y.squeeze()).sum().item()
            total += y.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model and return metrics"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y, _ in test_loader:
                output = model(x)
                loss = criterion(output[:, -1, :], y.squeeze())
                
                total_loss += loss.item()
                predictions = output[:, -1, :].argmax(dim=1)
                correct += (predictions == y.squeeze()).sum().item()
                total += y.size(0)
        
        return total_loss / len(test_loader), correct / total
    
    def run_comparative_study(self, train_loader: DataLoader, test_loader: DataLoader) -> Dict[str, Dict]:
        """Run comparative study across different architectures"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name} architecture...")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in tqdm(range(self.config.n_epochs)):
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
                test_loss, test_acc = self.evaluate(model, test_loader, criterion)
                
                # Update metrics
                self.analyzers[name].metrics.update(train_acc, test_acc, train_loss, test_loss)
                
                # Log to W&B
                wandb.log({
                    f"{name}/train_accuracy": train_acc,
                    f"{name}/test_accuracy": test_acc,
                    f"{name}/train_loss": train_loss,
                    f"{name}/test_loss": test_loss,
                    "epoch": epoch
                })
            
            # Analyze results
            results[name] = self.analyzers[name].analyze_training_dynamics()
        
        return results

def create_model_configs() -> List[GrokkingConfig]:
    """Create configurations for different architectures"""
    return [
        GrokkingConfig(
            architecture=ModelArchitecture.DEEP_NARROW,
            d_model=256,
            n_layers=12,
            n_heads=8,
            learning_rate=1e-3,
            weight_decay=0.01
        ),
        GrokkingConfig(
            architecture=ModelArchitecture.WIDE_SHALLOW,
            d_model=1024,
            n_layers=3,
            n_heads=16,
            learning_rate=1e-3,
            weight_decay=0.01
        ),
        GrokkingConfig(
            architecture=ModelArchitecture.VANILLA,
            d_model=256,
            n_layers=6,
            n_heads=8,
            learning_rate=1e-3,
            weight_decay=0.01
        )
    ]