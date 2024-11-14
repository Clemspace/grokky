import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time
from tqdm import tqdm

from ..models.config import ModelConfig
from ..utils.logging import ExperimentLogger
from .callbacks import TrainingCallback
from .scheduler import WarmupCosineScheduler

class Trainer:
    """Base trainer class with enhanced training capabilities"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        logger: ExperimentLogger,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model.to(config.training.device)
        self.config = config
        self.logger = logger
        self.callbacks = callbacks or []
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = amp.GradScaler(enabled=config.training.use_amp)
        
        # Initialize best metrics
        self.best_metrics = {"accuracy": 0.0, "loss": float("inf")}
        self.current_epoch = 0
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer with parameters from config"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98)
        )
    
    def _setup_scheduler(self) -> WarmupCosineScheduler:
        """Initialize learning rate scheduler"""
        return WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.n_epochs,
            min_lr=self.config.learning_rate / 100
        )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics = {"loss": 0.0, "accuracy": 0.0}
        total_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, (x, y, operations) in enumerate(pbar):
                # Move data to device
                x, y = x.to(self.config.training.device), y.to(self.config.training.device)
                
                # Forward pass with mixed precision
                with amp.autocast(enabled=self.config.training.use_amp):
                    output = self.model(x)
                    loss = self.criterion(output[:, -1, :], y.squeeze())
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )
                
                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = output[:, -1, :].argmax(dim=1)
                    accuracy = (predictions == y.squeeze()).float().mean()
                    
                    # Update running metrics
                    metrics["loss"] += loss.item()
                    metrics["accuracy"] += accuracy.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "accuracy": f"{accuracy.item():.4f}"
                    })
                
                # Call callbacks
                for callback in self.callbacks:
                    callback.on_batch_end(self, batch_idx, {
                        "loss": loss.item(),
                        "accuracy": accuracy.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
        
        # Calculate epoch metrics
        metrics = {k: v / total_batches for k, v in metrics.items()}
        return metrics
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        metrics = {"loss": 0.0, "accuracy": 0.0}
        total_batches = len(test_loader)
        
        with torch.no_grad():
            for x, y, operations in test_loader:
                x, y = x.to(self.config.training.device), y.to(self.config.training.device)
                output = self.model(x)
                loss = self.criterion(output[:, -1, :], y.squeeze())
                
                predictions = output[:, -1, :].argmax(dim=1)
                accuracy = (predictions == y.squeeze()).float().mean()
                
                metrics["loss"] += loss.item()
                metrics["accuracy"] += accuracy.item()
        
        metrics = {k: v / total_batches for k, v in metrics.items()}
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpoint_dir: Optional[str] = None
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """Complete training loop with evaluation"""
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            "train_loss": [], "train_accuracy": [],
            "test_loss": [], "test_accuracy": [],
        }
        
        # Training loop
        for epoch in range(self.config.n_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            test_metrics = self.evaluate(test_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["test_loss"].append(test_metrics["loss"])
            history["test_accuracy"].append(test_metrics["accuracy"])
            
            # Log metrics
            self.logger.log_metrics({
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "test/loss": test_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "learning_rate": self.scheduler.get_last_lr()[0]
            }, step=epoch)
            
            # Save checkpoint if improved
            if test_metrics["accuracy"] > self.best_metrics["accuracy"]:
                self.best_metrics = test_metrics
                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_dir / f"best_model_epoch_{epoch}.pt",
                        epoch,
                        test_metrics
                    )
            
            # Call callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, {
                    **train_metrics,
                    **{"test_" + k: v for k, v in test_metrics.items()}
                })
        
        return history, self.best_metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
