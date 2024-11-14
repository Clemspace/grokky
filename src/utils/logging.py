import logging
import sys
from pathlib import Path
from typing import Optional, Union
import wandb
from datetime import datetime

class ExperimentLogger:
    """Handles logging for experiments with both file and wandb support"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Union[str, Path] = "logs",
        wandb_project: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize wandb if project specified
        self.wandb_enabled = wandb_project is not None
        if self.wandb_enabled:
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                dir=str(self.log_dir)
            )
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to both file and wandb"""
        # Format metrics for logging
        metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        step_str = f"Step {step} - " if step is not None else ""
        self.logger.info(f"{step_str}{metrics_str}")
        
        # Log to wandb if enabled
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
    
    def log_config(self, config: dict):
        """Log configuration parameters"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.wandb_enabled:
            wandb.config.update(config)
    
    def log_model_summary(self, model_summary: str):
        """Log model architecture summary"""
        self.logger.info("\nModel Architecture:")
        self.logger.info(model_summary)
        
        if self.wandb_enabled:
            wandb.run.summary["model_architecture"] = model_summary
    
    def finish(self):
        """Clean up logging"""
        if self.wandb_enabled:
            wandb.finish()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

def setup_logging(experiment_name: str, log_dir: str = "logs", wandb_project: Optional[str] = None) -> ExperimentLogger:
    """Utility function to setup logging for an experiment"""
    return ExperimentLogger(experiment_name, log_dir, wandb_project)