from typing import Dict

class TrainingCallback:
    """Base class for training callbacks"""
    
    def on_batch_end(self, trainer: 'Trainer', batch: int, logs: Dict[str, float]):
        """Called at the end of a batch"""
        pass
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, float]):
        """Called at the end of an epoch"""
        pass

class ModelCheckpoint(TrainingCallback):
    """Callback to save model checkpoints"""
    
    def __init__(self, filepath: str, monitor: str = "test_accuracy", save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float("-inf")
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.save_best_only:
            if current > self.best_value:
                self.best_value = current
                trainer.save_checkpoint(
                    self.filepath.format(epoch=epoch, **logs),
                    epoch,
                    logs
                )
        else:
            trainer.save_checkpoint(
                self.filepath.format(epoch=epoch, **logs),
                epoch,
                logs
            )

class EarlyStopping(TrainingCallback):
    """Early stopping callback"""
    
    def __init__(self, monitor: str = "test_loss", patience: int = 10, min_delta: float = 0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_value = float("inf")
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, float]):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.model.stop_training = True
