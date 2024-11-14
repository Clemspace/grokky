from typing import List, DefaultDict, Dict
from collections import defaultdict
import torch
from .dataset import Operation


class MetricsTracker:
    def __init__(self, operations: List[Operation]):
        self.operations = operations
        self.reset()
    
    def reset(self):
        self.total_correct: DefaultDict[Operation, int] = defaultdict(int)
        self.total_samples: DefaultDict[Operation, int] = defaultdict(int)
        self.total_loss: DefaultDict[Operation, float] = defaultdict(float)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: float, operations: List[Operation]):
        for pred, target, op in zip(predictions, targets, operations):
            self.total_samples[op] += 1
            self.total_correct[op] += (pred == target).item()
            self.total_loss[op] += loss
            
    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        for op in self.operations:
            if self.total_samples[op] > 0:
                accuracy = self.total_correct[op] / self.total_samples[op]
                avg_loss = self.total_loss[op] / self.total_samples[op]
                metrics[f"accuracy_{op.value}"] = accuracy
                metrics[f"loss_{op.value}"] = avg_loss
                
                total_correct += self.total_correct[op]
                total_samples += self.total_samples[op]
                total_loss += self.total_loss[op]
        
        metrics["accuracy_overall"] = total_correct / total_samples if total_samples > 0 else 0
        metrics["loss_overall"] = total_loss / total_samples if total_samples > 0 else 0
        
        return metrics