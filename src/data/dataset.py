import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
import random

from ..types import Operation
from ..models.config import ModelConfig

class ArithmeticDataset(Dataset):
    """Dataset for arithmetic operations within a modular field"""
    
    def __init__(
        self,
        operations: List[Operation],
        modulo: int = 97,
        split: str = "train",
        train_frac: float = 0.8,
        seed: int = 42
    ):
        self.operations = operations
        self.modulo = modulo
        self.split = split
        random.seed(seed)
        
        # Create operation to index mapping starting from modulo
        self.op_to_idx = {op: idx + self.modulo for idx, op in enumerate(operations)}
        
        # Calculate total vocabulary size (numbers + operations)
        self.vocab_size = self.modulo + len(operations)
        
        print(f"Dataset vocabulary size: {self.vocab_size}")
        print(f"Operation indices: {self.op_to_idx}")
        
        # Generate data for each operation
        self.data = []
        for i in range(modulo):
            for j in range(modulo):
                if Operation.ADDITION in operations:
                    self.data.append((
                        i, j,
                        (i + j) % modulo,
                        Operation.ADDITION
                    ))
                
                if Operation.SUBTRACTION in operations:
                    self.data.append((
                        i, j,
                        (i - j) % modulo,
                        Operation.SUBTRACTION
                    ))
                
                if Operation.MULTIPLICATION in operations:
                    self.data.append((
                        i, j,
                        (i * j) % modulo,
                        Operation.MULTIPLICATION
                    ))
                
                if Operation.DIVISION in operations and j != 0:
                    try:
                        # Find k such that k * j ≡ i (mod modulo)
                        k = (i * pow(j, -1, modulo)) % modulo
                        if (k * j) % modulo == i:
                            self.data.append((i, j, k, Operation.DIVISION))
                    except ValueError:
                        continue
        
        # Shuffle data
        random.shuffle(self.data)
        
        # Split data
        split_idx = int(len(self.data) * train_frac)
        self.data = self.data[:split_idx] if split == "train" else self.data[split_idx:]
        
        print(f"{split} dataset size: {len(self.data)}")

        if len(self.data) == 0:
            raise ValueError("Dataset is empty! Check operation parameters and modulo.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        num1, num2, result, operation = self.data[idx]
        
        # Validate indices
        op_idx = self.op_to_idx[operation]
        if not (0 <= num1 < self.modulo and 0 <= num2 < self.modulo and 0 <= result < self.modulo):
            raise ValueError(f"Number out of range: num1={num1}, num2={num2}, result={result}")
        if not (self.modulo <= op_idx < self.vocab_size):
            raise ValueError(f"Operation index {op_idx} out of range [modulo, vocab_size)")
        
        # Create sequence: [num1, operator, num2]
        x = torch.tensor([num1, op_idx, num2], dtype=torch.long)
        y = torch.tensor([result], dtype=torch.long)
        
        return x, y, operation

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, Operation]]) -> Tuple[torch.Tensor, torch.Tensor, List[Operation]]:
    """Custom collate function for batching"""
    # Unzip the batch
    inputs, targets, operations = zip(*batch)
    
    # Stack tensors
    inputs = torch.stack(inputs)  # Shape: [batch_size, seq_len]
    targets = torch.stack(targets)  # Shape: [batch_size, 1]
    
    # Keep operations as list
    operations = list(operations)
    
    return inputs, targets, operations

def create_dataloaders(config: ModelConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders"""
    
    try:
        # Create datasets
        train_dataset = ArithmeticDataset(
            operations=config.operations,
            modulo=config.modulo,
            split="train",
            train_frac=config.train_frac,
            seed=config.seed
        )
        
        test_dataset = ArithmeticDataset(
            operations=config.operations,
            modulo=config.modulo,
            split="test",
            train_frac=config.train_frac,
            seed=config.seed
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
            num_workers=0  # Set to higher value if multiprocessing is needed
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )
        
        return train_loader, test_loader
    
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        raise

class MetricsTracker:
    """Tracks metrics for different operations"""
    
    def __init__(self, operations: List[Operation]):
        self.operations = operations
        self.reset()
    
    def reset(self):
        self.total_correct = defaultdict(int)
        self.total_samples = defaultdict(int)
        self.total_loss = defaultdict(float)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: float, operations: List[Operation]):
        """Update metrics for the current batch"""
        for pred, target, op in zip(predictions, targets, operations):
            self.total_samples[op] += 1
            self.total_correct[op] += (pred == target).item()
            self.total_loss[op] += loss
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return all metrics"""
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
        
        if total_samples > 0:
            metrics["accuracy_overall"] = total_correct / total_samples
            metrics["loss_overall"] = total_loss / total_samples
        
        return metrics