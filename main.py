# main.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import wandb
import argparse
import os
from data import ArithmeticDataset, MetricsTracker, Operation
from models import (
    CUDAOptimizedTransformer,
    CUDATrainer,
    ExperimentConfig,
    DeepNarrowTransformer,
    WarmupCosineScheduler,
    WiderTransformer,
    WandbTracker
)

# Experiment configurations for comparative study
@dataclass
class ComparisonExperimentConfig:
    # Base configuration shared across experiments
    base_config = ExperimentConfig(
        project_name="arithmetic-grokking",
        operations=[Operation.ADDITION, Operation.SUBTRACTION, 
                   Operation.MULTIPLICATION, Operation.DIVISION],
        n_epochs=1000,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.01,
        dropout=0.1,
        modulo=97,
        warmup_steps=100
    )
    
    # Our current deep narrow configuration
    deep_narrow = replace(base_config,
        experiment_name="deep_narrow_baseline",
        model_type="deep_narrow",
        d_model=256,
        n_layers=12,
        n_heads=8
    )
    
    # Wide shallow configuration
    wide_shallow = replace(base_config,
        experiment_name="wide_shallow_comparison",
        model_type="wide_shallow",
        d_model=1024,  # 4x wider
        n_layers=3,    # 1/4 the layers
        n_heads=16     # More heads to utilize wider model
    )
    
    # Vanilla Transformer configuration (using standard PyTorch implementation)
    vanilla = replace(base_config,
        experiment_name="vanilla_transformer",
        model_type="vanilla",
        d_model=256,
        n_layers=6,
        n_heads=8
    )

class VanillaTransformer(nn.Module):
    """Standard PyTorch Transformer implementation"""
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model)
        
        # Standard PyTorch Transformer encoder
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
        
        # Output head
        self.final = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Basic mask for transformer
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1)
        ).to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_encoding(x)
        
        # Transformer layers
        x = self.transformer(x, mask)
        
        # Final prediction
        x = self.final(x)
        
        return x

def run_comparative_study():
    """Run experiments with different architectures"""
    configs = ComparisonExperimentConfig()
    results = {}
    
    # Setup wandb for comparison
    wandb.init(project="arithmetic-grokking-comparison")
    
    try:
        # Run deep narrow experiment
        print("\nRunning Deep Narrow Experiment...")
        model_dn = DeepNarrowTransformer(configs.deep_narrow)
        results['deep_narrow'] = run_experiment(configs.deep_narrow)
        
        # Run wide shallow experiment
        print("\nRunning Wide Shallow Experiment...")
        model_ws = WiderTransformer(configs.wide_shallow)
        results['wide_shallow'] = run_experiment(configs.wide_shallow)
        
        # Run vanilla transformer experiment
        print("\nRunning Vanilla Transformer Experiment...")
        model_v = VanillaTransformer(configs.vanilla)
        results['vanilla'] = run_experiment(configs.vanilla)
        
        # Compare results
        compare_architectures(results)
        
    except Exception as e:
        print(f"Error during comparative study: {str(e)}")
        raise
    finally:
        wandb.finish()
    
    return results

def compare_architectures(results):
    """Analyze and compare results across architectures"""
    print("\nArchitecture Comparison Results")
    print("=" * 80)
    
    metrics = ['accuracy_overall', 'loss_overall']
    architectures = list(results.keys())
    
    # Compare final performance
    print("\nFinal Performance:")
    for arch in architectures:
        print(f"\n{arch.upper()}:")
        for metric in metrics:
            train_val = results[arch][0][f'train_{metric}']
            test_val = results[arch][0][f'test_{metric}']
            print(f"  {metric}:")
            print(f"    Train: {train_val:.4f}")
            print(f"    Test:  {test_val:.4f}")
    
    # Compare training dynamics
    print("\nTraining Dynamics:")
    for arch in architectures:
        print(f"\n{arch.upper()}:")
        train_metrics, test_metrics = results[arch]
        
        # Calculate grokking metrics
        grok_point = detect_grokking(train_metrics, test_metrics)
        if grok_point:
            print(f"  Grokking detected at epoch {grok_point}")
        
        # Analyze per-operation performance
        for op in Operation:
            op_acc = test_metrics.get(f'accuracy_{op.value}')
            if op_acc is not None:
                print(f"  {op.value}: {op_acc:.4f}")
    
    return

def detect_grokking(train_metrics, test_metrics, threshold=0.1):
    """Detect when grokking occurs based on test accuracy jump"""
    train_acc = train_metrics['accuracy_overall']
    test_acc = test_metrics['accuracy_overall']
    
    # Look for sudden improvements in test accuracy
    for i in range(1, len(test_acc)):
        if test_acc[i] - test_acc[i-1] > threshold:
            return i
    
    return None
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    metrics_tracker: MetricsTracker,
    wandb_tracker: Optional[WandbTracker] = None,
    epoch: int = 0
) -> Dict[str, float]:
    model.train()
    metrics_tracker.reset()
    
    for batch_idx, (x, y, operations) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        output = model(x)
        loss = criterion(output[:, -1, :], y.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        predictions = output[:, -1, :].argmax(dim=1)
        metrics_tracker.update(predictions, y.squeeze(), loss.item(), operations)
        
        if wandb_tracker and batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            metrics = metrics_tracker.get_metrics()
            metrics["batch"] = batch_idx
            wandb_tracker.log_metrics(metrics, step)
    
    return metrics_tracker.get_metrics()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    metrics_tracker: MetricsTracker
) -> Dict[str, float]:
    model.eval()
    metrics_tracker.reset()
    
    with torch.no_grad():
        for x, y, operations in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output[:, -1, :], y.squeeze())
            
            predictions = output[:, -1, :].argmax(dim=1)
            metrics_tracker.update(predictions, y.squeeze(), loss.item(), operations)
    
    return metrics_tracker.get_metrics()

def collate_batch(batch):
    """Custom collate function to handle Operation enums in the batch."""
    # Unzip the batch into separate lists
    inputs, targets, operations = zip(*batch)
    
    # Collate tensors normally
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    # Keep operations as a list
    operations = list(operations)
    
    return inputs, targets, operations

def setup_experiment(config: ExperimentConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loaders with custom collate_fn
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=collate_batch  # Add custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        collate_fn=collate_batch  # Add custom collate function
    )
    
    # Model
    model = (DeepNarrowTransformer(config) if config.model_type == "deep_narrow" 
            else WiderTransformer(config))
    model = model.to(device)
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)  # Following Transformer paper
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.n_epochs,
        eta_min=config.learning_rate / 100
    )
    
    return (
        device, model, train_loader, test_loader,
        criterion, optimizer, scheduler
    )

def run_experiment(config: ExperimentConfig) -> Tuple[nn.Module, Dict[str, float], Dict[str, float]]:
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb and metrics
    wandb_tracker = WandbTracker(config)
    metrics_tracker = MetricsTracker(config.operations)
    
    # Print configuration details
    print(f"Model vocabulary size: {config.vocab_size}")
    print(f"Operations: {[op.value for op in config.operations]}")
    
    try:
        # Setup experiment components
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
        
        # Verify vocabulary sizes match
        assert train_dataset.vocab_size == config.vocab_size, \
            f"Vocab size mismatch: dataset={train_dataset.vocab_size}, config={config.vocab_size}"
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Print some sample data
        for i in range(3):
            x, y, op = train_dataset[i]
            print(f"Sample {i}: Input: {x}, Target: {y}, Operation: {op}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_batch
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_batch
        )
        
        # Initialize model
        model = (DeepNarrowTransformer(config) if config.model_type == "deep_narrow"
                else WiderTransformer(config))
        model = model.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Initialize scheduler
        total_steps = config.n_epochs * len(train_loader)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr=config.learning_rate / 100
        )
        
        # Training loop
        best_accuracy = 0.0
        global_step = 0  # Add global step counter
        
        for epoch in tqdm(range(config.n_epochs)):
            try:
                train_metrics = train_epoch(
                    model, train_loader, optimizer, criterion,
                    device, metrics_tracker, wandb_tracker, epoch
                )
                
                test_metrics = evaluate(
                    model, test_loader, criterion,
                    device, metrics_tracker
                )
                
                # Update learning rate with current step
                scheduler.step(global_step)
                global_step += len(train_loader)  # Increment by number of batches
                
                # Log metrics
                if wandb_tracker:
                    wandb_tracker.log_metrics({
                        "epoch": epoch,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"test_{k}": v for k, v in test_metrics.items()},
                        "learning_rate": optimizer.param_groups[0]['lr']  # Log current learning rate
                    }, step=global_step)
                
                # Print progress
                if (epoch + 1) % 100 == 0:
                    print(f"\nEpoch {epoch + 1}")
                    print(f"Train metrics: {train_metrics}")
                    print(f"Test metrics: {test_metrics}")
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
                
            except RuntimeError as e:
                print(f"Error during epoch {epoch}: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise
                
    except Exception as e:
        print(f"Error during experiment: {str(e)}")
        raise
    finally:
        if wandb_tracker:
            wandb_tracker.finish()
    
    return model, train_metrics, test_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Run arithmetic grokking experiments')
    parser.add_argument('--model_type', type=str, default='deep_narrow',
                      choices=['deep_narrow', 'wide_shallow'])
    parser.add_argument('--operations', nargs='+', type=str,
                      default=[op.value for op in Operation])
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--experiment_name', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run arithmetic grokking experiments')
    parser.add_argument('--model_type', type=str, default='deep_narrow',
                      choices=['deep_narrow', 'wide_shallow'])
    parser.add_argument('--operations', nargs='+', type=str,
                      default=[op.value for op in Operation])
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()
    
    # Convert operation strings to Operation enum
    operations = [Operation(op) for op in args.operations]
    
# Create config
    config = ExperimentConfig(
    project_name="arithmetic-grokking",
    experiment_name=args.experiment_name,
    model_type=args.model_type,
    operations=operations,
    n_epochs=args.n_epochs,
    batch_size=max(args.batch_size, 512),
    learning_rate=args.learning_rate,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    use_amp=torch.cuda.is_available(),
    use_cuda=torch.cuda.is_available()
    )
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run the experiment
    model, train_metrics, test_metrics = run_experiment(config)