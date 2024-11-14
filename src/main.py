import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List


from src.models.architectures import DeepNarrowTransformer, WiderTransformer, VanillaTransformer
from src.models.config import ModelType, create_model_config
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logging import setup_logging
from src.analysis.grokking import GrokkingExperiment

def parse_args():
    parser = argparse.ArgumentParser(description='Run comparative study of transformer architectures')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name of the experiment')
    parser.add_argument('--wandb-project', type=str, default='transformer-grokking',
                      help='W&B project name')
    parser.add_argument('--n-epochs', type=int, default=250,
                      help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--early-stopping', action='store_true',
                      help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                      help='Patience for early stopping')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory for saving outputs')
    return parser.parse_args()

def setup_experiment_directories(base_dir: str) -> Dict[str, Path]:
    """Create experiment directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = Path(base_dir) / timestamp
    
    dirs = {
        'root': base_path,
        'checkpoints': base_path / 'checkpoints',
        'logs': base_path / 'logs',
        'results': base_path / 'results'
    }
    
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def create_model(model_type: ModelType, config) -> torch.nn.Module:
    """Factory function to create models"""
    models = {
        ModelType.DEEP_NARROW: DeepNarrowTransformer,
        ModelType.WIDE_SHALLOW: WiderTransformer,
        ModelType.VANILLA: VanillaTransformer
    }
    return models[model_type](config)

def run_architecture_experiment(
    model_type: ModelType,
    args,
    dirs: Dict[str, Path],
    shared_data: Dict
) -> Dict:
    """Run experiment for a single architecture"""
    # Create specific config for this architecture
    config = create_model_config(
        model_type=model_type,
        experiment_name=f"{args.experiment_name}_{model_type.value}",
        n_epochs=args.n_epochs,
        batch_size=args.batch_size
    )
    
    # Setup logging
    logger = setup_logging(
        experiment_name=config.experiment_name,
        log_dir=str(dirs['logs']),
        wandb_project=args.wandb_project
    )
    
    # Log configuration
    logger.log_config(vars(config))
    
    # Create model
    model = create_model(model_type, config)
    logger.log_model_summary(str(model))
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(dirs['checkpoints'] / f"{model_type.value}_best.pt"),
            monitor='test_accuracy',
            save_best_only=True
        )
    ]
    
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='test_loss',
                patience=args.patience
            )
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        logger=logger,
        callbacks=callbacks
    )
    
    # Train model
    history, best_metrics = trainer.train(
        train_loader=shared_data['train_loader'],
        test_loader=shared_data['test_loader']
    )
    
    # Save results
    results = {
        'history': history,
        'best_metrics': best_metrics,
        'config': vars(config)
    }
    
    # Clean up
    logger.finish()
    
    return results

def main():
    args = parse_args()
    
    # Setup directories
    dirs = setup_experiment_directories(args.output_dir)
    
    # Create a base config for dataloaders
    base_config = create_model_config(ModelType.DEEP_NARROW)
    
    # Create dataloaders (shared across experiments)
    train_loader, test_loader = create_dataloaders(base_config)
    
    shared_data = {
        'train_loader': train_loader,
        'test_loader': test_loader
    }
    
    # Set experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"comparative_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run experiments for each architecture
    results = {}
    for model_type in ModelType:
        print(f"\nRunning experiment for {model_type.value} architecture...")
        try:
            results[model_type.value] = run_architecture_experiment(
                model_type=model_type,
                args=args,
                dirs=dirs,
                shared_data=shared_data
            )
            print(f"Completed {model_type.value} experiment successfully!")
        except Exception as e:
            print(f"Error in {model_type.value} experiment: {str(e)}")
            continue
    
    # Analyze grokking phenomena
    print("\nAnalyzing grokking phenomena across architectures...")
    grokking_experiment = GrokkingExperiment(results)
    grokking_analysis = grokking_experiment.analyze()
    
    # Save comparative analysis
    analysis_path = dirs['results'] / 'comparative_analysis.txt'
    with open(analysis_path, 'w') as f:
        f.write("Comparative Analysis Results\n")
        f.write("==========================\n\n")
        
        for architecture, analysis in grokking_analysis.items():
            f.write(f"\n{architecture.upper()} Results:\n")
            f.write(f"Memorization epoch: {analysis['memorization_epoch']}\n")
            f.write(f"Grokking epoch: {analysis['grokking_epoch']}\n")
            f.write(f"Time to grok: {analysis['time_to_grok']}\n")
            f.write(f"Final generalization gap: {analysis['final_generalization_gap']:.4f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nExperiment completed! Results saved to {dirs['root']}")

if __name__ == "__main__":
    main()