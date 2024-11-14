import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class GrokkingMetrics:
    """Metrics for analyzing grokking phenomena"""
    memorization_epoch: Optional[int] = None
    grokking_epoch: Optional[int] = None
    time_to_grok: Optional[int] = None
    final_generalization_gap: float = 0.0
    max_train_accuracy: float = 0.0
    max_test_accuracy: float = 0.0
    steady_state_epoch: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'memorization_epoch': self.memorization_epoch,
            'grokking_epoch': self.grokking_epoch,
            'time_to_grok': self.time_to_grok,
            'final_generalization_gap': self.final_generalization_gap,
            'max_train_accuracy': self.max_train_accuracy,
            'max_test_accuracy': self.max_test_accuracy,
            'steady_state_epoch': self.steady_state_epoch
        }

class GrokkingExperiment:
    """Analyzes grokking phenomena in training results"""
    
    def __init__(
        self,
        results: Dict[str, Dict],
        memorization_threshold: float = 0.9,
        grokking_threshold: float = 0.9,
        sudden_improvement_threshold: float = 0.1,
        steady_state_window: int = 50
    ):
        self.results = results
        self.memorization_threshold = memorization_threshold
        self.grokking_threshold = grokking_threshold
        self.sudden_improvement_threshold = sudden_improvement_threshold
        self.steady_state_window = steady_state_window
    
    def detect_memorization(self, train_acc: List[float]) -> Optional[int]:
        """Detect when model achieves high training accuracy"""
        for epoch, acc in enumerate(train_acc):
            if acc >= self.memorization_threshold:
                return epoch
        return None
    
    def detect_grokking(
        self,
        train_acc: List[float],
        test_acc: List[float],
        start_epoch: Optional[int] = None
    ) -> Optional[int]:
        """Detect sudden improvement in test accuracy after memorization"""
        if start_epoch is None:
            start_epoch = 0
        
        for epoch in range(start_epoch + 1, len(test_acc)):
            if (test_acc[epoch] - test_acc[epoch-1] >= self.sudden_improvement_threshold and
                test_acc[epoch] >= self.grokking_threshold):
                return epoch
        return None
    
    def detect_steady_state(
        self,
        test_acc: List[float],
        window_size: int = None
    ) -> Optional[int]:
        """Detect when test accuracy stabilizes"""
        if window_size is None:
            window_size = self.steady_state_window
            
        if len(test_acc) < window_size:
            return None
        
        for i in range(len(test_acc) - window_size):
            window = test_acc[i:i+window_size]
            if np.std(window) < 0.01:  # Threshold for stability
                return i
        return None
    
    def analyze_architecture(
        self,
        arch_results: Dict
    ) -> GrokkingMetrics:
        """Analyze grokking metrics for a single architecture"""
        history = arch_results['history']
        train_acc = history['train_accuracy']
        test_acc = history['test_accuracy']
        
        # Detect key epochs
        mem_epoch = self.detect_memorization(train_acc)
        grok_epoch = self.detect_grokking(
            train_acc, test_acc,
            start_epoch=mem_epoch
        ) if mem_epoch is not None else None
        steady_epoch = self.detect_steady_state(test_acc)
        
        # Calculate metrics
        metrics = GrokkingMetrics(
            memorization_epoch=mem_epoch,
            grokking_epoch=grok_epoch,
            time_to_grok=grok_epoch - mem_epoch if all([mem_epoch, grok_epoch]) else None,
            final_generalization_gap=train_acc[-1] - test_acc[-1],
            max_train_accuracy=max(train_acc),
            max_test_accuracy=max(test_acc),
            steady_state_epoch=steady_epoch
        )
        
        return metrics
    
    def analyze(self) -> Dict[str, Dict[str, Any]]:
        """Analyze grokking phenomena across all architectures"""
        analysis = {}
        
        for arch_name, results in self.results.items():
            metrics = self.analyze_architecture(results)
            analysis[arch_name] = metrics.to_dict()
        
        return analysis
    
    def plot_learning_curves(
        self,
        save_path: Optional[Path] = None
    ) -> None:
        """Plot learning curves for all architectures"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        for arch_name, results in self.results.items():
            history = results['history']
            epochs = range(len(history['train_accuracy']))
            
            # Training curves
            ax1.plot(epochs, history['train_accuracy'], label=f'{arch_name} (train)')
            ax1.plot(epochs, history['test_accuracy'], label=f'{arch_name} (test)')
            
            # Generalization gap
            gen_gap = [t - v for t, v in zip(
                history['train_accuracy'],
                history['test_accuracy']
            )]
            ax2.plot(epochs, gen_gap, label=arch_name)
        
        # Customize plots
        ax1.set_title('Learning Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('Generalization Gap')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Train - Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_analysis(
        self,
        output_dir: Path,
        plot: bool = True
    ) -> None:
        """Save analysis results and plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        analysis = self.analyze()
        with open(output_dir / 'grokking_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4)
        
        # Save plots
        if plot:
            self.plot_learning_curves(output_dir / 'learning_curves.png')
        
        # Save summary
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("Grokking Analysis Summary\n")
            f.write("========================\n\n")
            
            for arch_name, metrics in analysis.items():
                f.write(f"\n{arch_name.upper()}\n{'-' * len(arch_name)}\n")
                f.write(f"Memorization epoch: {metrics['memorization_epoch']}\n")
                f.write(f"Grokking epoch: {metrics['grokking_epoch']}\n")
                f.write(f"Time to grok: {metrics['time_to_grok']}\n")
                f.write(f"Final generalization gap: {metrics['final_generalization_gap']:.4f}\n")
                f.write(f"Max train accuracy: {metrics['max_train_accuracy']:.4f}\n")
                f.write(f"Max test accuracy: {metrics['max_test_accuracy']:.4f}\n")
                f.write(f"Steady state epoch: {metrics['steady_state_epoch']}\n\n")