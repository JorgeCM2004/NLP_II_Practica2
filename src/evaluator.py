from typing import List, Dict, Any, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report)


import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    def __init__(self, label_names: Optional[List[str]] = None) -> None:

        self.label_names = label_names
    
    def compute_metrics(self, y_true: List[Any], y_pred: List[Any], average: str = 'macro') -> Dict[str, float]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average='macro'),
            "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
            "macro_precision": precision_score(y_true, y_pred, average='macro'),
            "macro_recall": recall_score(y_true, y_pred, average='macro'),
        }
        
        if self.label_names:
            per_class_f1 = f1_score(y_true, y_pred, average=None)
            for i, name in enumerate(self.label_names):
                if i < len(per_class_f1):
                    metrics[f"f1_{name}"] = per_class_f1[i]
        
        return metrics
    
    def get_classification_report(self, y_true: List[Any], y_pred: List[Any]) -> str:
        return classification_report(y_true, y_pred, target_names=self.label_names if self.label_names else None)
    
    def plot_confusion_matrix(self,y_true: List[Any],y_pred: List[Any],normalize: bool = True,figsize: Tuple[int, int] = (10, 8),title: str = "Confusion Matrix",cmap: str = "Blues") -> plt.Figure:
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=self.label_names if self.label_names else 'auto',
            yticklabels=self.label_names if self.label_names else 'auto',
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_training_curves(self,history: Dict[str, List[float]],figsize: Tuple[int, int] = (14, 5),title: str = "Training History") -> plt.Figure:

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        if 'train_loss' in history:
            axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if 'train_accuracy' in history:
            axes[1].plot(epochs, history['train_accuracy'], 'b-o', label='Train Acc')
        if 'val_accuracy' in history:
            axes[1].plot(epochs, history['val_accuracy'], 'r-o', label='Val Acc')
        if 'val_macro_f1' in history:
            axes[1].plot(epochs, history['val_macro_f1'], 'g-o', label='Val F1')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Accuracy / F1 Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_alpha_experiment(self,alpha_results: Dict[float, Dict[str, float]],figsize: Tuple[int, int] = (10, 6),title: str = "Hybrid Classifier: Effect of α") -> plt.Figure:

        alphas = list(alpha_results.keys())
        accuracies = [alpha_results[a]['accuracy'] for a in alphas]
        f1_scores = [alpha_results[a]['macro_f1'] for a in alphas]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(alphas))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='coral')
        
        ax.set_xlabel('α (Transformer weight)', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.2f}' for a in alphas])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_comparison_table(self,results: Dict[str, Dict[str, float]],metrics: List[str] = ['accuracy', 'macro_f1']) -> pd.DataFrame:
        data = []
        
        for model_name, model_metrics in results.items():
            row = {'Model': model_name}
            for metric in metrics:
                if metric in model_metrics:
                    row[metric.replace('_', ' ').title()] = model_metrics[metric]
            data.append(row)
        
        df = pd.DataFrame(data)
        
        for col in df.columns[1:]:
            if df[col].dtype in [np.float64, np.float32]:
                df[col] = df[col].apply(lambda x: f'{x:.4f}')
        
        return df
    
    def plot_model_comparison(self,results: Dict[str, Dict[str, float]],metrics: List[str] = ['accuracy', 'macro_f1'],figsize: Tuple[int, int] = (12, 6),title: str = "Model Comparison") -> plt.Figure:

        models = list(results.keys())
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(models))
        width = 0.8 / n_metrics
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
        
        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            offset = (i - n_metrics/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, 
                         label=metric.replace('_', ' ').title(),
                         color=colors[i])
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig
    
    def plot_compression_comparison(self,comparison: Dict[str, Any],figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        models = ['Teacher', 'Student']
        
        accuracy = [comparison['teacher']['accuracy'], 
                   comparison['student']['accuracy']]
        f1 = [comparison['teacher']['macro_f1'], 
              comparison['student']['macro_f1']]
        
        x = np.arange(2)
        width = 0.35
        
        axes[0].bar(x - width/2, accuracy, width, label='Accuracy', color='steelblue')
        axes[0].bar(x + width/2, f1, width, label='F1', color='coral')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].set_ylabel('Score')
        axes[0].set_title('Quality Comparison')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        
        params = [comparison['teacher']['parameters'] / 1e6,
                 comparison['student']['parameters'] / 1e6]
        colors = ['steelblue', 'coral']
        axes[1].bar(models, params, color=colors)
        axes[1].set_ylabel('Parameters (M)')
        axes[1].set_title('Model Size')
        
        for i, v in enumerate(params):
            axes[1].text(i, v + 0.5, f'{v:.1f}M', ha='center')
        
        if 'inference_time' in comparison.get('teacher', {}):
            times = [comparison['teacher']['inference_time'],
                    comparison['student']['inference_time']]
            axes[2].bar(models, times, color=colors)
            axes[2].set_ylabel('Inference Time (s)')
            axes[2].set_title(f'Speed (Speedup: {comparison.get("speedup", 1):.2f}x)')
            
            for i, v in enumerate(times):
                axes[2].text(i, v + 0.1, f'{v:.2f}s', ha='center')
        else:
            axes[2].text(0.5, 0.5, 'No timing data', ha='center', va='center',
                        transform=axes[2].transAxes)
            axes[2].set_title('Speed Comparison')
        
        fig.suptitle('Teacher vs Student Comparison', fontsize=14)
        plt.tight_layout()
        return fig
    
    def save_figure(self,fig: plt.Figure,path: str,dpi: int = 150) -> None:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {path}")
