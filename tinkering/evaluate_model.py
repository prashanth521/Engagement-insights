import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from model import create_model


def load_model_checkpoint(ckpt_path: str, device):
    """Load trained model from checkpoint."""
    # Try to import torch_directml if available
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print("DirectML detected, loading checkpoint with DirectML support...")
        ckpt = torch.load(ckpt_path, map_location=dml_device)
        # Move tensors to CPU
        ckpt['model_state'] = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                               for k, v in ckpt['model_state'].items()}
    except (ImportError, Exception) as e:
        print(f"DirectML not available or error: {e}")
        print("Attempting to load with weights_only=False...")
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except:
            # Last resort: try loading without any special handling
            print("Trying alternative loading method...")
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    
    num_classes = len(ckpt['classes'])
    arch = ckpt.get('arch', 'resnet18')
    
    model = create_model(num_classes=num_classes, pretrained=False, arch=arch)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    return model, ckpt['classes'], ckpt.get('img_size', 224)


def get_predictions(model, loader, device):
    """Get all predictions and ground truth labels."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs, class_names):
    """Compute comprehensive metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    precision_per_class = precision_score(y_true, y_pred, average=None)
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'f1': float(f1_per_class[i]),
            'recall': float(recall_per_class[i]),
            'precision': float(precision_per_class[i]),
            'sensitivity': float(recall_per_class[i])  # Sensitivity = Recall
        }
    
    # Statistical metrics
    metrics['mean_prediction'] = float(np.mean(y_pred))
    metrics['std_prediction'] = float(np.std(y_pred))
    metrics['mean_confidence'] = float(np.mean(np.max(y_probs, axis=1)))
    metrics['std_confidence'] = float(np.std(np.max(y_probs, axis=1)))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curves(y_true, y_probs, class_names, save_path):
    """Plot ROC curves for all classes."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")


def plot_prediction_heatmap(y_true, y_probs, class_names, save_path):
    """Plot heatmap of prediction probabilities."""
    plt.figure(figsize=(12, 8))
    
    # Sort by true labels for better visualization
    sorted_indices = np.argsort(y_true)
    sorted_probs = y_probs[sorted_indices]
    
    sns.heatmap(sorted_probs.T, cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                yticklabels=class_names, xticklabels=False)
    plt.title('Prediction Probability Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('Classes', fontsize=12)
    plt.xlabel('Samples (sorted by true label)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction heatmap to {save_path}")


def plot_confidence_histogram(y_probs, save_path):
    """Plot histogram of prediction confidence."""
    confidences = np.max(y_probs, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    plt.axvline(np.median(confidences), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(confidences):.3f}')
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Confidence', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence histogram to {save_path}")


def plot_class_distribution(y_true, y_pred, class_names, save_path):
    """Plot comparison of true vs predicted class distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True distribution
    true_counts = np.bincount(y_true, minlength=len(class_names))
    ax1.bar(class_names, true_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Predicted distribution
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    ax2.bar(class_names, pred_counts, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to {save_path}")


def visualize_model_architecture(model, save_path, input_size=(1, 3, 224, 224)):
    """Visualize model architecture using torchviz or torchsummary."""
    try:
        from torchsummary import summary
        import sys
        from io import StringIO
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        summary(model, input_size[1:])
        summary_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Save to text file
        with open(save_path.replace('.png', '.txt'), 'w') as f:
            f.write(summary_str)
        print(f"Saved model architecture summary to {save_path.replace('.png', '.txt')}")
        
    except ImportError:
        print("torchsummary not installed. Saving basic architecture info...")
        with open(save_path.replace('.png', '.txt'), 'w') as f:
            f.write(str(model))
        print(f"Saved basic model architecture to {save_path.replace('.png', '.txt')}")


def ablation_study(model, loader, device, class_names, save_path):
    """Perform ablation study by removing different components."""
    results = {}
    
    # Baseline: Full model
    print("\n=== Ablation Study ===")
    print("Testing full model...")
    y_pred, y_true, y_probs = get_predictions(model, loader, device)
    results['full_model'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # Ablation 1: Remove dropout
    print("Testing without dropout...")
    model_no_dropout = create_model(
        num_classes=len(class_names), 
        pretrained=False, 
        dropout_p=0.0,
        arch=model.config.arch
    )
    model_no_dropout.load_state_dict(model.state_dict(), strict=False)
    model_no_dropout.to(device)
    model_no_dropout.eval()
    
    y_pred, y_true, y_probs = get_predictions(model_no_dropout, loader, device)
    results['no_dropout'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # Ablation 2: Random initialization (no pretrained weights)
    print("Testing without pretrained weights...")
    model_no_pretrain = create_model(
        num_classes=len(class_names), 
        pretrained=False,
        dropout_p=model.config.dropout_p,
        arch=model.config.arch
    )
    model_no_pretrain.to(device)
    model_no_pretrain.eval()
    
    y_pred, y_true, y_probs = get_predictions(model_no_pretrain, loader, device)
    results['no_pretrain'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    
    # Plot ablation results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['accuracy', 'f1_macro', 'recall_macro']
    metric_names = ['Accuracy', 'F1 Score (Macro)', 'Recall (Macro)']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [results[k][metric] for k in ['full_model', 'no_dropout', 'no_pretrain']]
        bars = axes[idx].bar(['Full Model', 'No Dropout', 'No Pretrain'], values, 
                            color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
        axes[idx].set_ylabel(name, fontsize=12)
        axes[idx].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation study to {save_path}")
    
    # Save results to JSON
    json_path = save_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved ablation results to {json_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("test_data", type=str, help="Path to test dataset (ImageFolder format)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ablation", action="store_true", help="Perform ablation study")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, class_names, img_size = load_model_checkpoint(args.checkpoint, device)
    
    # Prepare test data
    print(f"Loading test data from {args.test_data}...")
    test_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = ImageFolder(root=args.test_data, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Get predictions
    print("Getting predictions...")
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs, class_names)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"\nMean Prediction: {metrics['mean_prediction']:.4f}")
    print(f"Std Prediction: {metrics['std_prediction']:.4f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(f"Std Confidence: {metrics['std_confidence']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-"*60)
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  F1 Score: {class_metrics['f1']:.4f}")
        print(f"  Recall/Sensitivity: {class_metrics['recall']:.4f}")
        print(f"  Precision: {class_metrics['precision']:.4f}")
    
    # Save metrics to JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         str(output_dir / "confusion_matrix.png"))
    plot_roc_curves(y_true, y_probs, class_names, 
                   str(output_dir / "roc_curves.png"))
    plot_prediction_heatmap(y_true, y_probs, class_names, 
                           str(output_dir / "prediction_heatmap.png"))
    plot_confidence_histogram(y_probs, 
                             str(output_dir / "confidence_histogram.png"))
    plot_class_distribution(y_true, y_pred, class_names, 
                           str(output_dir / "class_distribution.png"))
    visualize_model_architecture(model, 
                                 str(output_dir / "model_architecture.png"),
                                 input_size=(1, 3, img_size, img_size))
    
    # Ablation study (optional)
    if args.ablation:
        print("\nPerforming ablation study...")
        ablation_study(model, test_loader, device, class_names, 
                      str(output_dir / "ablation_study.png"))
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved classification report to {report_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete! All results saved to:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()

