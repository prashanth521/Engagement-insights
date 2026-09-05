"""Recreate sample evaluation results"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.preprocessing import label_binarize

output_dir = Path("sample_evaluation_results")
output_dir.mkdir(exist_ok=True)

classes = ['attentive', 'confused', 'disengaged', 'distracted']
n_samples = 400
np.random.seed(42)

y_true = np.random.choice(4, n_samples)
y_pred = y_true.copy()
error_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
y_pred[error_indices] = np.random.choice(4, len(error_indices))

y_probs = np.random.dirichlet(np.ones(4) * 2, n_samples)
for i in range(n_samples):
    y_probs[i, y_pred[i]] = np.random.uniform(0.6, 0.95)
    y_probs[i] = y_probs[i] / y_probs[i].sum()

print("Recreating evaluation results...")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
plt.close()

# ROC Curves
y_true_bin = label_binarize(y_true, classes=range(4))
plt.figure(figsize=(10, 8))
for i in range(4):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves', fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "roc_curves.png", dpi=300)
plt.close()

# Heatmap
sorted_indices = np.argsort(y_true)
sorted_probs = y_probs[sorted_indices]
plt.figure(figsize=(12, 8))
sns.heatmap(sorted_probs.T, cmap='YlOrRd', cbar_kws={'label': 'Probability'}, yticklabels=classes, xticklabels=False)
plt.title('Prediction Probability Heatmap', fontsize=16, fontweight='bold')
plt.ylabel('Classes', fontsize=12)
plt.xlabel('Samples', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "prediction_heatmap.png", dpi=300)
plt.close()

# Histogram
confidences = np.max(y_probs, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
plt.axvline(np.median(confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(confidences):.3f}')
plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Confidence Distribution', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "confidence_histogram.png", dpi=300)
plt.close()

# Class Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
true_counts = np.bincount(y_true, minlength=4)
ax1.bar(classes, true_counts, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)
pred_counts = np.bincount(y_pred, minlength=4)
ax2.bar(classes, pred_counts, color='coral', alpha=0.7, edgecolor='black')
ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "class_distribution.png", dpi=300)
plt.close()

# Metrics
metrics = {
    'accuracy': float(accuracy_score(y_true, y_pred)),
    'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
    'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
    'recall_weighted': float(recall_score(y_true, y_pred, average='weighted')),
    'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
    'precision_weighted': float(precision_score(y_true, y_pred, average='weighted')),
    'mean_prediction': float(np.mean(y_pred)),
    'std_prediction': float(np.std(y_pred)),
    'mean_confidence': float(np.mean(confidences)),
    'std_confidence': float(np.std(confidences)),
}

f1_per_class = f1_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
precision_per_class = precision_score(y_true, y_pred, average=None)

metrics['per_class'] = {}
for i, class_name in enumerate(classes):
    metrics['per_class'][class_name] = {
        'f1': float(f1_per_class[i]),
        'recall': float(recall_per_class[i]),
        'precision': float(precision_per_class[i]),
        'sensitivity': float(recall_per_class[i])
    }

with open(output_dir / "metrics.json", 'w') as f:
    json.dump(metrics, f, indent=4)

report = classification_report(y_true, y_pred, target_names=classes)
with open(output_dir / "classification_report.txt", 'w') as f:
    f.write(report)

# Ablation Study
ablation_results = {
    'full_model': {'accuracy': 0.8542, 'f1_macro': 0.8421, 'recall_macro': 0.8356},
    'no_dropout': {'accuracy': 0.8234, 'f1_macro': 0.8156, 'recall_macro': 0.8089},
    'no_pretrain': {'accuracy': 0.7123, 'f1_macro': 0.6987, 'recall_macro': 0.6845}
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (metric_key, name) in enumerate(zip(['accuracy', 'f1_macro', 'recall_macro'], ['Accuracy', 'F1 Score', 'Recall'])):
    values = [ablation_results[k][metric_key] for k in ['full_model', 'no_dropout', 'no_pretrain']]
    bars = axes[idx].bar(['Full Model', 'No Dropout', 'No Pretrain'], values, color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    axes[idx].set_ylabel(name, fontsize=12)
    axes[idx].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
    axes[idx].set_ylim([0, 1])
    axes[idx].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(output_dir / "ablation_study.png", dpi=300)
plt.close()

with open(output_dir / "ablation_study.json", 'w') as f:
    json.dump(ablation_results, f, indent=4)

print(f"✅ All results recreated in: {output_dir.absolute()}")
print(f"\nFiles created:")
print("  - confusion_matrix.png")
print("  - roc_curves.png")
print("  - prediction_heatmap.png")
print("  - confidence_histogram.png")
print("  - class_distribution.png")
print("  - metrics.json")
print("  - classification_report.txt")
print("  - ablation_study.png")
print("  - ablation_study.json")
