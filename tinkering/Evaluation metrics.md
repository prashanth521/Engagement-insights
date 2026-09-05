# Model Evaluation Guide

This guide explains how to use the comprehensive model evaluation script (`evaluate_model.py`) to analyze your engagement detection model.

## Features

The evaluation script provides the following metrics and visualizations:

### 📊 Metrics Computed

1. **Classification Metrics**
   - F1 Score (Macro & Weighted)
   - Recall/Sensitivity (Macro & Weighted)
   - Precision (Macro & Weighted)
   - Accuracy

2. **Statistical Metrics**
   - Mean of predictions
   - Standard deviation of predictions
   - Mean confidence scores
   - Standard deviation of confidence scores

3. **Per-Class Metrics**
   - F1 Score for each class
   - Recall/Sensitivity for each class
   - Precision for each class

### 📈 Visualizations Generated

1. **Confusion Matrix** - Shows true vs predicted labels
2. **ROC Curves** - Receiver Operating Characteristic curves for all classes with AUC scores
3. **Prediction Heatmap** - Visualization of prediction probabilities across samples
4. **Confidence Histogram** - Distribution of model confidence scores
5. **Class Distribution** - Comparison of true vs predicted class distributions
6. **Model Architecture** - Text summary of the model structure

### 🔬 Ablation Study

Optional ablation study that tests:
- Full model performance (baseline)
- Model without dropout
- Model without pretrained weights

This helps understand the contribution of different components to model performance.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Evaluation

To run a basic evaluation on your test dataset:

```bash
python evaluate_model.py <checkpoint_path> <test_data_path>
```

**Example:**
```bash
python evaluate_model.py checkpoints/engagement_resnet.pt data/test
```

### With Custom Output Directory

```bash
python evaluate_model.py checkpoints/engagement_resnet.pt data/test --output_dir my_results
```

### With Ablation Study

```bash
python evaluate_model.py checkpoints/engagement_resnet.pt data/test --ablation
```

### Full Command with All Options

```bash
python evaluate_model.py checkpoints/engagement_resnet.pt data/test \
    --output_dir evaluation_results \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --ablation
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `checkpoint` | str | Required | Path to model checkpoint (.pt file) |
| `test_data` | str | Required | Path to test dataset (ImageFolder format) |
| `--output_dir` | str | `evaluation_results` | Directory to save results |
| `--batch_size` | int | 32 | Batch size for evaluation |
| `--num_workers` | int | 4 | Number of data loading workers |
| `--device` | str | auto | Device to use (cuda/cpu) |
| `--ablation` | flag | False | Perform ablation study |

## Test Data Format

The test data should be organized in ImageFolder format:

```
test_data/
├── attentive/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── distracted/
│   ├── image1.jpg
│   └── ...
├── confused/
│   └── ...
└── disengaged/
    └── ...
```

## Output Files

After running the evaluation, you'll find the following files in the output directory:

### Metrics Files
- `metrics.json` - All computed metrics in JSON format
- `classification_report.txt` - Detailed classification report

### Visualization Files
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curves.png` - ROC curves for all classes
- `prediction_heatmap.png` - Heatmap of prediction probabilities
- `confidence_histogram.png` - Distribution of confidence scores
- `class_distribution.png` - True vs predicted class distributions
- `model_architecture.txt` - Model architecture summary

### Ablation Study Files (if --ablation flag used)
- `ablation_study.png` - Comparison of different model configurations
- `ablation_study.json` - Ablation study results in JSON format

## Example Output

When you run the script, you'll see output like:

```
Loading model from checkpoints/engagement_resnet.pt...
Loading test data from data/test...
Getting predictions...
Evaluating: 100%|████████████████████| 32/32 [00:15<00:00,  2.13it/s]

Computing metrics...

============================================================
EVALUATION METRICS
============================================================
Accuracy: 0.8542
F1 Score (Macro): 0.8421
F1 Score (Weighted): 0.8498
Recall (Macro): 0.8356
Recall (Weighted): 0.8542
Precision (Macro): 0.8512
Precision (Weighted): 0.8567

Mean Prediction: 1.4523
Std Prediction: 1.1234
Mean Confidence: 0.8912
Std Confidence: 0.1456

Per-Class Metrics:
------------------------------------------------------------

attentive:
  F1 Score: 0.8923
  Recall/Sensitivity: 0.8756
  Precision: 0.9095

distracted:
  F1 Score: 0.8234
  Recall/Sensitivity: 0.8123
  Precision: 0.8347

confused:
  F1 Score: 0.8156
  Recall/Sensitivity: 0.8034
  Precision: 0.8281

disengaged:
  F1 Score: 0.8371
  Recall/Sensitivity: 0.8512
  Precision: 0.8234

Saved metrics to evaluation_results/metrics.json

Generating visualizations...
Saved confusion matrix to evaluation_results/confusion_matrix.png
Saved ROC curves to evaluation_results/roc_curves.png
Saved prediction heatmap to evaluation_results/prediction_heatmap.png
Saved confidence histogram to evaluation_results/confidence_histogram.png
Saved class distribution to evaluation_results/class_distribution.png
Saved model architecture summary to evaluation_results/model_architecture.txt
Saved classification report to evaluation_results/classification_report.txt

============================================================
Evaluation complete! All results saved to: evaluation_results
============================================================
```

## Understanding the Metrics

### F1 Score
Harmonic mean of precision and recall. Range: 0-1 (higher is better)
- **Macro**: Unweighted mean across all classes
- **Weighted**: Weighted by class support

### Recall/Sensitivity
Proportion of actual positives correctly identified. Range: 0-1 (higher is better)
- Important for ensuring we don't miss positive cases

### Precision
Proportion of predicted positives that are correct. Range: 0-1 (higher is better)
- Important for minimizing false alarms

### ROC-AUC
Area Under the ROC Curve. Range: 0-1 (higher is better)
- 0.5 = random classifier
- 1.0 = perfect classifier

## Tips for Interpretation

1. **Check the confusion matrix** to see which classes are being confused with each other
2. **Review per-class metrics** to identify which classes need improvement
3. **Examine the confidence histogram** to understand model certainty
4. **Use the ablation study** to understand which components contribute most to performance
5. **Compare ROC curves** to see which classes are easier/harder to classify

## Troubleshooting

### CUDA Out of Memory
Reduce batch size: `--batch_size 16`

### Slow Evaluation
Reduce number of workers or use GPU: `--device cuda`

### Missing Dependencies
Install all requirements: `pip install -r requirements.txt`

## Next Steps

After evaluation, you can:
1. Identify weak classes and collect more training data
2. Adjust model architecture based on ablation study results
3. Fine-tune hyperparameters for underperforming classes
4. Use insights to improve data augmentation strategies

## Support

For issues or questions, please refer to the main project documentation.
results:
============================================================
SAMPLE EVALUATION METRICS
============================================================
Accuracy: 0.8900
F1 Score (Macro): 0.8892
Recall (Macro): 0.8909
Precision (Macro): 0.8889

Mean Confidence: 0.5079
Std Confidence: 0.0617

Per-Class Metrics:
------------------------------------------------------------

attentive:
  F1 Score: 0.8629
  Recall/Sensitivity: 0.8947
  Precision: 0.8333

confused:
  F1 Score: 0.8851
  Recall/Sensitivity: 0.8953
  Precision: 0.8750

disengaged:
  F1 Score: 0.9038
  Recall/Sensitivity: 0.9038
  Precision: 0.9038

distracted:
  F1 Score: 0.9050
  F1 Score: 0.9050
  Recall/Sensitivity: 0.8696
  Precision: 0.9434