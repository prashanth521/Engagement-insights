# Engagement Insights: Real-time Engagement Detection System

## 🎯 Project Overview
A deep learning-based system for real-time engagement detection in educational/professional settings. The system classifies engagement levels into four states: Attentive, Distracted, Confused, and Disengaged.

## 🚀 Key Features
- Real-time engagement detection
- Multi-class classification (4 states)
- Pretrained ResNet architecture
- Comprehensive evaluation metrics
- Easy-to-use interface

## 🛠️ Technical Architecture

### Model Architecture
- Base: ResNet (18/50) with transfer learning
- Custom classification head
- Dropout for regularization (p=0.2)
- Input: 224x224 RGB images
- Output: 4-class probability distribution

### Technologies Used
- Python 3.11+
- PyTorch
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## 📊 Datasets
The model is trained on multiple datasets:
1. **AffectNet**: Large-scale facial expression dataset
   - Repurposed for engagement detection
   - Used for both emotion and engagement training

2. **Custom Engagement Dataset**
   - Derived from AffectNet
   - Four engagement states:
     - Attentive
     - Distracted
     - Confused
     - Disengaged

## 📈 Evaluation Metrics

### Classification Metrics
- F1 Score (Macro & Weighted)
- Recall/Sensitivity
- Precision
- Accuracy

### Visualizations
1. Confusion Matrix
2. ROC Curves with AUC scores
3. Prediction Heatmaps
4. Confidence Distribution
5. Class Distribution Analysis

### Model Performance
- Comprehensive evaluation across all engagement states
- Ablation studies to validate architecture choices:
  - Impact of dropout
  - Effect of pretrained weights
  - Architecture comparisons (ResNet18 vs ResNet50)

## 🔧 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/SidhuVenkat/Engagement-Insights.git
cd Engagement-Insights
```

2. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training
```bash
python train.py data/processed_engagement_from_affectnet_folder/train --epochs 50 --batch 32 --arch resnet18
```

### Evaluation
```bash
python evaluate_model.py --model checkpoints/engagement_resnet.pt --data data/processed_engagement_from_affectnet_folder/test
```

### Real-time Detection
```bash
python app.py --model checkpoints/engagement_resnet.pt
```

## 📝 Project Structure
```
├── app.py                 # Real-time detection application
├── model.py              # Model architecture definition
├── train.py              # Training script
├── evaluate_model.py     # Evaluation script
├── requirements.txt      # Dependencies
└── tools/               # Data preparation & utility scripts
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors
- Sidhu Venkat

## 📚 References
- ResNet Paper: "Deep Residual Learning for Image Recognition"
- AffectNet Dataset: "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild"




   Results of the project:
  <img width="4500" height="1500" alt="Image" src="https://github.com/user-attachments/assets/ceea7e9f-93fe-4e78-b0fc-af81bb9bf966" />
  <img width="3000" height="1800" alt="Image" src="https://github.com/user-attachments/assets/93870a64-82b4-46bd-9be0-99a497f26b6d" />

<img width="3000" height="2400" alt="Image" src="https://github.com/user-attachments/assets/a3269f01-c431-4a53-b007-a9f075d825f8" />

<img width="3600" height="2400" alt="Image" src="https://github.com/user-attachments/assets/828afd3d-1a51-4ee4-a3b4-e75ef63fd935" />

<img width="3000" height="2400" alt="Image" src="https://github.com/user-attachments/assets/8e524671-26e2-4530-a099-ccc7c6327e1a" />

<img width="275" height="357" alt="Image" src="https://github.com/user-attachments/assets/af609539-8126-41e4-8cb6-17c5e97e0154" />



  👀👀sample outputs :
  Heuiristic Model prediction and also the Model prediction :
  with the help of EAR ratio as EAR is 0 it detects as eyes are closed and gives prediction as student is in sleepy mode 
  and also by the mouth yawnness also predicts the student is in sleepy mode 
  the three sample outs are here:
  <img width="803" height="640" alt="Image" src="https://github.com/user-attachments/assets/9f2dd749-03c9-445f-b7d8-e1c0eab40092" />

<img width="790" height="632" alt="Image" src="https://github.com/user-attachments/assets/642830cc-68be-43f3-a6f0-68cb8b107a3d" />

<img width="788" height="630" alt="Image" src="https://github.com/user-attachments/assets/e174d119-92bf-4144-bdbb-18361ed371ce" />
  
  
