# Engagement Monitor (MVP)

## Project Summary

Engagement Monitor is a real-time computer vision system designed to analyze and classify human engagement levels through facial expression and head pose analysis. The system combines traditional computer vision techniques with deep learning to provide accurate engagement detection across various scenarios including education, workplace environments, and research settings.

### Objectives
1. **Accurate Engagement Detection**: Develop a robust system that can reliably detect and classify different levels of user engagement in real-time.
2. **Accessibility**: Create a solution that works across different hardware configurations, from high-end GPUs to edge devices.
3. **User Privacy**: Implement privacy-focused features to ensure ethical usage and data protection.
4. **Actionable Insights**: Provide meaningful analytics and visualizations to help understand engagement patterns.
5. **Ease of Integration**: Design the system to be easily integrable with existing educational or workplace platforms.
6. **Customization**: Allow for easy adjustment of detection parameters to suit different use cases and environments.
7. **Performance**: Maintain real-time processing capabilities while ensuring high accuracy in engagement classification.

### Key Features
- **Dual-Mode Operation**:
  - **Heuristic Mode**: Uses MediaPipe FaceMesh with rule-based analysis (EAR, MAR, gaze, and head tilt)
  - **ML Mode**: Leverages CNN models (ResNet-18/ResNet-50) for more nuanced engagement classification
- **Four Engagement States**:
  - **Attentive/Interested**: Active engagement
  - **Distracted/Bored**: Diverted attention
  - **Confused**: Signs of confusion or uncertainty
  - **Disengaged/Sleeping**: Complete lack of engagement
- **Cross-Platform Support**: Optimized for various hardware backends (CUDA, DirectML, Intel XPU, CPU)
- **Real-time Performance**: Efficient processing for live video feeds

## Future Enhancements

### 1. Enhanced Model Capabilities
- **Multi-modal Analysis**: Incorporate audio features (speech patterns, tone) and body language analysis
- **Temporal Modeling**: Implement LSTM or Transformer-based models to analyze engagement patterns over time
- **Personalized Baselines**: Adapt to individual users' baseline expressions and engagement patterns
- **Context-Aware Analysis**: Consider contextual factors like time of day, task type, and environmental conditions

### 2. Improved User Experience
- **Interactive Dashboard**: Web-based interface for real-time monitoring and analytics
- **Customizable Alerts**: Configurable notifications for specific engagement patterns
- **Multi-Person Tracking**: Support for analyzing engagement in group settings
- **Privacy-Preserving Features**: On-device processing and privacy filters

### 3. Advanced Analytics
- **Engagement Heatmaps**: Visualize engagement patterns over time
- **Predictive Analytics**: Forecast future engagement levels based on historical data
- **Integration Capabilities**: API endpoints for integration with learning management systems (LMS) or workplace tools
- **Detailed Reporting**: Generate comprehensive engagement reports with actionable insights

### 4. Performance Optimizations
- **Edge Deployment**: Optimize for edge devices and embedded systems
- **Model Quantization**: Reduce model size and improve inference speed
- **Hardware Acceleration**: Enhanced support for various hardware accelerators

### 5. Research & Development
- **Synthetic Data Generation**: Create synthetic datasets for rare engagement scenarios
- **Cross-Cultural Validation**: Ensure model generalizability across different demographics
- **Active Learning**: Continuously improve the model with user feedback

## Setup (Windows/PowerShell)

```powershell
cd tinkering
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (Heuristic only)

```powershell
python app.py
```

## Run (With trained model)

```powershell
python app.py --ckpt checkpoints/engagement_resnet18.pth
```

This overlays both heuristic and model predictions.

## Datasets and Training

You can train on an `ImageFolder` directory:
```
DATA_ROOT/
  attentive/
  distracted/
  confused/
  disengaged/
```
Images can be RGB face crops or full frames (the app will auto-crop face during inference).

### Convert FER-2013 CSV to ImageFolder (demo mapping)
FER-2013 provides emotions; we map emotions to engagement classes heuristically for demo purposes. For research or production, collect/label engagement directly (e.g., DAiSEE) or use a proper mapping.

```powershell
# Place fer2013.csv somewhere, then:
python convert_fer2013.py path\to\fer2013.csv data\fer2013_engaged --usage Training
```

This creates `data/fer2013_engaged/{attentive,distracted,confused}`. (No explicit `disengaged` class in FER; consider augmenting with sleepy/eyes-closed datasets or negatives.)

### Train

```powershell
# ResNet-50 on FER-2013-engagement
python train.py data\fer2013_engaged --arch resnet50 --out checkpoints\engagement_resnet50_fer.pth --epochs 10 --batch 64 --val_split 0.1

# Combine FER-2013-engagement + AffectNet (prepared as ImageFolder with the same 4 class folders)
python train.py data\fer2013_engaged path\to\AffectNet --arch resnet50 --out checkpoints\engagement_resnet50_fer_affectnet.pth --epochs 12 --batch 64 --val_split 0.1
```

- Uses ResNet-18 with ImageNet weights
- Mixed precision on CUDA by default
- Saves best checkpoint with class list and image size

### Inference on a single image

```powershell
python infer.py checkpoints\engagement_resnet50_fer_affectnet.pth path\to\image.jpg
```

## Heuristic Tuning
Edit thresholds in `engagement.py` -> `EngagementConfig`:
- `ear_closed_threshold`
- `mar_yawn_threshold`
- `gaze_forward_min`
- `head_tilt_abs_max_deg`
- `away_duration_seconds`
- `window_seconds`

## Notes & Ethics
- Lighting and camera placement matter.
- FER mapping to engagement is approximate; for reliable results, use engagement-labeled datasets like DAiSEE, EmotiW, or custom annotations.
- Obtain consent and avoid storing raw video by default. Show confidence scores in UI. 