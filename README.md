---
license: mit
tags:
- sign-language
- computer-vision
- image-classification
- pytorch
- resnet18
- accessibility
- mediapipe
datasets:
- grassknoted/asl-alphabet
language:
- en
metrics:
- accuracy
- precision
- recall
- f1
library_name: pytorch
pipeline_tag: image-classification
---

# Real-Time Sign Language Translator

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![MediaPipe 0.10.31](https://img.shields.io/badge/MediaPipe-0.10.31-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy: 99.60%](https://img.shields.io/badge/Accuracy-99.60%25-brightgreen.svg)](https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator)

## Model Description

A production-ready deep learning system for American Sign Language (ASL) recognition achieving **99.60% accuracy** with **real-time performance (25-30 FPS)** on consumer hardware. The system combines ResNet18 with MediaPipe hand detection for robust gesture classification.

**Key Features:**
- 99.60% test accuracy on 26 ASL letter classes (A-Z)
- Real-time inference at 25-30 FPS on consumer GPU (NVIDIA MX450)
- MediaPipe integration for hand isolation and background removal
- Transfer learning from ImageNet (convergence in 9 epochs)
- Production-ready with comprehensive documentation

## Model Architecture

```
Webcam Input → MediaPipe Hand Detection (~15ms)
                      ↓
              Hand Region Extraction (21 landmarks)
                      ↓
              Preprocessing (224×224, Normalize)
                      ↓
              ResNet18 Model (~13ms)
                      ↓
              Softmax Classification (26 classes)
                      ↓
              Top-3 Predictions + Confidence
```

**ResNet18 + Custom Classification Head:**
```python
Input: 224×224×3 RGB Image
    ↓
ResNet18 Backbone (Pretrained on ImageNet)
├── Conv1: 7×7, 64 filters, stride=2
├── Layer1: 2× Residual Blocks (64 filters)
├── Layer2: 2× Residual Blocks (128 filters)
├── Layer3: 2× Residual Blocks (256 filters)
├── Layer4: 2× Residual Blocks (512 filters)
└── Global Average Pooling → 512-D features
    ↓
Custom Classification Head
├── Dropout(0.5)
├── Linear(512 → 512)
├── ReLU()
├── Dropout(0.3)
└── Linear(512 → 26)
    ↓
Output: 26 class logits (A-Z)
```

**Model Statistics:**
- Total Parameters: 11,452,506
- Model Size: 44 MB
- Trainable Parameters: 100% (full fine-tuning)

## Performance

### Test Set Results (13,050 images)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.60%** |
| Precision (Macro Avg) | 99.59% |
| Recall (Macro Avg) | 99.57% |
| F1-Score (Macro Avg) | 99.58% |
| Correct Predictions | 12,998 / 13,050 |
| Misclassifications | 52 |

### Real-Time Inference

| Configuration | FPS | Latency | Accuracy |
|--------------|-----|---------|----------|
| Without Hand Detection | 76.4 | 13 ms | Poor (~10%) |
| **With MediaPipe** | **25-30** | **30-35 ms** | **Excellent (~95%)** |

### Per-Class Performance

**Perfect Classes (100% F1-Score):**
C, D, F, L, Q, W, Y, Z

**Most Challenging Classes:**
- M: 98.08% (confused with N - visually similar)
- U: 98.79% (confused with V - finger orientation)
- A: 99.12% (confused with E - thumb position)

## Training Details

- **Hardware**: NVIDIA GeForce MX450 (2GB VRAM)
- **Framework**: PyTorch 2.7.1 + CUDA 12.9
- **Epochs**: 9 (manually stopped, best at epoch 8)
- **Optimizer**: Adam (lr=0.001→0.0005, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Batch Size**: 32
- **Training Time**: ~1 hour
- **Best Validation Accuracy**: 99.72% (epoch 8)

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|-------|-----------|-----------|----------|---------|-----|
| 1 | 0.3521 | 89.45% | 0.0893 | 97.21% | 0.001 |
| 2 | 0.0745 | 97.68% | 0.0421 | 98.76% | 0.001 |
| 3 | 0.0412 | 98.71% | 0.0298 | 99.12% | 0.001 |
| 4 | 0.0289 | 99.08% | 0.0234 | 99.34% | 0.001 |
| 5 | 0.0221 | 99.31% | 0.0198 | 99.51% | 0.0005 |
| 6 | 0.0187 | 99.42% | 0.0176 | 99.58% | 0.0005 |
| 7 | 0.0165 | 99.53% | 0.0162 | 99.64% | 0.0005 |
| **8** | **0.0152** | **99.61%** | **0.0151** | **99.72%** | **0.0005** |
| 9 | 0.0143 | 99.67% | 0.0158 | 99.69% | 0.0005 |

## Usage

### Installation

```bash
pip install torch torchvision mediapipe opencv-python numpy pillow
```

### Download Model

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="huzaifanasirrr/realtime-sign-language-translator",
    filename="best_model.pth"
)

# Download MediaPipe hand detector
mediapipe_path = hf_hub_download(
    repo_id="huzaifanasirrr/realtime-sign-language-translator",
    filename="hand_landmarker.task"
)
```

### Load Model

```python
import torch
import torch.nn as nn
from torchvision import models

# Define model architecture
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=26, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageModel(num_classes=26)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded successfully!")
print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
```

### Real-Time Inference with MediaPipe

```python
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from PIL import Image

# Setup MediaPipe hand detector
base_options = python.BaseOptions(model_asset_path=mediapipe_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
hands = vision.HandLandmarker.create_from_options(options)

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class mapping
idx_to_class = {i: chr(65+i) for i in range(26)}  # A-Z

# Capture from webcam
cap = cv2.VideoCapture(0)
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect hand
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hands.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # ~30 FPS
    
    if results.hand_landmarks:
        landmarks = results.hand_landmarks[0]
        
        # Extract hand region
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x_min = max(0, int(min(x_coords)) - 40)
        y_min = max(0, int(min(y_coords)) - 40)
        x_max = min(w, int(max(x_coords)) + 40)
        y_max = min(h, int(max(y_coords)) + 40)
        
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        # Preprocess and predict
        pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
        tensor = preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, dim=1)
            
            predicted_class = idx_to_class[top_idx.item()]
            confidence = top_prob.item() * 100
            
            # Display prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_class}: {confidence:.1f}%", 
                       (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
    
    cv2.imshow('ASL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Dataset

### ASL Alphabet Dataset (Kaggle)

- **Source**: [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Total Images**: 87,000
- **Classes**: 26 letters (A-Z)
- **Format**: RGB, 200×200 pixels
- **Background**: Plain (controlled environment)

### Data Split

```
Total: 87,000 images
├── Training:   60,900 (70%) → 2,342 per class
├── Validation: 13,050 (15%) → 502 per class
└── Test:       13,050 (15%) → 502 per class
```

### Data Augmentation

**Training Augmentation:**
- Random Rotation (±15°)
- Random Affine Translation (10%)
- Random Horizontal Flip (30%)
- Color Jitter (±20% brightness, contrast, saturation)
- ImageNet Normalization

**Impact:** +2.52% validation accuracy improvement

## Model Files

- `best_model.pth` - PyTorch checkpoint (44 MB) - Epoch 8, 99.72% val acc
- `hand_landmarker.task` - MediaPipe hand detection model (7.8 MB)
- `config.json` - Training configuration
- `class_mapping.json` - A-Z class mappings
- `training_history.png` - Training curves visualization
- `confusion_matrix.png` - Test set confusion matrix
- `sample_predictions.png` - Sample predictions with confidence
- `classification_report.txt` - Detailed per-class metrics
- `requirements.txt` - Python dependencies

## Limitations

1. **Static Gestures Only**: Recognizes only static alphabet letters (A-Z), no dynamic gestures
2. **Single Hand**: Processes only one hand at a time, no two-handed signs
3. **Lighting Sensitivity**: Performance degrades in very low light conditions
4. **Hand Orientation**: Expects specific orientations matching training data
5. **Dataset Bias**: Trained on single signer's hands (may not generalize to all hand sizes/skin tones)
6. **No Word Formation**: Letter-by-letter recognition only, no automatic word construction

## Citation

If you use this model in your research, please cite:

```bibtex
@software{nasir2025signlanguage,
  author = {Nasir, Huzaifa},
  title = {Real-Time Sign Language Translator: ResNet18 + MediaPipe for ASL Recognition},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/huzaifanasirrr/realtime-sign-language-translator},
  note = {GitHub: https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator}
}
```

## Academic Report

A comprehensive academic report in Springer LNCS format is available in the [GitHub repository](https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator/blob/main/RealTime-Sign-Language-Translator.pdf).

**Report Highlights:**
- Title: Real-Time Sign Language Recognition using Deep Convolutional Neural Networks and MediaPipe Hand Detection
- Institution: National University of Computer and Emerging Sciences
- Format: Springer Lecture Notes in Computer Science (LNCS)
- Length: 776 lines, comprehensive analysis with 20+ citations

## Author

**Huzaifa Nasir**  
National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan  
📧 nasirhuzaifa95@gmail.com  
🌐 [Portfolio](https://huzaifa-nasir-portfolio.vercel.app)  
🔗 [GitHub Repository](https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator)

## License

MIT License - See [LICENSE](https://github.com/Huzaifanasir95/RealTime-Sign-Language-Translator/blob/main/LICENSE) file for details.

## Acknowledgments

- **Akash (grassknoted)** for the ASL Alphabet dataset on Kaggle
- **Google Research** for MediaPipe hand tracking framework
- **PyTorch Team** for the deep learning framework
- **ResNet18** pretrained on ImageNet (He et al., 2016)
- **FAST-NUCES** for computational resources

Research conducted at FAST-NUCES Islamabad. Inspired by the need for accessible communication tools for the deaf and hard-of-hearing community.

---

**Made with ❤️ for accessibility and inclusion**  
*Breaking communication barriers, one sign at a time*
