# 🐟 Fish Species Classification using Deep Learning

A comprehensive deep learning project for classifying 31 different species of fish using Convolutional Neural Networks (CNN) with transfer learning.

## 📋 Project Overview

This project implements and compares **6 different CNN architectures** for fish species classification, achieving up to **93.45% validation accuracy** using transfer learning with InceptionV3. The project addresses real-world challenges including class imbalance and limited training data through advanced techniques like data augmentation and class weighting.

### Key Features

- **Multiple CNN Architectures**: Custom CNN, VGG-style, ResNet50, EfficientNetB0, MobileNetV2, and InceptionV3
- **Transfer Learning**: Leveraging pre-trained ImageNet weights for improved performance
- **Class Imbalance Handling**: Computed class weights (11.11:1 ratio) and extensive data augmentation
- **Comprehensive Analysis**: Complete EDA, preprocessing pipeline, and model evaluation
- **Production-Ready**: Modular code structure with saved models and training configurations

## 🎯 Results

### Model Performance Comparison

| Model | Validation Accuracy | Validation Loss | Training Time | Parameters |
|-------|-------------------|-----------------|---------------|------------|
| **InceptionV3** 🏆 | **93.45%** | 0.2156 | 115 min | 23.8M |
| EfficientNetB0 | 91.87% | 0.2634 | 82 min | 5.3M |
| ResNet50 | 89.23% | 0.3421 | 98 min | 25.6M |
| MobileNetV2 | 88.21% | 0.3745 | 65 min | 3.5M |
| VGG-Style CNN | 78.21% | 0.6892 | 72 min | 15.2M |
| Custom CNN | 70.12% | 0.9234 | 45 min | 3.8M |

### Key Findings

- **Transfer learning** significantly outperforms custom architectures (93.45% vs 70.12%)
- **InceptionV3** achieves best performance with its multi-scale feature extraction
- **EfficientNetB0** offers excellent accuracy-to-efficiency ratio (91.87% in 82 min)
- **MobileNetV2** is ideal for resource-constrained deployment (88.21%, lightweight)

## 📊 Dataset

- **Total Images**: 13,331 fish images
- **Classes**: 31 different fish species
- **Train/Val/Test Split**: 8,819 / 2,751 / 1,761 images (66% / 21% / 13%)
- **Image Format**: JPEG, varying dimensions (resized to 224x224 or 299x299)
- **Class Distribution**: Imbalanced (110 - 1,222 images per class)

### Fish Species

```
Bangus, Big Head Carp, Black Sea Sprat, Grass Carp, Hourse Mackerel,
Gilt-Head Bream, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet,
Trout, Green Spotted Puffer, and 18 more species...
```

## 🏗️ Project Structure

```
FishClassification/
├── data/
│   └── raw/
│       └── FishImgDataset/
│           ├── train/          # Training images (8,819)
│           ├── val/            # Validation images (2,751)
│           └── test/           # Test images (1,761)
│
├── models/
│   └── cnn/
│       ├── custom_cnn_best.keras
│       ├── vgg_style_cnn_best.keras
│       ├── resnet50_best.keras
│       ├── efficientnet_b0_best.keras
│       ├── mobilenet_v2_best.keras
│       ├── inception_v3_best.keras
│       └── config.json          # Training configuration
│
├── reports/
│   ├── figures/                 # Training curves & visualizations
│   └── metrics/                 # Performance metrics & history CSVs
│
├── 01_setup_and_eda.ipynb       # Data exploration & analysis
├── 02_data_preprocessing.ipynb  # Preprocessing & augmentation
├── 03_model_training.ipynb      # Model training & comparison
├── Fish_Classification_Complete.ipynb  # Complete Colab notebook
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy, Pandas, Matplotlib, Seaborn
Pillow, scikit-learn
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd FishClassification

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn

# For Google Colab (GPU recommended)
pip install gdown  # For dataset download
```

### Dataset Setup

**Option 1: Manual Upload**
1. Upload `FishImgDataset.zip` to your Google Drive
2. Place it in `/content/drive/MyDrive/FishClassification/data/`
3. The notebook will auto-extract to `data/raw/`

**Option 2: Direct Download**
- Use the provided Google Drive folder ID in the notebook
- Automatic download via `gdown`

### Running the Project

**For Google Colab (Recommended):**
```bash
# Open Fish_Classification_Complete.ipynb
# Mount Google Drive and run cells sequentially
```

**For Local Jupyter:**
```bash
# Run notebooks in order
jupyter notebook 01_setup_and_eda.ipynb
jupyter notebook 02_data_preprocessing.ipynb
jupyter notebook 03_model_training.ipynb
```

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis
- Image property inspection (dimensions, formats)
- Class imbalance detection (11.11:1 ratio)
- Sample visualization across all classes

### 2. Data Preprocessing
- **Image Augmentation**: Rotation (20°), shifts (20%), shear, zoom, horizontal flip, brightness adjustment
- **Class Weights**: Computed to handle imbalance
- **Normalization**: Model-specific preprocessing (ImageNet statistics for transfer learning)
- **Data Generators**: Memory-efficient batch loading with real-time augmentation

### 3. Model Architectures

#### Custom CNN (Baseline)
- 4 convolutional blocks with batch normalization
- Global pooling and dense layers
- Dropout for regularization

#### VGG-Style CNN
- Deeper architecture with multiple 3x3 convolutions
- Progressive channel increase (64 → 512)
- Aggressive dropout (0.25 - 0.5)

#### Transfer Learning Models
- **ResNet50**: Deep residual learning with skip connections
- **EfficientNetB0**: Compound scaling for efficiency
- **MobileNetV2**: Lightweight with inverted residuals
- **InceptionV3**: Multi-scale feature extraction

**Transfer Learning Strategy:**
- Load ImageNet pre-trained weights
- Freeze early layers (keep learned features)
- Fine-tune last 20-30 layers
- Add custom classification head
- Lower learning rate (0.0001 vs 0.001)

### 4. Training Configuration
- **Optimizer**: Adam (lr=0.001 for custom, 0.0001 for transfer learning)
- **Loss**: Categorical cross-entropy
- **Batch Size**: 32
- **Max Epochs**: 50
- **Callbacks**: 
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save best model)

## 📈 Training Insights

### Learning Curves Analysis

1. **InceptionV3**: Smooth convergence, minimal overfitting, best generalization
2. **EfficientNetB0**: Fast convergence, stable training, excellent efficiency
3. **ResNet50**: Strong performance, slight overfitting controlled by dropout
4. **MobileNetV2**: Quick training, good for real-time applications
5. **VGG-Style**: Moderate performance, prone to overfitting without augmentation
6. **Custom CNN**: Baseline performance, limited capacity for complex features

### Overfitting Mitigation
- Data augmentation (7x effective dataset size)
- Dropout layers (0.25 - 0.5)
- Batch normalization
- Early stopping
- L2 regularization in dense layers

## 🎓 Technical Highlights

- **GPU Acceleration**: TensorFlow GPU support with memory growth
- **Memory Efficiency**: ImageDataGenerator for batch loading
- **Reproducibility**: Fixed random seeds (42)
- **Model Checkpointing**: Save best models during training
- **Comprehensive Logging**: JSON history and CSV summaries

## 📝 Key Learnings

1. **Transfer Learning is Essential**: Pre-trained models achieve 23% better accuracy than custom architectures
2. **Architecture Matters**: Inception's multi-scale approach excels at fine-grained classification
3. **Data Augmentation Works**: Critical for handling class imbalance and limited data
4. **Efficiency Trade-offs**: MobileNetV2 offers 88% accuracy with 3.5M params vs InceptionV3's 93% with 23M params

## 🔮 Future Improvements

- [ ] Ensemble methods (soft/hard voting)
- [ ] Advanced augmentation (Mixup, CutMix)
- [ ] Attention mechanisms (SE blocks, CBAM)
- [ ] Model compression (quantization, pruning)
- [ ] Test-time augmentation (TTA)
- [ ] Learning rate scheduling experiments
- [ ] Focal loss for extreme imbalance

## 📦 Model Deployment

Saved models can be loaded and used for inference:

```python
from tensorflow import keras

# Load best model
model = keras.models.load_model('models/cnn/inception_v3_best.keras')

# Predict on new image
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('new_fish.jpg', target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# InceptionV3 preprocessing
from tensorflow.keras.applications.inception_v3 import preprocess_input
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted: {CLASSES[class_idx]} ({confidence:.2%})")
```

## 📚 References

- He et al. (2016) - Deep Residual Learning for Image Recognition
- Szegedy et al. (2016) - Rethinking the Inception Architecture
- Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
- Sandler et al. (2018) - MobileNetV2: Inverted Residuals and Linear Bottlenecks

## 👤 Author

Machine Learning Project - Deep Learning for Computer Vision

## 📄 License

This project is for educational purposes.

---

**Note**: This project demonstrates best practices in deep learning including data preprocessing, transfer learning, model comparison, and comprehensive evaluation. The modular structure and detailed documentation make it suitable for learning and adaptation to other image classification tasks.
