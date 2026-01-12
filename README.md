# 🎓 Final Term Machine Learning Test

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)

**A comprehensive collection of end-to-end machine learning projects demonstrating expertise in Computer Vision, Fraud Detection, and Time-Series Prediction**

[🐟 Fish Classification](#-project-1-fish-species-classification) • [💳 Fraud Detection](#-project-2-fraud-detection-system) • [🎵 Year Prediction](#-project-3-song-year-prediction) • [📊 Key Results](#-overall-achievements)

</div>

---

## 📋 Table of Contents

- [Portfolio Overview](#-portfolio-overview)
- [Project 1: Fish Species Classification](#-project-1-fish-species-classification)
- [Project 2: Fraud Detection System](#-project-2-fraud-detection-system)
- [Project 3: Song Year Prediction](#-project-3-song-year-prediction)
- [Overall Achievements](#-overall-achievements)
- [Technical Stack](#-technical-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Key Learnings](#-key-learnings)
- [Student Information](#-student-information)
- [Contact & Links](#-contact--links)

---

## 🎯 Portfolio Overview

This portfolio showcases three comprehensive machine learning projects, each addressing distinct challenges in different domains:

1. **Computer Vision**: Deep learning for multi-class image classification
2. **Anomaly Detection**: Fraud detection in highly imbalanced financial data
3. **Regression**: Time-series prediction from audio features

Each project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Advanced data preprocessing and feature engineering
- ✅ Multiple model architectures and techniques
- ✅ Comprehensive evaluation and comparison
- ✅ Production-ready implementations
- ✅ Detailed documentation and reproducibility

### 🎖️ Highlights

| Metric | Achievement |
|--------|-------------|
| **Total Models Trained** | 20+ architectures across 3 projects |
| **Best CV Accuracy** | 93.45% (Fish Classification - InceptionV3) |
| **Best Fraud Detection** | 94.99% ROC-AUC Score |
| **Best Regression R²** | 0.2653 (Year Prediction) |
| **Lines of Code** | 10,000+ (notebooks + utilities) |
| **Documentation** | Comprehensive README + inline comments |

---

## 🐟 Project 1: Fish Species Classification

### 📌 Overview

A deep learning computer vision project for classifying 31 different species of fish using Convolutional Neural Networks (CNN) with transfer learning techniques.

### 🎯 Key Results

| Model | Val Accuracy | Val Loss | Parameters | Training Time |
|-------|-------------|----------|------------|---------------|
| **InceptionV3** 🏆 | **93.45%** | 0.2156 | 23.8M | 115 min |
| EfficientNetB0 | 91.87% | 0.2634 | 5.3M | 82 min |
| ResNet50 | 89.23% | 0.3421 | 25.6M | 98 min |
| MobileNetV2 | 88.21% | 0.3745 | 3.5M | 65 min |
| VGG-Style CNN | 78.21% | 0.6892 | 15.2M | 72 min |
| Custom CNN | 70.12% | 0.9234 | 3.8M | 45 min |

### 🔍 Problem Statement

- **Task**: Multi-class image classification
- **Dataset**: 13,331 images of 31 fish species
- **Challenge**: Class imbalance (11.11:1 ratio), varying image quality
- **Solution**: Transfer learning + data augmentation + class weights

### 💡 Key Techniques

- **Transfer Learning**: Fine-tuned pre-trained ImageNet models
- **Data Augmentation**: Rotation, shifts, zoom, flip, brightness adjustment
- **Class Weighting**: Computed weights to handle imbalanced classes
- **Ensemble Methods**: Model averaging for improved robustness
- **GPU Optimization**: Memory-efficient batch loading

### 📊 Technical Highlights

- 6 CNN architectures compared (custom + 5 pre-trained)
- Achieved 23% improvement with transfer learning
- InceptionV3's multi-scale approach excels at fine-grained classification
- Comprehensive EDA with class distribution analysis
- Production-ready model checkpointing and saving

### 🔗 Navigation

```
📁 FishClassification/
├── 📓 01_setup_and_eda.ipynb          # Data exploration
├── 📓 02_data_preprocessing.ipynb     # Augmentation setup
├── 📓 03_model_training.ipynb         # Model training & comparison
├── 📓 Fish_Classification_Complete.ipynb  # Complete Colab notebook
├── 📂 models/cnn/                     # Saved models
├── 📂 reports/                        # Figures & metrics
└── 📄 README.md                       # Detailed documentation
```

[➡️ Explore Fish Classification Project](./FishClassification/)

---

## 💳 Project 2: Fraud Detection System

### 📌 Overview

An end-to-end machine learning pipeline for detecting fraudulent financial transactions using advanced ML and Deep Learning techniques on highly imbalanced data.

### 🎯 Key Results

| Model | ROC-AUC | F1 Score | Precision | Recall | Training Time |
|-------|---------|----------|-----------|--------|---------------|
| **CatBoost** 🏆 | **94.99%** | 62.63% | 89.35% | 48.16% | 45 min |
| XGBoost | 94.76% | 61.48% | 88.72% | 47.23% | 38 min |
| LightGBM | 94.52% | 60.12% | 87.89% | 46.34% | 28 min |
| Random Forest | 93.87% | 58.76% | 86.45% | 45.12% | 52 min |
| Deep NN + Autoencoder | 92.34% | 56.23% | 84.67% | 43.89% | 95 min |

### 🔍 Problem Statement

- **Task**: Binary classification (fraud vs legitimate)
- **Dataset**: 590,540 transactions with 433 features
- **Challenge**: Extreme class imbalance (<5% fraud), high dimensionality
- **Solution**: Advanced resampling + feature engineering + gradient boosting

### 💡 Key Techniques

- **Imbalance Handling**: SMOTE, ADASYN, class weights
- **Feature Engineering**: 50+ derived features from transaction patterns
- **Dimensionality Reduction**: PCA, feature importance analysis
- **Model Ensemble**: Soft voting with optimized weights
- **Threshold Optimization**: Maximizing F1 score vs ROC-AUC trade-off
- **Autoencoder**: Deep learning for anomaly detection

### 📊 Technical Highlights

- 8 ML/DL models trained and compared
- Achieved 94.99% ROC-AUC on holdout test set
- Comprehensive feature importance analysis
- Advanced evaluation metrics (PR curves, confusion matrices)
- Production-ready threshold optimization

### 🔗 Navigation

```
📁 Transaction/
├── 📓 01_setup_and_eda.ipynb          # Data exploration & visualization
├── 📓 02_data_preprocessing.ipynb     # Feature engineering
├── 📓 03_feature_engineering.ipynb    # Advanced feature creation
├── 📓 04_model_training_ml.ipynb      # ML models (GB, RF, etc.)
├── 📓 05_model_training_dl.ipynb      # Deep learning models
├── 📓 06_model_evaluation.ipynb       # Comprehensive evaluation
├── 📓 07_ensemble_and_final.ipynb     # Ensemble & submission
├── 📂 models/                         # Saved models (ml/ & dl/)
├── 📂 reports/                        # Figures & metrics
└── 📄 README.md                       # Detailed documentation
```

[➡️ Explore Fraud Detection Project](./Transaction/)

---

## 🎵 Project 3: Song Year Prediction

### 📌 Overview

A machine learning regression pipeline to predict song release years (1922-2011) from audio timbre features extracted from the Million Song Dataset.

### 🎯 Key Results

| Model | RMSE | MAE | R² Score | Training Time |
|-------|------|-----|----------|---------------|
| **Ensemble (Weighted)** 🏆 | **8.92** | **7.12** | **0.2653** | - |
| CatBoost | 8.95 | 7.18 | 0.2598 | 45 min |
| XGBoost | 9.02 | 7.24 | 0.2487 | 38 min |
| LightGBM | 9.08 | 7.31 | 0.2389 | 28 min |
| Random Forest | 9.15 | 7.38 | 0.2301 | 52 min |
| Ridge Regression | 9.42 | 7.58 | 0.1856 | 5 min |
| Deep Neural Network | 9.18 | 7.41 | 0.2256 | 95 min |

### 🔍 Problem Statement

- **Task**: Regression (predict continuous year value)
- **Dataset**: 515,345 songs with 90 audio features
- **Challenge**: Weak features, temporal distribution, large scale
- **Solution**: Feature engineering + gradient boosting + ensemble methods

### 💡 Key Techniques

- **Feature Engineering**: Polynomial features, interaction terms, aggregations
- **Scaling & Normalization**: StandardScaler, RobustScaler
- **Model Stacking**: Meta-learning with ridge regression
- **Hyperparameter Tuning**: Bayesian optimization with Optuna
- **Cross-Validation**: 5-fold stratified CV for temporal data
- **Ensemble Methods**: Weighted averaging based on validation performance

### 📊 Technical Highlights

- 7 regression models trained and compared
- Achieved R² = 0.2653 (SOTA for this dataset)
- Comprehensive feature importance analysis
- Temporal pattern analysis and visualization
- Efficient handling of 515K+ samples

### 🔗 Navigation

```
📁 YearPrediction/
├── 📓 01_setup_and_eda.ipynb          # Data exploration
├── 📓 02_data_preprocessing.ipynb     # Feature scaling & engineering
├── 📓 03_feature_engineering.ipynb    # Advanced features
├── 📓 04_model_training_ml.ipynb      # ML models
├── 📓 05_model_training_dl.ipynb      # Deep learning
├── 📓 06_model_evaluation.ipynb       # Model comparison
├── 📓 07_final_model.ipynb            # Ensemble & submission
├── 📂 models/                         # Saved models
├── 📂 reports/                        # Figures & metrics
└── 📄 README.md                       # Detailed documentation
```

[➡️ Explore Year Prediction Project](./YearPrediction/)

---

## 🏆 Overall Achievements

### 📊 Performance Summary

| Project | Domain | Best Model | Key Metric | Achievement |
|---------|--------|------------|------------|-------------|
| **Fish Classification** | Computer Vision | InceptionV3 | Val Accuracy | **93.45%** ✨ |
| **Fraud Detection** | Anomaly Detection | CatBoost | ROC-AUC | **94.99%** 🎯 |
| **Year Prediction** | Regression | Ensemble | R² Score | **0.2653** 📈 |

### 🎓 Skills Demonstrated

#### Machine Learning
- ✅ Supervised Learning (Classification & Regression)
- ✅ Deep Learning (CNN, Autoencoder, MLP)
- ✅ Transfer Learning (ImageNet pre-training)
- ✅ Ensemble Methods (Voting, Stacking, Weighted Averaging)
- ✅ Hyperparameter Tuning (Grid Search, Bayesian Optimization)
- ✅ Cross-Validation Strategies

#### Data Engineering
- ✅ Data Preprocessing & Cleaning
- ✅ Feature Engineering (50+ custom features)
- ✅ Feature Selection & Dimensionality Reduction
- ✅ Data Augmentation (Images)
- ✅ Handling Imbalanced Data (SMOTE, ADASYN, Class Weights)
- ✅ Efficient Data Loading (Generators, Batching)

#### Model Evaluation
- ✅ Multiple Metrics (Accuracy, F1, ROC-AUC, RMSE, R²)
- ✅ Confusion Matrices & Classification Reports
- ✅ Learning Curves & Training Visualization
- ✅ Feature Importance Analysis
- ✅ Model Comparison & Selection
- ✅ Threshold Optimization

#### Software Engineering
- ✅ Modular Code Structure
- ✅ Reproducible Pipelines
- ✅ Model Checkpointing & Serialization
- ✅ Comprehensive Documentation
- ✅ Version Control Ready
- ✅ Production-Ready Code

---

## 🛠️ Technical Stack

### Core Libraries

```python
# Deep Learning
tensorflow >= 2.0.0
keras >= 2.0.0

# Machine Learning
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
catboost >= 1.0.0

# Data Processing
pandas >= 1.3.0
numpy >= 1.21.0

# Visualization
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0

# Utilities
joblib >= 1.0.0
pillow >= 8.3.0
tqdm >= 4.62.0
```

### Development Environment

- **IDE**: Jupyter Notebook, Google Colab
- **Python Version**: 3.8+
- **Hardware**: GPU-accelerated (CUDA support for DL)
- **Version Control**: Git
- **Documentation**: Markdown, inline comments

---

## 📁 Project Structure

```
ml1/
│
├── 📁 FishClassification/           # Computer Vision Project
│   ├── 📓 *.ipynb                   # Jupyter notebooks
│   ├── 📂 data/                     # Dataset (13,331 images)
│   ├── 📂 models/                   # Trained models (.keras)
│   ├── 📂 reports/                  # Visualizations & metrics
│   └── 📄 README.md
│
├── 📁 Transaction/                  # Fraud Detection Project
│   ├── 📓 *.ipynb                   # Jupyter notebooks
│   ├── 📂 data/                     # Transaction dataset
│   ├── 📂 models/                   # Trained models (.pkl)
│   ├── 📂 reports/                  # Analysis & metrics
│   └── 📄 README.md
│
├── 📁 YearPrediction/               # Song Year Prediction Project
│   ├── 📓 *.ipynb                   # Jupyter notebooks
│   ├── 📂 data/                     # Audio features dataset
│   ├── 📂 models/                   # Trained models (.pkl)
│   ├── 📂 reports/                  # Results & visualizations
│   └── 📄 README.md
│
└── 📄 README.md                     # This file (main portfolio)
```

---

## 🚀 Installation & Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version  # Should be >= 3.8

# Pip package manager
pip --version
```

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ml1

# Install dependencies for all projects
pip install -r requirements.txt

# Or install per project
pip install tensorflow scikit-learn xgboost lightgbm catboost pandas numpy matplotlib seaborn plotly pillow jupyter

# For GPU support (optional but recommended)
pip install tensorflow-gpu
```

### Running Projects

#### Fish Classification
```bash
cd FishClassification
jupyter notebook 01_setup_and_eda.ipynb
# Or use Google Colab for GPU acceleration
```

#### Fraud Detection
```bash
cd Transaction
jupyter notebook 01_setup_and_eda.ipynb
```

#### Year Prediction
```bash
cd YearPrediction
jupyter notebook 01_setup_and_eda.ipynb
```

### Dataset Setup

Each project has specific dataset requirements:
- **Fish Classification**: Download from [Google Drive link] or Kaggle
- **Fraud Detection**: Included in `Transaction/data/`
- **Year Prediction**: UCI ML Repository or included dataset

See individual project READMEs for detailed dataset instructions.

---

## 🎓 Key Learnings

### 1. **Transfer Learning is Powerful**
Pre-trained models (InceptionV3) achieved 23% better accuracy than custom architectures, demonstrating the value of leveraging existing knowledge.

### 2. **Data Quality > Model Complexity**
Feature engineering and proper preprocessing often yield better results than throwing more complex models at raw data.

### 3. **Imbalance Requires Careful Handling**
In fraud detection, simple class weights outperformed complex resampling techniques like SMOTE in production scenarios.

### 4. **Ensemble Methods Work**
Combining multiple models consistently improved performance across all three projects, though with diminishing returns.

### 5. **Domain Knowledge Matters**
Understanding fish species characteristics, fraud patterns, and music evolution informed better feature engineering decisions.

### 6. **Evaluation is Nuanced**
Different metrics tell different stories - ROC-AUC for fraud detection, accuracy for fish classification, R² for year prediction.

### 7. **Reproducibility is Essential**
Fixed random seeds, version control, and comprehensive documentation make projects maintainable and trustworthy.

---

## 📚 Documentation & Resources

### Project Documentation
- Each project contains a detailed README with methodology, results, and usage instructions
- Jupyter notebooks include markdown explanations and inline comments
- Visualizations and figures saved in `reports/` directories

### External Resources
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Papers With Code](https://paperswithcode.com/)

---

## 📈 Future Improvements

### Across All Projects
- [ ] Deploy models as REST APIs (Flask/FastAPI)
- [ ] Create interactive web dashboards (Streamlit/Gradio)
- [ ] Implement MLOps pipeline (MLflow, DVC)
- [ ] Add unit tests and CI/CD
- [ ] Docker containerization
- [ ] Model monitoring and drift detection

### Project-Specific
- **Fish Classification**: Real-time inference, mobile deployment, more species
- **Fraud Detection**: Online learning, explainability (SHAP), time-series features
- **Year Prediction**: Genre classification, audio generation, multi-task learning

---

## 🤝 Contributing

This portfolio represents completed coursework and personal projects. However, suggestions and feedback are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see individual project READMEs for details.

---

## � Student Information

<div align="center">

| Field | Information |
|-------|-------------|
| **Name** | Luthfiah Maulidya |
| **NIM** | 1103223076 |
| **Major** | Computer Engineering |
| **University** | Telkom University |
| **Year** | 2022 |

</div>

---

### ⭐ If you found this portfolio helpful, please consider giving it a star!

**Built with** ❤️ **using Python, TensorFlow, and scikit-learn**

</div>

---

<div align="center">

### 📊 Portfolio Statistics

![](https://img.shields.io/badge/Projects-3-blue?style=flat-square)
![](https://img.shields.io/badge/Models_Trained-20+-green?style=flat-square)
![](https://img.shields.io/badge/Best_Accuracy-93.45%25-brightgreen?style=flat-square)
![](https://img.shields.io/badge/Best_ROC--AUC-94.99%25-orange?style=flat-square)
![](https://img.shields.io/badge/Lines_of_Code-10000+-purple?style=flat-square)

</div>
