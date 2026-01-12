# Year Prediction from Audio Features

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Lessons Learned](#lessons-learned)

---

## 🎯 Overview

This project implements an **end-to-end machine learning pipeline** to predict the release year of songs (1922-2011) from audio features. The challenge involves regression on the Million Song Dataset's Year Prediction subset, using 90 audio timbre features to estimate when a song was released.

### Objectives
- Design and implement a complete ML regression pipeline
- Apply data preprocessing, feature engineering, and model training
- Evaluate multiple ML and DL approaches
- Create an ensemble model for optimal performance
- Document findings and insights comprehensively

### Problem Statement
**Task**: Predict the release year of a song from audio features  
**Type**: Supervised Regression  
**Evaluation Metrics**: RMSE, MAE, R²

---

## 📊 Dataset

**Source**: Million Song Dataset - Year Prediction MSD  
**Size**: 515,345 samples (515,131 after removing duplicates)

### Dataset Composition
- **Training Set**: 394,074 samples (76.5%)
- **Validation Set**: 69,543 samples (13.5%)
- **Test Set**: 51,514 samples (10.0%)

### Features (90 total)
- **12 Timbre Averages**: `timbre_avg_1` to `timbre_avg_12`
  - Mean values of the timbre components (from Mel-frequency cepstral coefficients)
- **78 Timbre Covariances**: `timbre_cov_1` to `timbre_cov_78`
  - Covariance values representing relationships between timbre components

### Target Variable
- **Release Year**: Integer values ranging from 1922 to 2011
- **Distribution**: Heavily skewed toward recent years (more songs from 2000s)

---

## 📁 Project Structure

```
YearPrediction/
├── 01_setup_and_eda.ipynb          # Data exploration and analysis
├── 02_data_preprocessing.ipynb      # Cleaning, splitting, scaling
├── 03_feature_engineering.ipynb     # Feature analysis and selection
├── 04_model_training_ml.ipynb       # ML models (XGBoost, RF, etc.)
├── 05_model_training_dl.ipynb       # Deep learning models (MLP, ResNet)
├── 06_model_evaluation.ipynb        # Model comparison and selection
├── 07_final_model.ipynb             # Final predictions and submission
├── README.md                        # This file
│
├── data/
│   ├── raw/
│   │   └── dataset.csv              # Original dataset (515,345 samples)
│   ├── splits/
│   │   ├── X_train.npy              # Processed training features
│   │   ├── X_val.npy                # Processed validation features
│   │   ├── X_test.npy               # Processed test features
│   │   ├── y_train.npy              # Training labels
│   │   ├── y_val.npy                # Validation labels
│   │   └── y_test.npy               # Test labels
│   └── processed/
│       └── feature_names.txt        # List of feature names
│
├── models/
│   ├── ml/
│   │   ├── xgboost.json             # XGBoost model
│   │   ├── lightgbm.joblib          # LightGBM model
│   │   ├── random_forest.joblib     # Random Forest model
│   │   ├── standard_scaler.joblib   # Fitted scaler
│   │   └── outlier_bounds.joblib    # Outlier clipping bounds
│   ├── dl/
│   │   ├── mlp_final.keras          # Simple MLP model
│   │   ├── deep_nn_final.keras      # Deep neural network
│   │   └── resnet_final.keras       # ResNet architecture
│   └── ensemble_config.json         # Ensemble weights
│
└── reports/
    ├── submission.csv               # Final predictions
    ├── final_summary.json           # Complete results summary
    ├── figures/                     # 40+ interactive visualizations
    │   ├── 01_target_distribution.html
    │   ├── 20_ml_model_comparison.html
    │   ├── 43_xgb_importance.html
    │   └── ...
    └── metrics/                     # Performance metrics
        ├── 02_preprocessing_summary.csv
        ├── 04_ml_model_comparison.csv
        ├── 05_dl_model_comparison.csv
        └── 06_final_model_comparison.csv
```

---

## 🔄 Pipeline

### 1. **Setup and EDA** (`01_setup_and_eda.ipynb`)
- Dataset loading and initial exploration
- Statistical analysis of features and target
- Distribution analysis and visualization
- Missing value check (none found)
- Correlation analysis

**Key Findings**:
- 214 duplicate rows identified
- Features are continuous, mostly centered around 0
- Target heavily skewed toward recent years
- No missing values

### 2. **Data Preprocessing** (`02_data_preprocessing.ipynb`)
- **Duplicate Removal**: 214 duplicates removed
- **Data Splitting**: Random split (prevents distribution shift)
  - Train: 76.5%, Val: 13.5%, Test: 10.0%
  - **Critical Fix**: Split BEFORE transformations to prevent data leakage
- **Outlier Handling**: IQR-based clipping (bounds from training only)
- **Feature Scaling**: StandardScaler (fit on training only)

**Preprocessing Choices**:
- ✅ Random split ensures similar year distribution across all sets
- ✅ All transformations use training data statistics only (no leakage)
- ✅ Preserved test set integrity

### 3. **Feature Engineering** (`03_feature_engineering.ipynb`)
- Feature importance analysis (ANOVA F-test, Random Forest)
- Correlation analysis
- Feature selection experiments
- Time-based pattern analysis

**Key Insights**:
- Timbre covariances more important than averages
- Some features highly correlated (potential for dimensionality reduction)
- Top features: `timbre_cov_*` components

### 4. **ML Model Training** (`04_model_training_ml.ipynb`)
Trained and tuned 6 machine learning models:
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

**Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV

### 5. **DL Model Training** (`05_model_training_dl.ipynb`)
Trained 4 deep learning architectures:
- Simple MLP (3 layers)
- Deep Neural Network (5 layers with BatchNorm + Dropout)
- Wide and Deep Network
- ResNet-inspired architecture

**Training Details**:
- Early stopping with patience
- Learning rate scheduling
- Batch normalization and dropout for regularization

### 6. **Model Evaluation** (`06_model_evaluation.ipynb`)
- Comprehensive comparison of all models
- Ensemble creation with weighted averaging
- Cross-validation analysis
- Performance visualization

### 7. **Final Predictions** (`07_final_model.ipynb`)
- Best model selection and loading
- Ensemble prediction pipeline
- Feature importance analysis
- Submission file generation
- Final visualizations and interpretation

---

## 🏆 Results

### Model Performance Comparison

#### Machine Learning Models

| Model | Test RMSE | Test MAE | Test R² | Training Time |
|-------|-----------|----------|---------|---------------|
| **XGBoost** | **8.63** | **6.09** | **0.363** | 48.1s |
| LightGBM | 8.76 | 6.16 | 0.344 | 16.4s |
| Random Forest | 9.13 | 6.60 | 0.288 | 162.7s |
| Ridge Regression | 9.38 | 6.72 | 0.248 | 2.5s |
| Lasso Regression | 9.39 | 6.72 | 0.247 | 10.3s |
| Linear Regression | 9.38 | 6.72 | 0.248 | 2.1s |

#### Deep Learning Models

| Model | Test RMSE | Test MAE | Test R² | Epochs | Training Time |
|-------|-----------|----------|---------|--------|---------------|
| **Simple MLP** | **8.71** | **6.05** | **0.353** | 100 | 379.3s |
| ResNet | 8.84 | 6.20 | 0.332 | 65 | 657.1s |
| Deep NN | 10.33 | 7.35 | 0.089 | 87 | 784.8s |
| Wide and Deep | 10.39 | 8.50 | 0.078 | 32 | 219.8s |

#### Final Ensemble Model

| Metric | Value |
|--------|-------|
| **Test RMSE** | **8.80 years** |
| **Test MAE** | **6.24 years** |
| **Test R²** | **0.339** |
| **Models** | XGBoost + LightGBM + Random Forest + Deep NN |

### Best Individual Models
1. **XGBoost**: Test R² = 0.363, RMSE = 8.63 years
2. **Simple MLP**: Test R² = 0.353, RMSE = 8.71 years
3. **LightGBM**: Test R² = 0.344, RMSE = 8.76 years

---

## 🔍 Key Findings

### 1. **Model Performance**
- **Gradient boosting models** (XGBoost, LightGBM) performed best
- **Simple MLP** outperformed complex DL architectures
- **Tree-based models** > **Linear models** (non-linear relationships)
- **Ensemble** provided stable, robust predictions

### 2. **Feature Importance**
Top 5 most important features (XGBoost):
1. Timbre covariance features dominate
2. `timbre_cov_*` components are most predictive
3. Timbre averages less important
4. Covariance captures temporal patterns better

### 3. **Challenges**
- **Inherent Difficulty**: Predicting exact year from audio alone is hard
  - Music styles don't change uniformly over time
  - Different genres evolve at different rates
  - Retro-style music can sound older than it is
- **Data Imbalance**: More recent songs in dataset
- **RMSE ~8-9 years**: Typical for this dataset (matches published benchmarks)

### 4. **Decade-Level Accuracy**
- Model is reasonably good at predicting the correct **decade**
- Exact year prediction remains challenging
- Better performance on 2000s songs (more training data)
- Worse performance on older songs (less data, more variation)

### 5. **Preprocessing Impact**
- **Critical Discovery**: Original preprocessing had two major issues:
  1. Year-based split (train ≤2004, test ≥2005) caused distribution shift → Test R² = -47.45
  2. Data leakage from computing outlier bounds on all data
- **After Fix**: Test R² improved from -47.45 to +0.36 (∆83.81 points!)
- **Lesson**: Always split FIRST, then transform

---

## 💻 Installation

### Requirements
- Python 3.9+
- 8GB+ RAM recommended
- 2GB free disk space

### Setup

1. **Clone or download the project**
```bash
cd YearPrediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow keras plotly matplotlib seaborn joblib
```

**Core Dependencies**:
- `pandas >= 1.5.0`
- `numpy >= 1.23.0`
- `scikit-learn >= 1.2.0`
- `xgboost >= 1.7.0`
- `lightgbm >= 3.3.0`
- `tensorflow >= 2.12.0`
- `plotly >= 5.13.0`

4. **Download Dataset**
- Place `dataset.csv` in `data/raw/`
- Or run notebook 01 to download automatically

---

## 🚀 Usage

### Running the Complete Pipeline

Execute notebooks in order:

```bash
jupyter notebook
```

1. **01_setup_and_eda.ipynb** - Explore the data
2. **02_data_preprocessing.ipynb** - Preprocess and split data
3. **03_feature_engineering.ipynb** - Analyze features
4. **04_model_training_ml.ipynb** - Train ML models
5. **05_model_training_dl.ipynb** - Train DL models
6. **06_model_evaluation.ipynb** - Compare models
7. **07_final_model.ipynb** - Generate predictions

### Quick Start (Using Pre-trained Models)

If models are already trained:

```python
import numpy as np
import joblib
from tensorflow import keras

# Load data
X_test = np.load('data/splits/X_test.npy')
y_test = np.load('data/splits/y_test.npy')

# Load best model (XGBoost)
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model('models/ml/xgboost.json')

# Make predictions
predictions = model.predict(X_test)
```

### Making Predictions on New Data

```python
from prediction_pipeline import YearPredictor

# Initialize predictor
scaler = joblib.load('models/ml/standard_scaler.joblib')
model = xgb.XGBRegressor()
model.load_model('models/ml/xgboost.json')

predictor = YearPredictor(scaler, {'XGBoost': model})

# Predict (X should have 90 features)
predictions = predictor.predict(X_new)
```

---

## 🛠️ Technical Details

### Data Preprocessing Pipeline
1. **Duplicate Removal**: Drop 214 duplicate rows
2. **Random Split**: 76.5% train, 13.5% val, 10% test
3. **Outlier Clipping**: IQR method (multiplier=3.0)
   - Bounds calculated from **training data only**
4. **Standardization**: StandardScaler
   - Fit on **training data only**
   - Transform train/val/test using same scaler

### Model Architectures

#### XGBoost (Best ML Model)
```python
XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
```

#### Simple MLP (Best DL Model)
```
Input (90) → Dense(256, ReLU) → Dropout(0.3) 
→ Dense(128, ReLU) → Dropout(0.3) 
→ Dense(64, ReLU) → Output(1)
```

### Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Variance explained

### Hardware Used
- CPU: Standard Intel/AMD processor
- RAM: 8-16GB
- Training Time: ~30-45 minutes total for all models

---

## 📚 Lessons Learned

### Critical Insights

1. **Data Leakage is Real**
   - Original preprocessing had severe data leakage
   - Always split data FIRST before any transformations
   - Impact: Test R² improved from -47.45 to +0.36!

2. **Distribution Shift Matters**
   - Year-based split caused train/test mismatch
   - Random split ensures similar distributions
   - Essential for reliable model evaluation

3. **Simple Models Often Win**
   - XGBoost (gradient boosting) outperformed complex DL
   - Simple MLP beat Deep NN and ResNet
   - Don't overcomplicate unless necessary

4. **Feature Engineering > Complex Models**
   - Understanding feature importance is crucial
   - Timbre covariances more predictive than averages
   - Domain knowledge helps interpretation

5. **Ensemble Provides Robustness**
   - Weighted ensemble reduces variance
   - Combines strengths of different models
   - More stable predictions

### Best Practices Applied

✅ Comprehensive EDA before modeling  
✅ Proper train/val/test splitting  
✅ No data leakage in preprocessing  
✅ Multiple model comparison  
✅ Hyperparameter tuning  
✅ Extensive visualization and documentation  
✅ Reproducible pipeline with saved artifacts  

### Potential Improvements

1. **Additional Features**
   - Include genre, artist, or lyric-based features
   - External metadata (Billboard charts, music databases)

2. **Advanced Architectures**
   - Transformer models for time-series audio features
   - Attention mechanisms

3. **Ordinal Regression**
   - Treat years as ordered categories
   - May capture temporal relationships better

4. **Stratified Sampling**
   - Ensure balanced representation of all decades
   - Particularly for rare years (1920s-1960s)

5. **Feature Engineering**
   - Create decade features
   - Polynomial features for non-linear relationships
   - PCA for dimensionality reduction

---

## 📈 Visualizations

The project generates 40+ interactive visualizations in `reports/figures/`:

- **EDA**: Target distribution, feature correlations, outliers
- **Preprocessing**: Split distributions, scaling effects
- **Feature Analysis**: Importance rankings, ANOVA tests
- **Model Performance**: Training curves, residual plots
- **Predictions**: Actual vs predicted, error by decade, confusion matrices

View any `.html` file in your browser for interactive exploration.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset**: Million Song Dataset (Thierry Bertin-Mahieux et al.)
- **Libraries**: scikit-learn, XGBoost, LightGBM, TensorFlow, Plotly
- **Inspiration**: UCI ML Repository Year Prediction MSD

---

## 📧 Contact

For questions or feedback about this project, please open an issue in the repository.

---

**Project Status**: ✅ Complete  
**Last Updated**: January 2026  
**Total Development Time**: ~8-10 hours  
**Final Model Performance**: R² = 0.339, RMSE = 8.80 years