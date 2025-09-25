# Kepler Exoplanet Dataset - ML Guide

## 🚀 NASA Space Apps Challenge - Exoplanet Detection Project

This repository contains the processed Kepler dataset ready for machine learning classification to identify exoplanets, candidates, and false positives.

## 📋 Dataset Overview

- **Original Dataset**: 9,564 observations from NASA Kepler mission
- **Processed Dataset**: 9,201 observations (3.8% removed due to excessive missing data)
- **Features**: 127 numeric features for ML
- **Target Classes**: 3 classes (CONFIRMED, CANDIDATE, FALSE_POSITIVE)
- **Class Distribution**: Reasonably balanced (29.8% confirmed, 20.4% candidates, 49.8% false positives)

## 📁 File Structure

```
├── data/
│   ├── processed/
│   │   ├── kepler_train.csv          # Training set (60% - 5,520 samples)
│   │   ├── kepler_val.csv            # Validation set (20% - 1,840 samples)
│   │   ├── kepler_test.csv           # Test set (20% - 1,841 samples)
│   │   ├── kepler_train_minmax.csv   # MinMax scaled training (for tree models)
│   │   ├── kepler_val_minmax.csv     # MinMax scaled validation
│   │   ├── kepler_test_minmax.csv    # MinMax scaled test
│   │   └── kepler_full_processed.csv # Complete processed dataset
├── artifacts/
│   ├── standard_scaler.pkl          # StandardScaler object
│   ├── minmax_scaler.pkl           # MinMaxScaler object
│   ├── label_encoder.pkl           # Target label encoder
│   └── feature_metadata.pkl       # Feature names and metadata
└── keplar_processing.ipynb         # Complete preprocessing pipeline
```

## 🎯 Target Variable

- **Column**: `target` (encoded) and `target_name` (original labels)
- **Classes**:
  - `0` / `CANDIDATE`: Potential exoplanet requiring further verification
  - `1` / `CONFIRMED`: Verified exoplanet
  - `2` / `FALSE_POSITIVE`: Not a real planetary signal

## 🔧 Key Features for Classification

### Core Transit Parameters
- `koi_period`: Orbital period (days)
- `koi_duration`: Transit duration (hours)
- `koi_depth`: Transit depth (ppm)
- `koi_model_snr`: Signal-to-noise ratio

### Planetary Characteristics
- `koi_prad`: Planetary radius (Earth radii)
- `koi_insol`: Insolation flux (Earth flux)
- `koi_teq`: Equilibrium temperature (K)

### Stellar Properties
- `koi_steff`: Stellar effective temperature (K)
- `koi_slogg`: Stellar surface gravity
- `koi_srad`: Stellar radius (Solar radii)
- `koi_smass`: Stellar mass (Solar masses)

### Engineered Features
- `koi_period_log`: Log-transformed orbital period
- `koi_prad_log`: Log-transformed planetary radius
- `depth_duration_ratio`: Transit depth to duration ratio
- `planet_star_radius_ratio`: Planet-to-star radius ratio
- `habitable_zone`: Binary flag for habitable zone candidates
- `total_fp_flags`: Combined false positive flags

## 🤖 ML Recommendations

### Algorithm Suggestions

1. **Tree-Based Models** (Use MinMax scaled data)
   - Random Forest
   - XGBoost
   - LightGBM
   - Good for handling outliers and non-linear relationships

2. **Linear Models** (Use Standard scaled data)
   - Logistic Regression
   - SVM
   - Good baseline models

3. **Neural Networks** (Use Standard scaled data)
   - Multi-layer Perceptron
   - Good for complex patterns

4. **Ensemble Methods**
   - Combine multiple approaches
   - Voting classifiers
   - Stacking

### Evaluation Strategy

- **Primary Metric**: F1-score (macro or weighted)
- **Cross-Validation**: 5-fold stratified CV
- **Additional Metrics**: Precision, Recall, ROC-AUC
- **Final Evaluation**: Confusion matrix and classification report



## 🔍 Data Quality Notes

- **Missing Values**: All handled via median imputation
- **Outliers**: ~15-20% of data contains outliers (astronomical nature)
- **Scaling**: Two versions available (Standard and MinMax)
- **Class Balance**: Reasonable distribution, no special resampling needed
- **Feature Correlation**: No highly correlated features (|r| > 0.7)

## 🎨 Visualization Suggestions

- Class distribution plots
- Feature importance from tree models
- Confusion matrices
- ROC curves for each class
- Feature distributions by class
- Learning curves for model validation

## 🚀 Next Steps

1. Load the datasets and explore feature distributions
2. Implement cross-validation framework
3. Train multiple algorithms and compare performance
4. Perform hyperparameter tuning
5. Create ensemble models
6. Generate final predictions on test set
7. Analyze feature importance and model interpretability

## 📊 Expected Performance

Based on the data quality and class distribution, expect:
- **Baseline Accuracy**: ~50% (random classifier)
- **Target Performance**: 85-95% F1-score
- **Best Features**: SNR, period, depth, stellar properties

Good luck with the classification! 🌟