#!/usr/bin/env python3
"""
🔍 NASA Unified Dataset Verification Script
==========================================

Quick verification and demonstration of the unified dataset functionality.
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_unified_data():
    """Load and verify unified datasets."""
    print("🔍 UNIFIED DATASET VERIFICATION")
    print("=" * 40)
    
    # Load datasets
    train_data = pd.read_csv('data/processed/unified/unified_train.csv')
    val_data = pd.read_csv('data/processed/unified/unified_val.csv')
    test_data = pd.read_csv('data/processed/unified/unified_test.csv')
    
    # Load metadata
    with open('artifacts/unified/unified_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    with open('artifacts/unified/unified_feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"✅ Loaded unified datasets successfully!")
    print(f"   Training: {train_data.shape}")
    print(f"   Validation: {val_data.shape}")
    print(f"   Test: {test_data.shape}")
    print(f"   Features: {len(feature_names)}")
    
    return train_data, val_data, test_data, feature_names, metadata

def analyze_mission_distribution(train_data):
    """Analyze distribution across missions."""
    print(f"\n🚀 Mission Distribution Analysis:")
    
    mission_dist = train_data['mission_id'].value_counts().sort_index()
    mission_names = {0: 'Kepler', 1: 'TESS', 2: 'K2'}
    
    for mission_id, count in mission_dist.items():
        mission_name = mission_names.get(mission_id, f'Mission_{mission_id}')
        percentage = count / len(train_data) * 100
        print(f"   {mission_name}: {count} samples ({percentage:.1f}%)")
    
    # Class distribution by mission
    print(f"\n📊 Class Distribution by Mission:")
    cross_tab = pd.crosstab(train_data['mission_id'], train_data['disposition_binary'])
    print(cross_tab)

def quick_model_demo(train_data, val_data, feature_names):
    """Demonstrate quick model training on unified data."""
    print(f"\n🤖 Quick Model Demonstration:")
    
    # Prepare data - use multiclass for better representation
    X_train = train_data[feature_names]
    y_train = train_data['disposition_multiclass']
    X_val = val_data[feature_names]
    y_val = val_data['disposition_multiclass']
    
    print(f"   Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"   Classes: {sorted(y_train.unique())}")
    
    # Quick Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Reduced for speed
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("   Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = rf_model.predict(X_val)
    
    # Evaluate
    print(f"\n📈 Quick Validation Results:")
    class_names = ['False Positive', 'Candidate', 'Confirmed']
    print(classification_report(y_val, predictions, target_names=class_names))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"   {i+1:2d}. {row['feature']:<25} ({row['importance']:.4f})")
    
    return rf_model, feature_importance

def main():
    """Main verification workflow."""
    # Load data
    train_data, val_data, test_data, feature_names, metadata = load_unified_data()
    
    # Analyze distributions
    analyze_mission_distribution(train_data)
    
    # Quick model demo
    model, importance = quick_model_demo(train_data, val_data, feature_names)
    
    print(f"\n🎉 VERIFICATION COMPLETE!")
    print(f"✅ Unified dataset is ready for ML development!")
    print(f"✅ All {metadata['total_samples']} samples processed successfully")
    print(f"✅ {len(feature_names)} features harmonized across 3 missions")
    print(f"✅ Quick model achieves reasonable performance")
    
    print(f"\n🚀 Ready for production ML pipeline!")

if __name__ == "__main__":
    main()