#!/usr/bin/env python3
"""
🌟 NASA Exoplanet Dataset Harmonization Script
=============================================

Combines Kepler, TESS TOI, and K2 datasets into a unified ML-ready dataset.
This script harmonizes features across all three space missions for comprehensive 
exoplanet classification.

Author: Data Science Team
Date: September 2025
Project: NASA Space Apps Challenge - Exoplanet Detection
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load all three processed datasets with correct target columns."""
    print("🚀 Loading NASA Space Mission Datasets...")
    
    # Load Kepler data (has 'target' and 'target_name')
    kepler_train = pd.read_csv('data/processed/kepler_train.csv')
    kepler_val = pd.read_csv('data/processed/kepler_val.csv') 
    kepler_test = pd.read_csv('data/processed/kepler_test.csv')
    kepler_full = pd.concat([kepler_train, kepler_val, kepler_test], ignore_index=True)
    
    # Standardize target columns for Kepler
    kepler_full['disposition_binary'] = kepler_full['target']
    kepler_full['disposition_multiclass'] = kepler_full['target']  # Kepler is already binary
    kepler_full['original_disposition'] = kepler_full['target_name']
    kepler_full = kepler_full.drop(['target', 'target_name'], axis=1)
    
    kepler_full['mission'] = 'Kepler'
    kepler_full['mission_id'] = 0
    
    # Load TESS data (has 'disposition_binary' and 'disposition_multiclass')
    tess_train = pd.read_csv('data/processed/tess/tess_train.csv')
    tess_val = pd.read_csv('data/processed/tess/tess_val.csv')
    tess_test = pd.read_csv('data/processed/tess/tess_test.csv')
    tess_full = pd.concat([tess_train, tess_val, tess_test], ignore_index=True)
    
    # TESS already has correct target columns
    if 'target_name' in tess_full.columns:
        tess_full['original_disposition'] = tess_full['target_name']
        tess_full = tess_full.drop(['target_name'], axis=1)
    else:
        tess_full['original_disposition'] = 'TESS_' + tess_full['disposition_multiclass'].astype(str)
    
    tess_full['mission'] = 'TESS'
    tess_full['mission_id'] = 1
    
    # Load K2 data (has 'disposition_binary' and 'disposition_multiclass')
    k2_train = pd.read_csv('data/processed/k2/k2_train.csv')
    k2_val = pd.read_csv('data/processed/k2/k2_val.csv')
    k2_test = pd.read_csv('data/processed/k2/k2_test.csv')
    k2_full = pd.concat([k2_train, k2_val, k2_test], ignore_index=True)
    
    # K2 already has correct target columns
    k2_full['original_disposition'] = 'K2_' + k2_full['disposition_multiclass'].astype(str)
    k2_full['mission'] = 'K2'
    k2_full['mission_id'] = 2
    
    print(f"✅ Loaded datasets:")
    print(f"   Kepler: {len(kepler_full)} samples")
    print(f"   TESS: {len(tess_full)} samples") 
    print(f"   K2: {len(k2_full)} samples")
    
    return kepler_full, tess_full, k2_full

def harmonize_features(kepler_df, tess_df, k2_df):
    """Harmonize features across all datasets."""
    print("\n🔧 Harmonizing Features Across Missions...")
    
    # Find common features (excluding target variables and mission info)
    exclude_cols = ['disposition_binary', 'disposition_multiclass', 'original_disposition', 'mission', 'mission_id']
    
    kepler_features = set(kepler_df.columns) - set(exclude_cols)
    tess_features = set(tess_df.columns) - set(exclude_cols)
    k2_features = set(k2_df.columns) - set(exclude_cols)
    
    # Find intersection and union
    common_features = kepler_features & tess_features & k2_features
    all_features = kepler_features | tess_features | k2_features
    
    print(f"   Common features across all missions: {len(common_features)}")
    print(f"   Total unique features: {len(all_features)}")
    
    # Mission-specific features
    kepler_specific = kepler_features - (tess_features | k2_features)
    tess_specific = tess_features - (kepler_features | k2_features)
    k2_specific = k2_features - (kepler_features | tess_features)
    
    print(f"   Kepler-specific features: {len(kepler_specific)}")
    print(f"   TESS-specific features: {len(tess_specific)}")
    print(f"   K2-specific features: {len(k2_specific)}")
    
    # Create unified feature set
    unified_feature_list = list(all_features)
    target_cols = ['disposition_binary', 'disposition_multiclass', 'original_disposition', 'mission', 'mission_id']
    unified_columns = unified_feature_list + target_cols
    
    print(f"   Final unified features: {len(unified_feature_list)}")
    
    # Harmonize each dataset by adding missing columns with zeros
    def harmonize_single_dataset(df, dataset_name):
        df_harmonized = df.copy()
        
        # Add missing feature columns with zeros
        for col in unified_feature_list:
            if col not in df_harmonized.columns:
                df_harmonized[col] = 0.0
        
        # Ensure we have all required columns
        for col in target_cols:
            if col not in df_harmonized.columns:
                if col == 'original_disposition':
                    df_harmonized[col] = f"{dataset_name}_unknown"
                elif col in ['disposition_binary', 'disposition_multiclass']:
                    df_harmonized[col] = 0
                elif col == 'mission':
                    df_harmonized[col] = dataset_name
                elif col == 'mission_id':
                    df_harmonized[col] = 0
        
        # Reorder columns
        df_harmonized = df_harmonized[unified_columns]
        return df_harmonized
    
    kepler_harmonized = harmonize_single_dataset(kepler_df, 'Kepler')
    tess_harmonized = harmonize_single_dataset(tess_df, 'TESS')
    k2_harmonized = harmonize_single_dataset(k2_df, 'K2')
    
    return kepler_harmonized, tess_harmonized, k2_harmonized, unified_feature_list

def create_unified_dataset(kepler_df, tess_df, k2_df):
    """Combine all datasets into unified format."""
    print("\n🌟 Creating Unified NASA Exoplanet Dataset...")
    
    # Combine all datasets
    unified_df = pd.concat([kepler_df, tess_df, k2_df], ignore_index=True)
    
    # Add dataset source tracking
    unified_df['dataset_source'] = unified_df['mission'].copy()
    
    print(f"✅ Unified dataset created:")
    print(f"   Total samples: {len(unified_df)}")
    print(f"   Total features: {unified_df.shape[1] - 5}")  # Exclude targets and mission info
    
    # Class distribution analysis
    print(f"\n📊 Unified Class Distribution:")
    print("Binary classification:")
    binary_dist = unified_df['disposition_binary'].value_counts()
    for class_val, count in binary_dist.items():
        print(f"   Class {class_val}: {count} ({count/len(unified_df)*100:.1f}%)")
    
    print("Multi-class classification:")
    multi_dist = unified_df['disposition_multiclass'].value_counts()
    for class_val, count in multi_dist.items():
        print(f"   Class {class_val}: {count} ({count/len(unified_df)*100:.1f}%)")
    
    print("By mission:")
    mission_dist = unified_df['mission'].value_counts()
    for mission, count in mission_dist.items():
        print(f"   {mission}: {count} ({count/len(unified_df)*100:.1f}%)")
    
    return unified_df

def create_unified_splits(unified_df, feature_columns):
    """Create new train/val/test splits from unified dataset."""
    print("\n🎯 Creating Unified Train/Validation/Test Splits...")
    
    from sklearn.model_selection import train_test_split
    
    # Features and targets
    X = unified_df[feature_columns + ['mission_id']].copy()
    y_binary = unified_df['disposition_binary'].copy()
    y_multiclass = unified_df['disposition_multiclass'].copy()
    
    # Stratified split: 60% train, 20% val, 20% test
    X_train, X_temp, y_bin_train, y_bin_temp, y_multi_train, y_multi_temp = train_test_split(
        X, y_binary, y_multiclass, test_size=0.4, random_state=42, 
        stratify=y_multiclass
    )
    
    X_val, X_test, y_bin_val, y_bin_test, y_multi_val, y_multi_test = train_test_split(
        X_temp, y_bin_temp, y_multi_temp, test_size=0.5, random_state=42,
        stratify=y_multi_temp
    )
    
    print(f"✅ Split sizes:")
    print(f"   Training: {len(X_train)} samples ({len(X_train)/len(unified_df)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(unified_df)*100:.1f}%)")
    print(f"   Test: {len(X_test)} samples ({len(X_test)/len(unified_df)*100:.1f}%)")
    
    return (X_train, X_val, X_test, 
            y_bin_train, y_bin_val, y_bin_test,
            y_multi_train, y_multi_val, y_multi_test)

def save_unified_datasets(X_train, X_val, X_test, 
                         y_bin_train, y_bin_val, y_bin_test,
                         y_multi_train, y_multi_val, y_multi_test):
    """Save unified datasets and metadata."""
    print("\n💾 Saving Unified Datasets...")
    
    # Create unified output directory
    os.makedirs('data/processed/unified', exist_ok=True)
    os.makedirs('artifacts/unified', exist_ok=True)
    
    # Combine features with targets
    train_data = X_train.copy()
    train_data['disposition_binary'] = y_bin_train
    train_data['disposition_multiclass'] = y_multi_train
    
    val_data = X_val.copy()
    val_data['disposition_binary'] = y_bin_val
    val_data['disposition_multiclass'] = y_multi_val
    
    test_data = X_test.copy()
    test_data['disposition_binary'] = y_bin_test
    test_data['disposition_multiclass'] = y_multi_test
    
    # Save datasets
    train_data.to_csv('data/processed/unified/unified_train.csv', index=False)
    val_data.to_csv('data/processed/unified/unified_val.csv', index=False)
    test_data.to_csv('data/processed/unified/unified_test.csv', index=False)
    
    # Save feature names
    feature_names = [col for col in X_train.columns]
    with open('artifacts/unified/unified_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save metadata
    unified_metadata = {
        'dataset_name': 'NASA_Unified_Exoplanet_Dataset',
        'missions': ['Kepler', 'TESS', 'K2'],
        'total_samples': len(train_data) + len(val_data) + len(test_data),
        'n_features': len(feature_names),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'class_distribution_binary': {
            'train': y_bin_train.value_counts().to_dict(),
            'val': y_bin_val.value_counts().to_dict(),
            'test': y_bin_test.value_counts().to_dict()
        },
        'class_distribution_multiclass': {
            'train': y_multi_train.value_counts().to_dict(),
            'val': y_multi_val.value_counts().to_dict(),
            'test': y_multi_test.value_counts().to_dict()
        },
        'mission_distribution': {
            'train': X_train['mission_id'].value_counts().to_dict(),
            'val': X_val['mission_id'].value_counts().to_dict(),
            'test': X_test['mission_id'].value_counts().to_dict()
        }
    }
    
    with open('artifacts/unified/unified_metadata.pkl', 'wb') as f:
        pickle.dump(unified_metadata, f)
    
    print("✅ Unified datasets saved:")
    print(f"   📄 data/processed/unified/unified_train.csv ({train_data.shape})")
    print(f"   📄 data/processed/unified/unified_val.csv ({val_data.shape})")
    print(f"   📄 data/processed/unified/unified_test.csv ({test_data.shape})")
    print(f"   🔧 artifacts/unified/unified_feature_names.pkl")
    print(f"   📊 artifacts/unified/unified_metadata.pkl")
    
    return unified_metadata

def main():
    """Main harmonization pipeline."""
    print("🌟 NASA EXOPLANET DATASET HARMONIZATION")
    print("=" * 50)
    
    # Load individual datasets
    kepler_df, tess_df, k2_df = load_datasets()
    
    # Harmonize features
    kepler_harm, tess_harm, k2_harm, feature_list = harmonize_features(kepler_df, tess_df, k2_df)
    
    # Create unified dataset
    unified_df = create_unified_dataset(kepler_harm, tess_harm, k2_harm)
    
    # Create new splits
    splits = create_unified_splits(unified_df, feature_list)
    
    # Save everything
    metadata = save_unified_datasets(*splits)
    
    print(f"\n🎉 DATASET HARMONIZATION COMPLETE!")
    print(f"🚀 Ready for unified ML model training!")
    print(f"📊 Total samples: {metadata['total_samples']}")
    print(f"🎯 Features: {metadata['n_features']}")
    print(f"🌟 Missions: {', '.join(metadata['missions'])}")

if __name__ == "__main__":
    main()