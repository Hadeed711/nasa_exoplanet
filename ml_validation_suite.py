#!/usr/bin/env python3
"""
🧪 ML IMPLEMENTATION VALIDATION SUITE
====================================
Comprehensive testing of all ML models for NASA Exoplanet Detection
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

class MLValidationSuite:
    def __init__(self):
        self.results = {}
        self.model_paths = {
            'Kepler': 'kepler model training/',
            'TESS': 'TESS model training/',
            'K2': 'k2 model training/'
        }
        self.data_paths = {
            'Kepler': 'data/processed/',
            'TESS': 'data/processed/tess/',
            'K2': 'data/processed/k2/'
        }
        
    def test_dataset_integrity(self):
        """Test if all datasets are properly processed and available"""
        print("🔍 TESTING DATASET INTEGRITY")
        print("=" * 60)
        
        integrity_results = {}
        
        for mission in ['Kepler', 'TESS', 'K2']:
            print(f"\n📊 Testing {mission} Dataset:")
            
            try:
                # Determine file paths
                if mission == 'Kepler':
                    train_file = 'data/processed/kepler_train.csv'
                    test_file = 'data/processed/kepler_test.csv'
                elif mission == 'TESS':
                    train_file = 'data/processed/tess/tess_train.csv'
                    test_file = 'data/processed/tess/tess_test.csv'
                else:  # K2
                    train_file = 'data/processed/k2/k2_train.csv'
                    test_file = 'data/processed/k2/k2_test.csv'
                
                # Check if files exist
                if not os.path.exists(train_file) or not os.path.exists(test_file):
                    print(f"❌ Missing data files for {mission}")
                    integrity_results[mission] = {'status': 'FAILED', 'reason': 'Missing files'}
                    continue
                
                # Load datasets
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                
                # Check required columns (different for each mission)
                if mission == 'K2':
                    required_cols = ['disposition_binary']
                else:  # Kepler and TESS
                    required_cols = ['target', 'target_name']
                
                missing_cols = [col for col in required_cols if col not in train_df.columns]
                
                if missing_cols:
                    print(f"❌ Missing required columns: {missing_cols}")
                    integrity_results[mission] = {'status': 'FAILED', 'reason': f'Missing columns: {missing_cols}'}
                    continue
                
                # Calculate metrics
                total_samples = len(train_df) + len(test_df)
                
                # Determine target column and feature count
                if mission == 'K2':
                    target_col = 'disposition_binary'
                    exclude_cols = ['disposition_binary', 'disposition_multiclass']
                else:  # Kepler and TESS
                    target_col = 'target'
                    exclude_cols = ['target', 'target_name']
                
                n_features = len([col for col in train_df.columns if col not in exclude_cols])
                
                # Check class distribution
                class_dist = train_df[target_col].value_counts()
                class_balance = class_dist.min() / class_dist.max() if len(class_dist) > 1 else 1.0
                
                # Check for null values
                null_percentage = (train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1])) * 100
                
                print(f"✅ {mission} Dataset Valid:")
                print(f"   📊 Total samples: {total_samples:,}")
                print(f"   🎯 Features: {n_features}")
                print(f"   📋 Train/Test: {len(train_df)}/{len(test_df)}")
                print(f"   ⚖️ Class balance: {class_balance:.3f}")
                print(f"   🔍 Null percentage: {null_percentage:.2f}%")
                
                integrity_results[mission] = {
                    'status': 'PASSED',
                    'total_samples': total_samples,
                    'n_features': n_features,
                    'class_balance': class_balance,
                    'null_percentage': null_percentage,
                    'train_size': len(train_df),
                    'test_size': len(test_df)
                }
                
            except Exception as e:
                print(f"❌ Error testing {mission}: {str(e)}")
                integrity_results[mission] = {'status': 'FAILED', 'reason': str(e)}
        
        return integrity_results
    
    def test_model_files(self):
        """Test if ML models are properly trained and saved"""
        print("\n🤖 TESTING ML MODEL FILES")
        print("=" * 60)
        
        model_results = {}
        
        for mission in ['Kepler', 'TESS', 'K2']:
            print(f"\n🔍 Testing {mission} Model:")
            
            try:
                model_dir = self.model_paths[mission]
                
                # Look for model files
                model_files = []
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith('.pkl') and 'model' in file.lower():
                            model_files.append(os.path.join(model_dir, file))
                
                if not model_files:
                    print(f"❌ No model files found for {mission}")
                    model_results[mission] = {'status': 'FAILED', 'reason': 'No model files found'}
                    continue
                
                # Test the first model file found
                model_file = model_files[0]
                print(f"   📁 Model file: {os.path.basename(model_file)}")
                
                # Load model
                model = joblib.load(model_file)
                model_type = type(model).__name__
                print(f"   🏷️ Model type: {model_type}")
                
                # Check model attributes
                has_predict = hasattr(model, 'predict')
                has_predict_proba = hasattr(model, 'predict_proba')
                has_feature_importances = hasattr(model, 'feature_importances_')
                
                print(f"   ✅ Has predict method: {has_predict}")
                print(f"   ✅ Has predict_proba method: {has_predict_proba}")
                print(f"   ✅ Has feature importances: {has_feature_importances}")
                
                if has_predict and has_predict_proba:
                    model_results[mission] = {
                        'status': 'PASSED',
                        'model_type': model_type,
                        'model_file': model_file,
                        'has_feature_importances': has_feature_importances
                    }
                else:
                    model_results[mission] = {
                        'status': 'PARTIAL',
                        'model_type': model_type,
                        'reason': 'Missing prediction methods'
                    }
                    
            except Exception as e:
                print(f"❌ Error testing {mission} model: {str(e)}")
                model_results[mission] = {'status': 'FAILED', 'reason': str(e)}
        
        return model_results
    
    def validate_model_performance(self):
        """Validate actual model performance on test data"""
        print("\n🎯 VALIDATING MODEL PERFORMANCE")
        print("=" * 60)
        
        performance_results = {}
        
        for mission in ['Kepler', 'TESS', 'K2']:
            print(f"\n🚀 Testing {mission} Model Performance:")
            
            try:
                # Load test data
                if mission == 'Kepler':
                    test_file = 'data/processed/kepler_test.csv'
                elif mission == 'TESS':
                    test_file = 'data/processed/tess/tess_test.csv'
                else:  # K2
                    test_file = 'data/processed/k2/k2_test.csv'
                
                if not os.path.exists(test_file):
                    print(f"❌ Test data not found for {mission}")
                    performance_results[mission] = {'status': 'FAILED', 'reason': 'Test data missing'}
                    continue
                
                test_df = pd.read_csv(test_file)
                
                # Load model
                model_dir = self.model_paths[mission]
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f.lower()]
                
                if not model_files:
                    print(f"❌ Model not found for {mission}")
                    performance_results[mission] = {'status': 'FAILED', 'reason': 'Model missing'}
                    continue
                
                model_file = os.path.join(model_dir, model_files[0])
                model = joblib.load(model_file)
                
                # Prepare test data (handle different column names)
                if mission == 'K2':
                    X_test = test_df.drop(['disposition_binary', 'disposition_multiclass'], axis=1, errors='ignore')
                    y_test = test_df['disposition_binary']
                else:  # Kepler and TESS
                    X_test = test_df.drop(['target', 'target_name'], axis=1, errors='ignore')
                    y_test = test_df['target']
                
                # Remove any additional columns that might cause issues
                non_feature_cols = ['target_name', 'default_flag', 'k2_campaigns_num']
                for col in non_feature_cols:
                    if col in X_test.columns:
                        X_test = X_test.drop(col, axis=1)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                print(f"   ✅ Model Performance:")
                print(f"      🎯 Accuracy: {accuracy:.4f}")
                print(f"      🎯 F1 Score: {f1:.4f}")
                print(f"      🎯 Precision: {precision:.4f}")
                print(f"      🎯 Recall: {recall:.4f}")
                print(f"      🎯 ROC-AUC: {roc_auc:.4f}")
                
                # Performance thresholds for hackathon readiness
                is_ready = accuracy > 0.70 and f1 > 0.30 and roc_auc > 0.60
                readiness_status = "HACKATHON READY ✅" if is_ready else "NEEDS IMPROVEMENT ⚠️"
                print(f"      🏆 Status: {readiness_status}")
                
                performance_results[mission] = {
                    'status': 'PASSED',
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'roc_auc': roc_auc,
                    'hackathon_ready': is_ready,
                    'test_samples': len(y_test),
                    'model_type': type(model).__name__
                }
                
            except Exception as e:
                print(f"❌ Error validating {mission} performance: {str(e)}")
                performance_results[mission] = {'status': 'FAILED', 'reason': str(e)}
        
        return performance_results
    
    def test_prediction_files(self):
        """Test if prediction files are properly generated"""
        print("\n📄 TESTING PREDICTION FILES")
        print("=" * 60)
        
        prediction_results = {}
        
        for mission in ['Kepler', 'TESS', 'K2']:
            print(f"\n📊 Testing {mission} Predictions:")
            
            try:
                model_dir = self.model_paths[mission]
                
                # Look for prediction files
                prediction_files = []
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if 'prediction' in file.lower() and file.endswith('.csv'):
                            prediction_files.append(os.path.join(model_dir, file))
                
                if not prediction_files:
                    print(f"❌ No prediction files found for {mission}")
                    prediction_results[mission] = {'status': 'FAILED', 'reason': 'No prediction files'}
                    continue
                
                # Load and validate prediction file
                pred_file = prediction_files[0]
                pred_df = pd.read_csv(pred_file)
                
                print(f"   📁 Prediction file: {os.path.basename(pred_file)}")
                print(f"   📊 Shape: {pred_df.shape}")
                print(f"   🔍 Columns: {list(pred_df.columns)}")
                
                # Check required columns
                required_cols = ['true_label', 'predicted_label']
                has_required = all(col in pred_df.columns for col in required_cols)
                
                if has_required:
                    print(f"   ✅ Has required columns: {required_cols}")
                    prediction_results[mission] = {
                        'status': 'PASSED',
                        'prediction_file': pred_file,
                        'shape': pred_df.shape,
                        'columns': list(pred_df.columns)
                    }
                else:
                    print(f"   ❌ Missing required columns")
                    prediction_results[mission] = {
                        'status': 'PARTIAL',
                        'reason': 'Missing required columns'
                    }
                    
            except Exception as e:
                print(f"❌ Error testing {mission} predictions: {str(e)}")
                prediction_results[mission] = {'status': 'FAILED', 'reason': str(e)}
        
        return prediction_results
    
    def generate_overall_report(self, integrity_results, model_results, performance_results, prediction_results):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("🏆 OVERALL ML VALIDATION REPORT")
        print("="*80)
        
        # Count successes
        total_missions = 3
        integrity_passed = sum(1 for r in integrity_results.values() if r['status'] == 'PASSED')
        models_passed = sum(1 for r in model_results.values() if r['status'] == 'PASSED')
        performance_passed = sum(1 for r in performance_results.values() if r['status'] == 'PASSED')
        predictions_passed = sum(1 for r in prediction_results.values() if r['status'] == 'PASSED')
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   🗃️  Dataset Integrity: {integrity_passed}/{total_missions} ({'✅' if integrity_passed == total_missions else '⚠️'})")
        print(f"   🤖 Model Files: {models_passed}/{total_missions} ({'✅' if models_passed == total_missions else '⚠️'})")
        print(f"   🎯 Model Performance: {performance_passed}/{total_missions} ({'✅' if performance_passed == total_missions else '⚠️'})")
        print(f"   📄 Prediction Files: {predictions_passed}/{total_missions} ({'✅' if predictions_passed == total_missions else '⚠️'})")
        
        # Overall readiness score
        total_checks = integrity_passed + models_passed + performance_passed + predictions_passed
        max_checks = total_missions * 4
        readiness_score = (total_checks / max_checks) * 100
        
        print(f"\n🏆 HACKATHON READINESS SCORE: {readiness_score:.1f}%")
        
        if readiness_score >= 90:
            print("🚀 STATUS: EXCELLENT - Ready for submission!")
        elif readiness_score >= 75:
            print("✅ STATUS: GOOD - Minor issues to address")
        elif readiness_score >= 60:
            print("⚠️  STATUS: NEEDS WORK - Several issues to fix")
        else:
            print("❌ STATUS: NOT READY - Major issues need resolution")
        
        # Detailed performance summary
        print(f"\n📈 DETAILED MODEL PERFORMANCE:")
        for mission in ['Kepler', 'TESS', 'K2']:
            if mission in performance_results and performance_results[mission]['status'] == 'PASSED':
                perf = performance_results[mission]
                status = "✅ READY" if perf['hackathon_ready'] else "⚠️ NEEDS WORK"
                print(f"   {mission:8}: Acc={perf['accuracy']:.3f} | F1={perf['f1_score']:.3f} | AUC={perf['roc_auc']:.3f} | {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS FOR STREAMLIT UI:")
        print("   1. ✅ Use separate model approach (not unified)")
        print("   2. ✅ Load models from respective directories")
        print("   3. ✅ Implement prediction pipeline for each mission")
        print("   4. ✅ Add model performance display")
        print("   5. ✅ Include feature importance visualization")
        
        return {
            'readiness_score': readiness_score,
            'integrity_passed': integrity_passed,
            'models_passed': models_passed,
            'performance_passed': performance_passed,
            'predictions_passed': predictions_passed,
            'overall_status': 'READY' if readiness_score >= 75 else 'NEEDS_WORK'
        }
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🚀 STARTING COMPREHENSIVE ML VALIDATION")
        print("="*80)
        
        # Run all tests
        integrity_results = self.test_dataset_integrity()
        model_results = self.test_model_files()
        performance_results = self.validate_model_performance()
        prediction_results = self.test_prediction_files()
        
        # Generate overall report
        overall_results = self.generate_overall_report(
            integrity_results, model_results, 
            performance_results, prediction_results
        )
        
        # Store all results
        self.results = {
            'integrity': integrity_results,
            'models': model_results,
            'performance': performance_results,
            'predictions': prediction_results,
            'overall': overall_results
        }
        
        return self.results

if __name__ == "__main__":
    # Run validation suite
    validator = MLValidationSuite()
    results = validator.run_full_validation()
    
    print(f"\n🎉 VALIDATION COMPLETE!")
    print(f"Results saved in validator.results for detailed analysis")