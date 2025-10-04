#!/usr/bin/env python3
"""
🚀 Production Model Loader for NASA Exoplanet Detection
======================================================
Ready-to-use model loading and prediction pipeline for Streamlit UI
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ExoplanetModelPipeline:
    """
    Production-ready model pipeline for NASA Exoplanet Detection
    Handles all 3 missions: Kepler, TESS, K2
    """
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models"""
        model_paths = {
            'Kepler': 'kepler model training/best_exoplanet_model_LightGBM.pkl',
            'TESS': 'TESS model training/best_exoplanet_model_XGBoost.pkl',
            'K2': 'k2 model training/best_exoplanet_model_CatBoost.pkl'
        }
        
        for mission, path in model_paths.items():
            try:
                if os.path.exists(path):
                    model = joblib.load(path)
                    self.models[mission] = model
                    self.model_info[mission] = {
                        'type': type(model).__name__,
                        'path': path,
                        'status': 'loaded'
                    }
                    print(f"✅ {mission} model loaded: {type(model).__name__}")
                else:
                    print(f"❌ {mission} model not found: {path}")
                    self.model_info[mission] = {'status': 'not_found', 'path': path}
            except Exception as e:
                print(f"❌ Error loading {mission} model: {e}")
                self.model_info[mission] = {'status': 'error', 'error': str(e)}
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get pre-computed model performance metrics"""
        performance = {
            'Kepler': {
                'accuracy': 0.9598,
                'f1_score': 0.9599,
                'precision': 0.9601,
                'recall': 0.9598,
                'roc_auc': 0.9920,
                'test_samples': 1841
            },
            'TESS': {
                'accuracy': 0.9176,
                'f1_score': 0.9174,
                'precision': 0.9173,
                'recall': 0.9176,
                'roc_auc': 0.9018,
                'test_samples': 1395
            },
            'K2': {
                'accuracy': 0.9111,
                'f1_score': 0.9105,
                'precision': 0.9124,
                'recall': 0.9111,
                'roc_auc': 0.9749,
                'test_samples': 799
            }
        }
        return performance
    
    def get_feature_info(self, mission: str) -> Dict[str, Any]:
        """Get feature information for a specific mission"""
        feature_info = {
            'Kepler': {
                'total_features': 127,
                'key_features': [
                    'koi_period', 'koi_prad', 'koi_dor', 'koi_duration', 
                    'koi_depth', 'koi_teq', 'koi_insol'
                ],
                'target_column': 'target',
                'description': 'Kepler Objects of Interest (KOI) features'
            },
            'TESS': {
                'total_features': 84,
                'key_features': [
                    'pl_orbper', 'pl_rade', 'pl_trandurh', 'pl_trandep',
                    'st_tmag', 'st_teff', 'st_logg', 'st_rad'
                ],
                'target_column': 'target',
                'description': 'TESS Objects of Interest (TOI) features'
            },
            'K2': {
                'total_features': 60,
                'key_features': [
                    'pl_orbper', 'pl_rade', 'pl_masse', 'pl_dens',
                    'pl_insol', 'pl_eqt', 'st_teff', 'st_rad'
                ],
                'target_column': 'disposition_binary',
                'description': 'K2 mission exoplanet candidates'
            }
        }
        return feature_info.get(mission, {})
    
    def predict_single(self, mission: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            mission: 'Kepler', 'TESS', or 'K2'
            features: Dictionary of feature values
            
        Returns:
            Dictionary with prediction results
        """
        if mission not in self.models:
            return {
                'error': f'Model for {mission} not available',
                'status': 'failed'
            }
        
        try:
            model = self.models[mission]
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0]
            
            # Get class labels
            classes = model.classes_ if hasattr(model, 'classes_') else [0, 1]
            
            # Create result
            result = {
                'mission': mission,
                'prediction': int(prediction),
                'prediction_label': 'Confirmed Planet' if prediction == 1 else 'Not Planet',
                'confidence': float(max(probability)),
                'probabilities': {
                    'Not Planet': float(probability[0]),
                    'Confirmed Planet': float(probability[1]) if len(probability) > 1 else 0.0
                },
                'model_type': type(model).__name__,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'status': 'failed'
            }
    
    def predict_batch(self, mission: str, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple samples
        
        Args:
            mission: 'Kepler', 'TESS', or 'K2'
            features_df: DataFrame with features
            
        Returns:
            DataFrame with predictions added
        """
        if mission not in self.models:
            raise ValueError(f'Model for {mission} not available')
        
        try:
            model = self.models[mission]
            
            # Make predictions
            predictions = model.predict(features_df)
            probabilities = model.predict_proba(features_df)
            
            # Add predictions to DataFrame
            result_df = features_df.copy()
            result_df['prediction'] = predictions
            result_df['prediction_label'] = ['Confirmed Planet' if p == 1 else 'Not Planet' for p in predictions]
            result_df['confidence'] = np.max(probabilities, axis=1)
            result_df['prob_not_planet'] = probabilities[:, 0]
            result_df['prob_confirmed_planet'] = probabilities[:, 1] if probabilities.shape[1] > 1 else 0.0
            
            return result_df
            
        except Exception as e:
            raise ValueError(f'Batch prediction failed: {str(e)}')
    
    def get_feature_importance(self, mission: str, top_n: int = 10) -> Dict[str, float]:
        """Get feature importance for a model"""
        if mission not in self.models:
            return {}
        
        try:
            model = self.models[mission]
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names (you might need to adjust this based on your training)
                # For now, using generic names
                n_features = len(model.feature_importances_)
                feature_names = [f'feature_{i}' for i in range(n_features)]
                
                # Create importance dictionary
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                
                # Sort and return top N
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_importance[:top_n])
            else:
                return {}
                
        except Exception as e:
            print(f"Error getting feature importance for {mission}: {e}")
            return {}
    
    def get_available_missions(self) -> list:
        """Get list of available missions"""
        return list(self.models.keys())
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models"""
        return self.model_info
    
    def validate_features(self, mission: str, features: Dict[str, float]) -> Tuple[bool, str]:
        """Validate input features for a mission"""
        feature_info = self.get_feature_info(mission)
        
        if not feature_info:
            return False, f"Unknown mission: {mission}"
        
        # Basic validation (you can expand this)
        required_features = feature_info.get('key_features', [])
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            return False, f"Missing required features: {missing_features}"
        
        return True, "Features valid"

# Example usage for Streamlit
def demo_usage():
    """Demonstrate how to use the pipeline"""
    print("🚀 NASA Exoplanet Detection Pipeline Demo")
    print("="*50)
    
    # Initialize pipeline
    pipeline = ExoplanetModelPipeline()
    
    # Show available missions
    missions = pipeline.get_available_missions()
    print(f"Available missions: {missions}")
    
    # Show model performance
    performance = pipeline.get_model_performance()
    for mission, perf in performance.items():
        if mission in missions:
            print(f"\n{mission} Model Performance:")
            print(f"  Accuracy: {perf['accuracy']:.4f}")
            print(f"  F1 Score: {perf['f1_score']:.4f}")
    
    # Example prediction (you'll need real feature values)
    if 'K2' in missions:
        sample_features = {
            'pl_orbper': 10.0,
            'pl_rade': 2.0,
            'pl_masse': 5.0,
            'pl_dens': 3.0,
            'pl_insol': 100.0,
            'pl_eqt': 300.0,
            'st_teff': 5800.0,
            'st_rad': 1.0
        }
        
        # Note: This will fail without proper feature preprocessing
        # but shows the API structure
        try:
            result = pipeline.predict_single('K2', sample_features)
            print(f"\nExample prediction: {result}")
        except Exception as e:
            print(f"Example prediction failed (expected): {e}")

if __name__ == "__main__":
    demo_usage()