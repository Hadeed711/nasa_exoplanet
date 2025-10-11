#!/usr/bin/env python3
"""
🧪 Test Validation Removal & Balanced Scoring
============================================
Test the updated prediction system with various inputs
"""

from production_model_pipeline import ExoplanetModelPipeline
import pandas as pd

def test_diverse_inputs():
    """Test with diverse input values to check balance"""
    print("🧪 Testing Balanced Prediction System...")
    print("="*50)
    
    # Initialize pipeline
    pipeline = ExoplanetModelPipeline()
    
    # Test cases with diverse values
    test_cases = [
        {
            'name': 'Extreme Low Values',
            'mission': 'Kepler',
            'features': {
                'koi_period': 0.1,
                'koi_prad': 0.01,
                'koi_duration': 0.1,
                'koi_depth': 1.0,
                'koi_teq': 50.0
            }
        },
        {
            'name': 'Extreme High Values',
            'mission': 'Kepler',
            'features': {
                'koi_period': 10000.0,
                'koi_prad': 100.0,
                'koi_duration': 50.0,
                'koi_depth': 500000.0,
                'koi_teq': 5000.0
            }
        },
        {
            'name': 'Negative Values',
            'mission': 'TESS',
            'features': {
                'pl_orbper': -5.0,
                'pl_rade': -1.0,
                'pl_trandurh': -2.0,
                'st_tmag': -10.0,
                'st_teff': -1000.0
            }
        },
        {
            'name': 'Very Large Values',
            'mission': 'K2',
            'features': {
                'pl_orbper': 999999.0,
                'pl_rade': 1000.0,
                'pl_masse': 50000.0,
                'pl_dens': 500.0,
                'st_teff': 99999.0
            }
        },
        {
            'name': 'Zero Values',
            'mission': 'Kepler',
            'features': {
                'koi_period': 0.0,
                'koi_prad': 0.0,
                'koi_duration': 0.0,
                'koi_depth': 0.0,
                'koi_teq': 0.0
            }
        },
        {
            'name': 'Realistic Earth-like',
            'mission': 'TESS',
            'features': {
                'pl_orbper': 365.25,
                'pl_rade': 1.0,
                'pl_trandurh': 3.5,
                'st_tmag': 10.0,
                'st_teff': 5800.0
            }
        },
        {
            'name': 'Hot Jupiter',
            'mission': 'K2',
            'features': {
                'pl_orbper': 3.5,
                'pl_rade': 11.0,
                'pl_masse': 300.0,
                'pl_dens': 1.2,
                'st_teff': 6000.0
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print(f"Mission: {test_case['mission']}")
        print(f"Features: {test_case['features']}")
        
        try:
            result = pipeline.predict_single(test_case['mission'], test_case['features'])
            
            if result['status'] == 'success':
                prediction = result['prediction_label']
                confidence = result['confidence']
                
                print(f"✅ Result: {prediction}")
                print(f"🎯 Confidence: {confidence:.1%}")
                
                results.append({
                    'test': test_case['name'],
                    'prediction': result['prediction'],
                    'confidence': confidence
                })
            else:
                print(f"❌ Error: {result.get('error', 'Unknown')}")
                results.append({
                    'test': test_case['name'],
                    'prediction': 'Error',
                    'confidence': 0.0
                })
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'test': test_case['name'],
                'prediction': 'Exception',
                'confidence': 0.0
            })
    
    # Summary analysis
    print("\n" + "="*50)
    print("📊 RESULTS SUMMARY:")
    
    successful_predictions = [r for r in results if r['prediction'] not in ['Error', 'Exception']]
    
    if successful_predictions:
        positive_predictions = sum(1 for r in successful_predictions if r['prediction'] == 1)
        negative_predictions = sum(1 for r in successful_predictions if r['prediction'] == 0)
        avg_confidence = sum(r['confidence'] for r in successful_predictions) / len(successful_predictions)
        
        print(f"✅ Total Tests: {len(results)}")
        print(f"🌟 Exoplanet Predictions: {positive_predictions}")
        print(f"❌ Non-Planet Predictions: {negative_predictions}")
        print(f"📊 Average Confidence: {avg_confidence:.1%}")
        print(f"⚖️ Balance Ratio: {positive_predictions}/{negative_predictions}")
        
        # Check if we have better balance
        if negative_predictions > 0:
            print("✅ GOOD: System now predicts both exoplanets AND non-planets!")
        else:
            print("⚠️ WARNING: Still biased toward exoplanet predictions")
    
    print("\n🏁 Validation removal test completed!")
    
    return results

if __name__ == "__main__":
    test_diverse_inputs()