#!/usr/bin/env python3
"""
🧪 Test Deterministic Predictions
===============================
Test that same inputs always give same results
"""

from production_model_pipeline import ExoplanetModelPipeline

def test_deterministic_predictions():
    """Test that predictions are consistent for same inputs"""
    print("🧪 Testing Deterministic Predictions...")
    print("="*50)
    
    # Initialize pipeline
    pipeline = ExoplanetModelPipeline()
    
    # Test case with fixed values
    test_features = {
        'koi_period': 10.5,
        'koi_prad': 2.1,
        'koi_duration': 3.2,
        'koi_depth': 1500.0,
        'koi_teq': 280.0
    }
    
    print(f"🔬 Test Features: {test_features}")
    print("\n📊 Running 5 consecutive predictions with SAME inputs:")
    
    results = []
    
    for i in range(5):
        result = pipeline.predict_single('Kepler', test_features)
        
        if result['status'] == 'success':
            prediction = result['prediction']
            confidence = result['confidence']
            label = result['prediction_label']
            
            print(f"Run {i+1}: {label} - Confidence: {confidence:.3f}")
            
            results.append({
                'run': i+1,
                'prediction': prediction,
                'confidence': confidence,
                'label': label
            })
        else:
            print(f"Run {i+1}: ERROR - {result.get('error', 'Unknown')}")
    
    # Check consistency
    print("\n" + "="*50)
    print("🔍 CONSISTENCY CHECK:")
    
    if len(results) >= 2:
        # Check if all predictions are the same
        first_prediction = results[0]['prediction']
        first_confidence = results[0]['confidence']
        
        all_same_prediction = all(r['prediction'] == first_prediction for r in results)
        all_same_confidence = all(abs(r['confidence'] - first_confidence) < 0.001 for r in results)
        
        if all_same_prediction and all_same_confidence:
            print("✅ SUCCESS: All predictions are IDENTICAL!")
            print(f"   Prediction: {results[0]['label']}")
            print(f"   Confidence: {first_confidence:.3f}")
            return True
        else:
            print("❌ FAILURE: Predictions are inconsistent!")
            print("   Predictions:", [r['prediction'] for r in results])
            print("   Confidences:", [f"{r['confidence']:.3f}" for r in results])
            return False
    else:
        print("❌ FAILURE: Not enough successful predictions to test")
        return False

def test_different_inputs():
    """Test that different inputs give different results"""
    print("\n" + "="*50)
    print("🧪 Testing Different Inputs Give Different Results...")
    
    pipeline = ExoplanetModelPipeline()
    
    test_cases = [
        {
            'name': 'Earth-like',
            'features': {'koi_period': 365.0, 'koi_prad': 1.0, 'koi_teq': 280.0}
        },
        {
            'name': 'Hot Jupiter',
            'features': {'koi_period': 3.5, 'koi_prad': 11.0, 'koi_teq': 1500.0}
        },
        {
            'name': 'Invalid',
            'features': {'koi_period': -1.0, 'koi_prad': -5.0, 'koi_teq': -100.0}
        }
    ]
    
    results = []
    
    for case in test_cases:
        result = pipeline.predict_single('Kepler', case['features'])
        if result['status'] == 'success':
            print(f"{case['name']:12}: {result['prediction_label']} - {result['confidence']:.3f}")
            results.append(result['confidence'])
        else:
            print(f"{case['name']:12}: ERROR")
    
    # Check that different inputs give different outputs
    if len(set(results)) > 1:
        print("✅ SUCCESS: Different inputs produce different results!")
        return True
    else:
        print("⚠️ WARNING: All inputs produced same result")
        return False

if __name__ == "__main__":
    deterministic_ok = test_deterministic_predictions()
    different_ok = test_different_inputs()
    
    print("\n" + "="*50)
    print("🏁 FINAL RESULT:")
    
    if deterministic_ok and different_ok:
        print("✅ ALL TESTS PASSED! Predictions are now deterministic!")
    else:
        print("❌ SOME TESTS FAILED! Check the issues above.")
    
    print(f"Deterministic: {'✅' if deterministic_ok else '❌'}")
    print(f"Different Inputs: {'✅' if different_ok else '❌'}")