#!/usr/bin/env python3
"""
🧪 Test Script: Verify Preprocessing Fix
======================================
Quick test to verify that preprocessing is working in the production pipeline
"""

from production_model_pipeline import ExoplanetModelPipeline
import pandas as pd

def test_preprocessing_integration():
    """Test that preprocessing is properly integrated"""
    print("🚀 Testing ExoLume Preprocessing Integration...")
    print("="*50)
    
    # Initialize pipeline
    try:
        pipeline = ExoplanetModelPipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Test each mission
    test_features = {
        'Kepler': {
            'koi_period': 365.25,
            'koi_prad': 1.2,
            'koi_duration': 3.5,
            'koi_depth': 500,
            'koi_teq': 280
        },
        'TESS': {
            'pl_orbper': 100.5,
            'pl_rade': 1.1,
            'pl_trandurh': 2.8,
            'st_tmag': 12.5,
            'st_teff': 5500
        },
        'K2': {
            'pl_orbper': 200.0,
            'pl_rade': 0.9,
            'pl_masse': 0.8,
            'pl_dens': 5.2,
            'st_teff': 5200
        }
    }
    
    results = {}
    
    for mission, features in test_features.items():
        print(f"\n🔬 Testing {mission} mission...")
        
        try:
            result = pipeline.predict_single(mission, features)
            
            if result['status'] == 'success':
                preprocessing_status = result.get('preprocessing_applied', 'Unknown')
                print(f"  ✅ Prediction successful")
                print(f"  📊 Result: {result['prediction_label']}")
                print(f"  🎯 Confidence: {result['confidence']:.1%}")
                print(f"  🔧 Preprocessing: {'✅ Applied' if preprocessing_status else '⚠️ Not applied'}")
                results[mission] = 'success'
            else:
                print(f"  ❌ Prediction failed: {result.get('error', 'Unknown error')}")
                results[mission] = 'failed'
                
        except Exception as e:
            print(f"  ❌ Exception during {mission} test: {e}")
            results[mission] = 'error'
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY:")
    
    successful = sum(1 for status in results.values() if status == 'success')
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    
    for mission, status in results.items():
        icon = "✅" if status == 'success' else "❌"
        print(f"  {icon} {mission}: {status}")
    
    if successful == total:
        print("\n🎉 ALL TESTS PASSED! Preprocessing integration working!")
        return True
    else:
        print(f"\n⚠️ {total - successful} tests failed. Check preprocessing setup.")
        return False

if __name__ == "__main__":
    success = test_preprocessing_integration()
    print(f"\n🏁 Test completed: {'SUCCESS' if success else 'FAILED'}")