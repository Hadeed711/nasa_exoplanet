#!/usr/bin/env python3
"""
🧪 Streamlit App Test Script
============================
Test the Streamlit app components without full launch
"""

import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
        
        import pandas as pd
        print("✅ Pandas imported")
        
        import numpy as np
        print("✅ NumPy imported")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported")
        
        import joblib
        print("✅ Joblib imported")
        
        from sklearn.metrics import accuracy_score
        print("✅ Scikit-learn imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🤖 Testing model loading...")
    
    try:
        # Import the main app class
        from streamlit_app import NASAExoplanetDetectionUI
        
        # Initialize the UI (this will attempt to load models)
        app = NASAExoplanetDetectionUI()
        
        print(f"✅ App initialized successfully")
        print(f"📊 Models loaded: {len(app.models)}")
        
        for mission, status in app.model_info.items():
            print(f"   {mission}: {status.get('status', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_app_components():
    """Test individual app components"""
    print("\n🔧 Testing app components...")
    
    try:
        from streamlit_app import NASAExoplanetDetectionUI
        app = NASAExoplanetDetectionUI()
        
        # Test performance data
        performance = app.get_model_performance()
        print(f"✅ Performance data loaded: {len(performance)} missions")
        
        # Test mission info
        mission_info = app.get_mission_info()
        print(f"✅ Mission info loaded: {len(mission_info)} missions")
        
        # Test feature definitions
        kepler_features = app.get_kepler_features()
        tess_features = app.get_tess_features()
        k2_features = app.get_k2_features()
        
        print(f"✅ Feature definitions loaded:")
        print(f"   Kepler: {len(kepler_features)} features")
        print(f"   TESS: {len(tess_features)} features")
        print(f"   K2: {len(k2_features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Component testing error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 NASA Exoplanet Detection Hub - Component Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test model loading
    if not test_model_loading():
        all_passed = False
    
    # Test components
    if not test_app_components():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Streamlit app is ready to launch!")
        print("\n🚀 To start the app, run:")
        print("   streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()