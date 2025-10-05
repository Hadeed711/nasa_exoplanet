#!/usr/bin/env python3
"""
🚀 NASA Exoplanet Detection Hub - Application Launcher
=====================================================
Quick launcher for the Streamlit application
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'scikit-learn', 'lightgbm', 'xgboost', 'catboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist"""
    model_files = [
        'kepler model training/best_exoplanet_model_LightGBM.pkl',
        'TESS model training/best_exoplanet_model_XGBoost.pkl',
        'k2 model training/best_exoplanet_model_CatBoost.pkl'
    ]
    
    missing_files = []
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} missing")
    
    if missing_files:
        print(f"\n⚠️  Missing model files: {len(missing_files)}")
        print("Some prediction features may not work properly.")
        return False
    
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching NASA Exoplanet Detection Hub...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped. Thank you for using NASA Exoplanet Detection Hub!")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")

def main():
    """Main launcher function"""
    print("🚀 NASA Exoplanet Detection Hub - Launcher")
    print("=" * 50)
    
    print("\n🔍 Checking system requirements...")
    packages_ok = check_requirements()
    
    print("\n🔍 Checking model files...")
    models_ok = check_model_files()
    
    if packages_ok:
        print("\n✅ All requirements satisfied!")
        launch_streamlit()
    else:
        print("\n❌ Please install missing requirements first:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()