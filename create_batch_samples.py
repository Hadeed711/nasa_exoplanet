#!/usr/bin/env python3
"""
🚀 Kepler Batch Prediction Sample Generator
==========================================
Creates a sample CSV file for testing batch prediction in Streamlit app
"""

import pandas as pd
import numpy as np

def create_kepler_sample_csv():
    """Create sample CSV with Kepler features for batch prediction"""
    
    print('🚀 Creating Kepler Dataset Sample CSV for Batch Prediction')
    print('='*60)
    
    # Define Kepler features based on the UI form
    kepler_features = {
        'koi_period': 'Orbital Period (days)',
        'koi_prad': 'Planet Radius (Earth radii)', 
        'koi_dor': 'Distance/Star Radius Ratio',
        'koi_duration': 'Transit Duration (hours)',
        'koi_depth': 'Transit Depth (ppm)',
        'koi_teq': 'Equilibrium Temperature (K)'
    }
    
    print(f'📊 Creating sample data with {len(kepler_features)} features...')
    
    # Generate realistic sample data
    np.random.seed(42)  # For reproducible results
    n_samples = 50
    
    # Create realistic ranges for each feature
    sample_data = {
        'koi_period': np.random.uniform(0.5, 500, n_samples),  # days
        'koi_prad': np.random.uniform(0.5, 20, n_samples),     # Earth radii
        'koi_dor': np.random.uniform(5, 200, n_samples),       # ratio
        'koi_duration': np.random.uniform(0.5, 12, n_samples), # hours
        'koi_depth': np.random.uniform(50, 50000, n_samples),  # ppm
        'koi_teq': np.random.uniform(200, 2000, n_samples)     # K
    }
    
    # Add some realistic patterns to make data more interesting
    for i in range(n_samples):
        # Larger planets tend to have deeper transits
        if sample_data['koi_prad'][i] > 5:
            sample_data['koi_depth'][i] *= np.random.uniform(2, 5)
        
        # Hot planets (shorter periods) tend to be hotter
        if sample_data['koi_period'][i] < 10:
            sample_data['koi_teq'][i] = np.random.uniform(800, 2000)
        
        # Longer periods usually mean longer transit durations
        if sample_data['koi_period'][i] > 100:
            sample_data['koi_duration'][i] *= np.random.uniform(1.5, 3)
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Round to reasonable precision
    df = df.round(3)
    
    # Add some metadata columns
    df['sample_id'] = [f'KEP_{i+1:03d}' for i in range(n_samples)]
    df['notes'] = ['Generated sample for batch prediction testing'] * n_samples
    
    # Reorder columns to put ID first
    cols = ['sample_id'] + list(kepler_features.keys()) + ['notes']
    df = df[cols]
    
    print(f'✅ Generated {len(df)} samples')
    print(f'📋 Features: {list(kepler_features.keys())}')
    print(f'📊 Sample statistics:')
    for feature in kepler_features.keys():
        print(f'   {feature}: {df[feature].min():.2f} - {df[feature].max():.2f}')
    
    # Save to CSV
    filename = 'kepler_batch_prediction_sample.csv'
    df.to_csv(filename, index=False)
    print(f'✅ Saved to: {filename}')
    
    return df, filename

def create_tess_sample_csv():
    """Create sample CSV with TESS features for batch prediction"""
    
    print('\n🛰️ Creating TESS Dataset Sample CSV')
    print('='*40)
    
    # TESS features
    tess_features = {
        'pl_orbper': 'Orbital Period (days)',
        'pl_rade': 'Planet Radius (Earth radii)',
        'pl_trandurh': 'Transit Duration (hours)',
        'pl_trandep': 'Transit Depth (ppm)',
        'st_tmag': 'TESS Magnitude',
        'st_teff': 'Stellar Temperature (K)'
    }
    
    np.random.seed(123)
    n_samples = 30
    
    sample_data = {
        'pl_orbper': np.random.uniform(0.5, 200, n_samples),    # days
        'pl_rade': np.random.uniform(0.3, 15, n_samples),       # Earth radii
        'pl_trandurh': np.random.uniform(0.3, 8, n_samples),    # hours
        'pl_trandep': np.random.uniform(20, 20000, n_samples),  # ppm
        'st_tmag': np.random.uniform(6, 16, n_samples),         # magnitude
        'st_teff': np.random.uniform(3000, 8000, n_samples)     # K
    }
    
    df = pd.DataFrame(sample_data)
    df = df.round(3)
    
    df['sample_id'] = [f'TES_{i+1:03d}' for i in range(n_samples)]
    df['notes'] = ['Generated TESS sample'] * n_samples
    
    cols = ['sample_id'] + list(tess_features.keys()) + ['notes']
    df = df[cols]
    
    filename = 'tess_batch_prediction_sample.csv'
    df.to_csv(filename, index=False)
    print(f'✅ TESS sample saved to: {filename}')
    
    return df, filename

def create_k2_sample_csv():
    """Create sample CSV with K2 features for batch prediction"""
    
    print('\n🌌 Creating K2 Dataset Sample CSV')
    print('='*40)
    
    # K2 features
    k2_features = {
        'pl_orbper': 'Orbital Period (days)',
        'pl_rade': 'Planet Radius (Earth radii)',
        'pl_masse': 'Planet Mass (Earth masses)',
        'pl_dens': 'Planet Density (g/cm³)',
        'pl_insol': 'Insolation (Earth flux)',
        'st_teff': 'Stellar Temperature (K)'
    }
    
    np.random.seed(456)
    n_samples = 25
    
    sample_data = {
        'pl_orbper': np.random.uniform(1, 300, n_samples),      # days
        'pl_rade': np.random.uniform(0.5, 12, n_samples),       # Earth radii
        'pl_masse': np.random.uniform(0.1, 500, n_samples),     # Earth masses
        'pl_dens': np.random.uniform(0.5, 15, n_samples),       # g/cm³
        'pl_insol': np.random.uniform(0.1, 5000, n_samples),    # Earth flux
        'st_teff': np.random.uniform(3500, 7000, n_samples)     # K
    }
    
    df = pd.DataFrame(sample_data)
    df = df.round(3)
    
    df['sample_id'] = [f'K2_{i+1:03d}' for i in range(n_samples)]
    df['notes'] = ['Generated K2 sample'] * n_samples
    
    cols = ['sample_id'] + list(k2_features.keys()) + ['notes']
    df = df[cols]
    
    filename = 'k2_batch_prediction_sample.csv'
    df.to_csv(filename, index=False)
    print(f'✅ K2 sample saved to: {filename}')
    
    return df, filename

def main():
    """Main function to create all sample CSV files"""
    print('🚀 NASA Exoplanet Batch Prediction Sample Generator')
    print('='*70)
    
    # Create sample files for all missions
    kepler_df, kepler_file = create_kepler_sample_csv()
    tess_df, tess_file = create_tess_sample_csv()
    k2_df, k2_file = create_k2_sample_csv()
    
    print('\n' + '='*70)
    print('🎯 BATCH PREDICTION SAMPLE FILES CREATED!')
    print('='*70)
    
    print(f'\n📁 Files Created:')
    print(f'   1. {kepler_file} ({len(kepler_df)} samples)')
    print(f'   2. {tess_file} ({len(tess_df)} samples)')
    print(f'   3. {k2_file} ({len(k2_df)} samples)')
    
    print(f'\n🎯 Usage Instructions:')
    print(f'1. Open your Streamlit app: http://localhost:8503')
    print(f'2. Go to "Batch Prediction" tab')
    print(f'3. Select mission (Kepler/TESS/K2)')
    print(f'4. Upload corresponding CSV file')
    print(f'5. Click "Run Batch Prediction"')
    print(f'6. Download results with predictions!')
    
    print(f'\n📋 Kepler Sample Preview (first 5 rows):')
    print(kepler_df[['sample_id', 'koi_period', 'koi_prad', 'koi_depth', 'koi_teq']].head().to_string(index=False))
    
    print(f'\n✅ All sample files ready for batch prediction testing!')

if __name__ == "__main__":
    main()