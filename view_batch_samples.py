#!/usr/bin/env python3
"""
📋 CSV File Viewer for Batch Prediction Samples
==============================================
"""

import pandas as pd

def view_csv_file(filename):
    """View contents of CSV file"""
    print(f'📋 VIEWING FILE: {filename}')
    print('='*60)
    
    try:
        df = pd.read_csv(filename)
        
        print(f'📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns')
        print(f'📋 Columns: {list(df.columns)}')
        
        print(f'\n🔍 First 10 rows:')
        print(df.head(10).to_string(index=False))
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            print(f'\n📊 Statistical Summary:')
            for col in numeric_cols:
                if col != 'sample_id':  # Skip ID columns
                    print(f'{col:15}: {df[col].min():8.2f} - {df[col].max():8.2f} (avg: {df[col].mean():8.2f})')
        
        return True
        
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return False

def main():
    """Main function"""
    print('📁 BATCH PREDICTION SAMPLE FILES VIEWER')
    print('='*70)
    
    files = [
        'kepler_batch_prediction_sample.csv',
        'tess_batch_prediction_sample.csv', 
        'k2_batch_prediction_sample.csv'
    ]
    
    for filename in files:
        if view_csv_file(filename):
            print(f'✅ {filename} is ready for batch prediction!')
        else:
            print(f'❌ {filename} has issues')
        print('\n' + '-'*60 + '\n')
    
    print('🎯 USAGE INSTRUCTIONS:')
    print('1. Open Streamlit app: http://localhost:8503')
    print('2. Go to "Batch Prediction" tab')
    print('3. Select mission (Kepler/TESS/K2)')
    print('4. Upload corresponding CSV file')
    print('5. Click "Run Batch Prediction"')
    print('6. Download results with predictions!')

if __name__ == "__main__":
    main()