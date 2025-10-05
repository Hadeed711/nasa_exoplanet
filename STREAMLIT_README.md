# 🚀 NASA Exoplanet Detection Hub - Streamlit UI

## Overview
This is a comprehensive Streamlit web application for NASA exoplanet detection using machine learning models trained on official NASA datasets from three major space missions: Kepler, TESS, and K2.

## Features

### 🌌 Mission Overview
- Detailed information about all three NASA missions
- Mission-specific statistics and performance metrics
- Interactive mission cards with key details

### 🔮 Prediction Interface
- Real-time exoplanet classification
- Mission-specific feature input forms
- Interactive prediction results with confidence scores
- Feature importance visualization

### 📈 Performance Dashboard
- Comprehensive model performance metrics
- Interactive charts and comparisons
- Multi-metric radar charts
- Detailed performance tables

### 📁 Batch Prediction
- CSV file upload for multiple predictions
- Downloadable results
- Batch processing visualization
- Summary statistics

### ℹ️ About Section
- Technical specifications
- Technology stack details
- Project information

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Hadeed711/nasa_exoplanet.git
cd nasa_exoplanet
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

## Model Files Required

Ensure the following model files are present:
- `kepler model training/best_exoplanet_model_LightGBM.pkl`
- `TESS model training/best_exoplanet_model_XGBoost.pkl`
- `k2 model training/best_exoplanet_model_CatBoost.pkl`

## Usage

### Single Prediction
1. Select a NASA mission (Kepler, TESS, or K2)
2. Enter the astronomical parameters in the feature input form
3. Click "Predict Exoplanet" to get results
4. View confidence scores and feature importance

### Batch Prediction
1. Go to the "Batch Prediction" tab
2. Select the mission type
3. Upload a CSV file with the required features
4. Click "Run Batch Prediction"
5. Download the results CSV

## Model Performance

| Mission | Model Type | Accuracy | F1 Score | ROC-AUC |
|---------|------------|----------|----------|---------|
| Kepler  | LightGBM   | 95.98%   | 95.99%   | 99.20%  |
| TESS    | XGBoost    | 91.76%   | 91.74%   | 90.18%  |
| K2      | CatBoost   | 91.11%   | 91.05%   | 97.49%  |

## Feature Input Guidelines

### Kepler Mission Features
- Orbital Period (days): 0.1 - 1000
- Planet Radius (Earth radii): 0.1 - 50
- Distance/Star Radius Ratio: 1 - 1000
- Transit Duration (hours): 0.1 - 24
- Transit Depth (ppm): 1 - 100000
- Equilibrium Temperature (K): 100 - 3000

### TESS Mission Features
- Orbital Period (days): 0.1 - 1000
- Planet Radius (Earth radii): 0.1 - 50
- Transit Duration (hours): 0.1 - 24
- Transit Depth (ppm): 1 - 100000
- TESS Magnitude: 1 - 20
- Stellar Temperature (K): 2000 - 10000

### K2 Mission Features
- Orbital Period (days): 0.1 - 1000
- Planet Radius (Earth radii): 0.1 - 50
- Planet Mass (Earth masses): 0.1 - 1000
- Planet Density (g/cm³): 0.1 - 20
- Insolation (Earth flux): 0.1 - 10000
- Stellar Temperature (K): 2000 - 10000

## Technical Architecture

### Backend Pipeline
- Model loading and management
- Prediction processing
- Feature validation
- Error handling

### Frontend Interface
- Streamlit-based web application
- Interactive Plotly visualizations
- Responsive design
- Real-time updates

### Data Processing
- Mission-specific feature processing
- Input validation and sanitization
- Batch processing capabilities
- CSV export functionality

## Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment
The app is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS
- Google Cloud Platform
- Azure

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

## API Integration

The application can be extended with REST API endpoints:

```python
# Example API endpoint structure
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mission = data['mission']
    features = data['features']
    
    result = pipeline.predict_single(mission, features)
    return jsonify(result)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA for providing the exoplanet datasets
- Space Apps Challenge organizers
- Open source ML community
- Streamlit team for the excellent framework

## Contact

For questions or support, please contact:
- GitHub: [@Hadeed711](https://github.com/Hadeed711)
- Repository: [nasa_exoplanet](https://github.com/Hadeed711/nasa_exoplanet)

## Screenshots

[Screenshots would be added here showing the different sections of the UI]

---

🚀 Built with ❤️ for NASA Space Apps Challenge 2024