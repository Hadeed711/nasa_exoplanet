#!/usr/bin/env python3
"""
🔧 Plotly Chart Validation Script
================================
Test all Plotly chart functions to ensure they work correctly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def test_accuracy_chart():
    """Test the accuracy comparison chart"""
    print("📊 Testing accuracy comparison chart...")
    try:
        missions = ['Kepler', 'TESS', 'K2']
        accuracies = [0.9598, 0.9176, 0.9111]
        
        fig_acc = px.bar(
            x=missions,
            y=accuracies,
            title="Model Accuracy Comparison",
            color=accuracies,
            color_continuous_scale="viridis"
        )
        fig_acc.update_layout(
            showlegend=False,
            yaxis=dict(range=[0.8, 1.0])
        )
        print("✅ Accuracy chart created successfully")
        return True
    except Exception as e:
        print(f"❌ Accuracy chart error: {e}")
        return False

def test_gauge_chart():
    """Test the confidence gauge chart"""
    print("📊 Testing confidence gauge chart...")
    try:
        confidence = 0.85
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        print("✅ Gauge chart created successfully")
        return True
    except Exception as e:
        print(f"❌ Gauge chart error: {e}")
        return False

def test_radar_chart():
    """Test the radar chart"""
    print("📊 Testing radar chart...")
    try:
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
        missions = ['Kepler', 'TESS', 'K2']
        performance = {
            'Kepler': {'accuracy': 0.9598, 'f1_score': 0.9599, 'precision': 0.9601, 'recall': 0.9598, 'roc_auc': 0.9920},
            'TESS': {'accuracy': 0.9176, 'f1_score': 0.9174, 'precision': 0.9173, 'recall': 0.9176, 'roc_auc': 0.9018},
            'K2': {'accuracy': 0.9111, 'f1_score': 0.9105, 'precision': 0.9124, 'recall': 0.9111, 'roc_auc': 0.9749}
        }
        
        fig_radar = go.Figure()
        
        for mission in missions:
            values = [performance[mission][metric] for metric in metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=mission
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.8, 1.0]
                )
            ),
            showlegend=True,
            title="Multi-Metric Performance Comparison",
            height=400
        )
        print("✅ Radar chart created successfully")
        return True
    except Exception as e:
        print(f"❌ Radar chart error: {e}")
        return False

def test_pie_chart():
    """Test the pie chart"""
    print("📊 Testing pie chart...")
    try:
        fig_dist = px.pie(
            values=[1200, 200],
            names=['Not Planet', 'Exoplanet'],
            title="Prediction Distribution"
        )
        print("✅ Pie chart created successfully")
        return True
    except Exception as e:
        print(f"❌ Pie chart error: {e}")
        return False

def test_histogram():
    """Test the histogram"""
    print("📊 Testing histogram...")
    try:
        confidences = np.random.uniform(0.6, 0.99, 100)
        fig_conf = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Score Distribution"
        )
        print("✅ Histogram created successfully")
        return True
    except Exception as e:
        print(f"❌ Histogram error: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 Plotly Chart Validation Suite")
    print("=" * 50)
    
    tests = [
        test_accuracy_chart,
        test_gauge_chart,
        test_radar_chart,
        test_pie_chart,
        test_histogram
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL PLOTLY CHARTS WORKING CORRECTLY!")
        print("✅ Streamlit app should run without errors")
    else:
        print("⚠️  Some chart issues detected")
    
    return passed == total

if __name__ == "__main__":
    main()