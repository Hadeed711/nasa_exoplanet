# Feature Schema - Kepler Exoplanet Dataset

## 📊 Complete Feature List (127 Features)

### Target Variable
- `target`: Encoded target (0=CANDIDATE, 1=CONFIRMED, 2=FALSE_POSITIVE)
- `target_name`: Original string labels

### 🪐 Core Planetary Parameters
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_period` | Orbital period | days | Continuous |
| `koi_prad` | Planetary radius | Earth radii | Continuous |
| `koi_sma` | Semi-major axis | AU | Continuous |
| `koi_incl` | Orbital inclination | degrees | Continuous |
| `koi_teq` | Equilibrium temperature | Kelvin | Continuous |
| `koi_insol` | Insolation flux | Earth flux | Continuous |

### 🌟 Stellar Properties
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_steff` | Stellar effective temperature | Kelvin | Continuous |
| `koi_slogg` | Stellar surface gravity | log10(cm/s²) | Continuous |
| `koi_srad` | Stellar radius | Solar radii | Continuous |
| `koi_smass` | Stellar mass | Solar masses | Continuous |
| `koi_smet` | Stellar metallicity | dex | Continuous |
| `koi_sage` | Stellar age | Gyr | Continuous |

### 🔭 Transit/Detection Parameters
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_duration` | Transit duration | hours | Continuous |
| `koi_depth` | Transit depth | ppm | Continuous |
| `koi_ror` | Planet-star radius ratio | ratio | Continuous |
| `koi_impact` | Impact parameter | dimensionless | Continuous |
| `koi_model_snr` | Signal-to-noise ratio | dimensionless | Continuous |
| `koi_ingress` | Ingress duration | hours | Continuous |

### 📊 Statistical/Quality Indicators
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_score` | Disposition score | 0-1 | Continuous |
| `koi_fpflag_nt` | Not transit-like flag | 0/1 | Binary |
| `koi_fpflag_ss` | Stellar eclipse flag | 0/1 | Binary |
| `koi_fpflag_co` | Centroid offset flag | 0/1 | Binary |
| `koi_fpflag_ec` | Ephemeris contamination flag | 0/1 | Binary |
| `koi_max_sngle_ev` | Max single event statistic | dimensionless | Continuous |
| `koi_max_mult_ev` | Max multiple event statistic | dimensionless | Continuous |
| `koi_num_transits` | Number of transits | count | Integer |

### 🔧 Engineered Features
| Feature | Description | Derivation | Type |
|---------|-------------|------------|------|
| `koi_period_log` | Log orbital period | log1p(koi_period) | Continuous |
| `koi_prad_log` | Log planetary radius | log1p(koi_prad) | Continuous |
| `koi_duration_log` | Log transit duration | log1p(koi_duration) | Continuous |
| `depth_duration_ratio` | Transit efficiency | koi_depth / koi_duration | Continuous |
| `planet_star_radius_ratio` | Size comparison | koi_prad / koi_srad | Continuous |
| `insol_log` | Log insolation | log1p(koi_insol) | Continuous |
| `habitable_zone` | Habitable zone flag | 0.5 ≤ koi_insol ≤ 2.0 | Binary |
| `total_fp_flags` | Total false positive flags | Sum of FP flags | Integer |

### 🌡️ Temperature Categories (if present)
| Feature | Description | Categories | Type |
|---------|-------------|------------|------|
| `stellar_temp_category` | Stellar temperature bins | Cool/Warm/Hot/Very_Hot | Categorical |

### 📏 Orbital Dynamics
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_eccen` | Orbital eccentricity | dimensionless | Continuous |
| `koi_longp` | Longitude of periastron | degrees | Continuous |
| `koi_dor` | Distance over star radius | ratio | Continuous |

### 🎨 Photometric Data
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_kepmag` | Kepler magnitude | magnitude | Continuous |
| Various magnitude bands | g', r', i', z', J, H, K | magnitude | Continuous |

### ⏰ Timing Information
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_time0` | Transit epoch | BJD | Continuous |
| `koi_time0bk` | Transit epoch (BKJD) | BKJD | Continuous |

### 🎯 Model Fitting
| Feature | Description | Units | Type |
|---------|-------------|-------|------|
| `koi_model_dof` | Degrees of freedom | count | Integer |
| `koi_model_chisq` | Chi-square | dimensionless | Continuous |
| `koi_srho` | Fitted stellar density | g/cm³ | Continuous |

## 📊 Feature Statistics Summary

### Continuous Features
- **Count**: ~115 features
- **Scaling**: Applied StandardScaler and MinMaxScaler
- **Range**: Highly variable (some astronomical quantities span orders of magnitude)
- **Missing Values**: Imputed with median values

### Binary Features  
- **Count**: ~8 features
- **Values**: 0/1 encoding
- **Usage**: Quality flags and derived indicators

### Key Feature Importance (Expected)
1. **koi_model_snr**: Primary detection metric
2. **koi_period**: Fundamental orbital property
3. **koi_depth**: Transit signal strength
4. **koi_duration**: Transit characteristics
5. **koi_score**: Disposition confidence
6. **total_fp_flags**: Quality assessment
7. **koi_steff**: Stellar context
8. **depth_duration_ratio**: Engineered signal quality

## 🔍 Usage Notes

- All features are numeric after preprocessing
- StandardScaler applied for linear models
- MinMaxScaler applied for tree-based models
- Log transformations applied to highly skewed features
- Ratio features capture physical relationships
- Flag combinations provide quality assessment

## 🎯 Feature Selection Recommendations

### High Priority (Always Include)
- koi_model_snr, koi_period, koi_depth, koi_duration
- koi_score, total_fp_flags
- koi_steff, koi_srad

### Medium Priority (Usually Include)
- koi_prad, koi_insol, koi_teq
- Engineered ratios and log features
- Individual FP flags

### Lower Priority (Consider for ensemble)
- Photometric magnitudes
- Timing uncertainties
- Model fitting parameters

Use feature importance from tree models and correlation analysis to refine selection for your specific use case.