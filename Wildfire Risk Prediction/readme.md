# Day 6 : Wildfire Risk Prediction 

## Overview

This project predicts wildfire risk and facilitates early detection using NASA's FIRMS (Fire Information for Resource Management System) VIIRS satellite data. The Jupyter notebook (`Day6_WildfireRisk.ipynb`) processes FIRMS data (or synthetic data if unavailable), performs exploratory data analysis (EDA), engineers features, trains machine learning models, and generates visualizations to forecast next-day fire occurrence in spatial grid cells.

The project is designed to showcase advanced data science skills, including geospatial time-series analysis, handling imbalanced datasets, and creating professional visualizations. It is optimized for low computational resources, making it suitable for personal machines or educational demonstrations.

**Key Components**:
- **Data Processing**: Loads and grids FIRMS VIIRS data for spatial analysis.
- **EDA**: Visualizes fire distributions, brightness, fire radiative power (FRP), and confidence levels.
- **Feature Engineering**: Extracts temporal, spatial, and intensity-based features.
- **Modeling**: Compares LightGBM, XGBoost, and RandomForest classifiers.
- **Evaluation**: Uses AUC, Average Precision (AP), precision, recall, and F1-score, with visualizations like precision-recall curves and risk heatmaps.

## Background and Theory

### Wildfire Risk Prediction
Wildfires are uncontrolled vegetation fires influenced by weather, fuel load, and ignition sources. Predicting wildfire risk involves estimating the probability of ignition and spread, which this project models as a binary classification problem (fire/no fire on the next day).

**Key Concepts**:
- **Ignition Risk**: Driven by factors like temperature, humidity, wind, and recent fire activity. This project uses satellite-derived fire detections to model historical patterns.
- **Fire Spread**: Fire spread is governed by **Rothermel's fire spread model**:

$$
R = \frac{I_R \cdot \xi \cdot (1 + \phi_w + \phi_s)}{\rho_b \cdot \epsilon \cdot Q_{ig}}
$$

Where:

- \( R \): Spread rate  
- \( I_R \): Reaction intensity  
- \( \xi \): Propagating flux ratio  
- \( \phi_w \): Wind factor  
- \( \phi_s \): Slope factor  
- \( \rho_b \): Bulk density  
- \( \epsilon \): Effective heating number  
- \( Q_{ig} \): Heat of pre-ignition  

> This project simplifies the model to focus on **ignition probability** due to data constraints.

- **Class Imbalance**: Fire events are rare, leading to imbalanced datasets. Techniques like `scale_pos_weight` in boosting models or `class_weight="balanced"` in RandomForest address this by prioritizing the minority class (fire occurrences).

### Satellite Data: FIRMS and VIIRS
- **FIRMS**: NASA's system for near real-time fire data from MODIS and VIIRS satellites.
- **VIIRS (Visible Infrared Imaging Radiometer Suite)**: Provides 375m resolution fire detections with:
  - **Brightness Temperature (bright_ti4/ti5)**: Measures thermal anomalies in mid-infrared channels (TI4: ~3.7μm, TI5: ~11.5μm). Higher `bright_ti4` indicates hotter fires.
  - **Fire Radiative Power (FRP)**: Quantifies fire intensity in megawatts, proportional to biomass burned.
  - **Confidence**: Categorized as low/nominal/high based on detection reliability.
  - **Day/Night**: Nighttime detections may be more reliable due to reduced solar interference.

**Theory: Fire Detection via Thermal Contrast** Fire detection relies on **thermal contrast**, modeled by **Planck's law**:

$$
B(\lambda, T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc/(\lambda kT)} - 1}
$$

Where:

- \( B(\lambda, T) \): Spectral radiance  
- \( \lambda \): Wavelength  
- \( T \): Temperature  
- \( h \): Planck's constant  
- \( c \): Speed of light  
- \( k \): Boltzmann constant  

 **Insight**: Hotter fires emit more radiation at shorter wavelengths (e.g., TI4 band), enabling detection of **smaller or hotter fires**.

### Feature Engineering
The project creates features to capture:
- **Temporal Trends**: Rolling counts of fires (1, 3, 7 days) and trends (difference in counts).
- **Intensity**: Average/maximum brightness (TI4/TI5), FRP, and confidence scores.
- **Spatial Patterns**: Distance to nearest fire using `cKDTree` for efficiency and grid cell coordinates.
- **Seasonality**: Cyclical encoding of day-of-year (`doy_sin`, `doy_cos`).

### Machine Learning Models
**LightGBM / XGBoost: Gradient Boosting for Tabular Data**
**LightGBM** and **XGBoost** are gradient boosting frameworks optimized for structured/tabular datasets. They effectively handle class imbalance using **weighted loss functions**.

Objective Function

The general form of the objective function is:

$$
L = \sum l(y_i, \hat{y}_i) + \Omega(f)
$$

Where:

- \( l(y_i, \hat{y}_i) \): Loss function (e.g., log loss)  
- \( \Omega(f) \): Regularization term to control tree complexity  

These models are widely used for classification, regression, and ranking tasks due to their speed, accuracy, and flexibility.

- **RandomForest**: Ensemble of decision trees, robust to overfitting and interpretable.
- **Evaluation Metrics**:
  - **AUC-ROC**: Measures ranking ability across thresholds.
  - **Average Precision (AP)**: Prioritizes precision-recall trade-off for imbalanced data.
  - **Precision/Recall/F1**: Balances true positives and false positives.

## Installation and Requirements

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Dependencies
Install required packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost joblib scipy requests
```

**Full Dependency List**:
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computations and array operations.
- `matplotlib`, `seaborn`: Visualization tools for professional plots and graphs.
- `scikit-learn`: Machine learning utilities (preprocessing, metrics, model selection).
- `lightgbm`, `xgboost`: Gradient boosting frameworks for efficient modeling.
- `joblib`: Model serialization for saving and loading trained models.
- `scipy`: Efficient spatial distance calculations using `cKDTree`.
- `requests`: HTTP requests for downloading FIRMS data.

## Usage

1. **Prepare Data**:
   - The notebook automatically downloads a FIRMS sample CSV from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/content/notebooks/sample_viirs_snpp_071223.csv) if not found in `data/`.
   - Place custom FIRMS data in `Day6_WildfireRisk/data/firms_sample.csv` for processing.
   - If insufficient data is provided (e.g., single-day data), synthetic data is generated with realistic seasonal patterns.

2. **Run the Notebook**:
   ```bash
   jupyter notebook Day6_WildfireRisk.ipynb
   ```
   The notebook:
   - Loads and preprocesses FIRMS data, handling single-day cases with a random train-test split.
   - Performs EDA with spatial, temporal, and intensity-based visualizations.
   - Engineers features for temporal trends, fire intensity, and spatial relationships.
   - Trains and compares LightGBM, XGBoost, and RandomForest models, selecting the best based on AUC.
   - Generates visualizations (precision-recall curves, feature importance, risk heatmaps).
   - Saves artifacts (model, scaler, dataset) in `artifacts/`.

3. **Artifacts**:
   - **Model**: `artifacts/best_wildfire_model.joblib`
   - **Scaler**: `artifacts/scaler.joblib`
   - **Dataset**: `artifacts/panel_dataset.csv`
   - Load model for predictions:
     ```python
     import joblib
     model = joblib.load('artifacts/best_wildfire_model.joblib')
     ```

4. **Customization**:
   - Adjust `GRID_RES` (default: 0.5 degrees) for finer or coarser spatial grids (note: finer grids increase computation time).
   - Uncomment weather API integration (Open-Meteo) in the notebook to add weather features like temperature and humidity, if desired.
   - Use multi-day FIRMS data for accurate temporal predictions and to enable time-based train-test splitting.

## Data Sources
- **FIRMS VIIRS Sample**: [Sample CSV](https://firms.modaps.eosdis.nasa.gov/content/notebooks/sample_viirs_snpp_071223.csv)
- **Synthetic Data**: Generated with a Poisson distribution for fire counts, mimicking seasonal patterns within a bounding box (default: 0°-30°E, 0°-13.5°N).
- **Full FIRMS Data**: Download from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (requires registration).
- **Optional Weather Data**: Open-Meteo API (commented out in the notebook to reduce runtime).

**Note**: The provided sample data contains single-day records (2023-07-12), limiting temporal features and requiring a synthetic target based on FRP percentiles. For accurate predictions, use multi-day FIRMS data.

## Key Features
- **EDA Visualizations**:
  - Spatial scatter plot (brightness as size, FRP as color) to visualize fire locations and intensity.
  - Histograms for `bright_ti4`, `bright_ti5`, and `frp` distributions.
  - Count plot for `daynight` and `confidence` distributions.
  - Correlation heatmap for fire metrics (`bright_ti4`, `bright_ti5`, `frp`).
- **Feature Engineering**:
  - **Temporal**: Fire counts over 1, 3, and 7 days, fire count trends.
  - **Intensity**: Average/maximum `bright_ti4`, `bright_ti5`, `frp`, and confidence scores.
  - **Spatial**: Grid cell coordinates, distance to nearest fire (computed efficiently with `cKDTree`).
  - **Other**: Day/night detection ratio, cyclical encoding of day-of-year (`doy_sin`, `doy_cos`).
- **Model Comparison**: LightGBM, XGBoost, and RandomForest, optimized for low computational resources.
- **Single-Day Data Handling**: Uses random train-test split and synthetic target (based on high FRP) when temporal data is limited.

## Model Results
- **Models Compared**: LightGBM, XGBoost, RandomForest.
- **Selection**: Best model chosen based on AUC, typically LightGBM for its speed and ability to handle imbalanced data.
- **Metrics**: AUC (~0.7-0.9 on synthetic data), Average Precision, precision, recall, F1-score, and confusion matrix.
- **Visualizations**:
  - Precision-recall curves comparing all models with AP scores.
  - Feature importance bar plot (top 10 features) using the "viridis" palette.
  - Spatial risk heatmap visualizing predicted fire probabilities with "YlOrRd" colormap.

**Example Output** (varies by data):
- Train/Test Shapes: Adjusted to ensure non-empty test set (e.g., 80/20 random split for single-day data).
- AUC: ~0.7-0.9 (synthetic data); real multi-day data may differ.
- AP: High for models handling class imbalance effectively.

## Limitations and Improvements
### Limitations
- Single-day data (e.g., sample CSV) restricts temporal modeling, requiring synthetic targets based on FRP percentiles.
- Grid-based aggregation may miss fine-grained spatial patterns.
- Weather and vegetation data (e.g., NDVI) are not included due to computational constraints, limiting real-world accuracy.

### Potential Improvements
- **Data Enhancements**:
  - Use multi-day FIRMS data for true temporal predictions and time-based train-test splitting.
  - Integrate weather data from Open-Meteo (temperature, humidity, wind speed).
  - Incorporate NDVI from Landsat or Sentinel satellites to capture vegetation fuel load.
- **Modeling**:
  - Explore spatial-temporal models like ConvLSTM or Graph Neural Networks with access to more computational resources.
  - Implement hyperparameter tuning using tools like Optuna for improved model performance.
- **Deployment**:
  - Develop a Streamlit or Flask web application for interactive fire risk visualization.
  - Deploy the model as an API for real-time wildfire risk predictions.

**Theoretical Resources**:
- Rothermel's Fire Spread Model: [USFS Publications](https://www.fs.usda.gov/treesearch/pubs/24630)
- VIIRS Fire Detection: [NASA FIRMS Documentation](https://firms.modaps.eosdis.nasa.gov/documentation/)
- Gradient Boosting: [Friedman et al., 2001](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Report bugs or suggest enhancements via GitHub Issues.

## Contact
For questions or feedback, reach out to [barotprasann@gmail.com] or open an issue on GitHub.
