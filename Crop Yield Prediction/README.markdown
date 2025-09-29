# Crop Yield Prediction: Climate-Aware Machine Learning Model

## Project Overview
This project predicts crop yields using climate, pesticide, and agricultural data, leveraging advanced machine learning techniques. It builds on the **Crop Yield Prediction Dataset** from Kaggle, merging raw data files (`pesticides.csv`, `rainfall.csv`, `temp.csv`, `yield.csv`) into a unified dataset for robust analysis. The project features enhanced exploratory data analysis (EDA), feature engineering, and multiple machine learning models (Random Forest, XGBoost, LightGBM, and an ensemble) to forecast crop yields globally. Visualizations and SHAP explainability provide deep insights into feature impacts, making this a comprehensive tool for agricultural analytics.

This repository is ideal for data scientists, agronomists, and researchers interested in climate-aware crop yield prediction. It was developed as a personal project to showcase data science skills and is shared on [GitHub](https://github.com/prasannbarot) and [LinkedIn](https://www.linkedin.com/in/prasannbarot/).

## Dataset 9in the data file)
The dataset is sourced from the [Crop Yield Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset), combining:
- **pesticides.csv**: Pesticide usage by country, crop, and year (source: FAO).
- **rainfall.csv**: Annual rainfall by country (source: World Bank).
- **temp.csv**: Average temperature by country and year (source: World Bank).
- **yield.csv**: Crop yield data (hg/ha) by country, crop, and year (source: FAO).
- **enhanced_yield_df.csv**: Merged and preprocessed dataset with additional features (e.g., rolling rainfall averages, lagged yields).

**Key Features**:
- Target: Crop yield (`hg/ha_yield`).
- Predictors: Rainfall (`average_rain_fall_mm_per_year`), temperature (`avg_temp`), pesticides (`pesticides_tonnes`), encoded country (`Area_encoded`), crop type (`Item_encoded`), and engineered features (e.g., `rainfall_rolling_5y`, `temp_pest_interaction`).
- Scope: Global, covering multiple crops (e.g., maize, wheat, rice) across 100+ countries.

## Project Structure
- **data/**: Contains raw input files (`pesticides.csv`, `rainfall.csv`, `temp.csv`, `yield.csv`) and the processed `enhanced_yield_df.csv`.
- **crop_yield_prediction.py**: Main Python script with data preprocessing, EDA, feature engineering, model training, and evaluation.
- **README.md**: This file, providing project overview and instructions.
- **merge_dataset**: This file contain feature engineering and process of marging 4 csv files to make one usefull marged dataset.
- **output.png**: This is output screenshort of the model performance as SHAP plot representation. 

## Features
- **Data Preprocessing**: Merges raw CSV files, handles missing values (forward-fill, median imputation), and standardizes units.
- **Exploratory Data Analysis (EDA)**:
  - Visualizations: Yield distributions, violin plots by crop, boxplots by country, time-series trends, correlation heatmaps, pairplots, and facet grids.
  - Outlier detection and missing value analysis.
- **Feature Engineering**:
  - Encoded categorical variables (`Area`, `Item`).
  - Added rolling averages (`rainfall_rolling_5y`), lagged yields (`yield_lag1`), and interaction terms (`temp_pest_interaction`).
- **Machine Learning**:
  - Models: Random Forest, XGBoost, LightGBM, and an ensemble (averaged predictions).
  - Hyperparameter tuning via GridSearchCV.
  - Cross-validation and learning curves for robustness.
  - Metrics: RMSE, MAE, R².
- **Explainability**: SHAP summary and waterfall plots to interpret feature contributions.
- **Visualizations**: Enhanced with Seaborn (e.g., violin plots, facet grids) for professional-grade insights.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/prasannbarot/crop-yield-prediction.git
   cd crop-yield-prediction
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) and place `pesticides.csv`, `rainfall.csv`, `temp.csv`, `yield.csv` in the `data/` directory.

## Usage
1. Ensure the raw CSV files are in the `data/` directory.
2. Run the script:
   ```bash
   python crop_yield_prediction.py
   ```
3. Outputs:
   - **Merged Dataset**: `data/enhanced_yield_df.csv`.
   - **Visualizations**: Saved in `figures/` (e.g., yield distributions, SHAP plots).
   - **Console Output**: Model performance (RMSE, MAE, R²), cross-validation scores, and insights.
4. Modify the script to:
   - Add new features (e.g., NDVI via external APIs).
   - Experiment with other models (e.g., Prophet for time-series forecasting).
   - Filter specific crops or countries.

## Key Insights
- **Key Predictors**: Rainfall trends, lagged yields, and temperature-pesticide interactions are critical drivers of crop yield (per SHAP analysis).
- **Global Trends**: Yields vary significantly by country (e.g., higher in developed nations) and crop (e.g., sugarcane vs. maize).
- **Model Performance**: The ensemble model (Random Forest + XGBoost + LightGBM) achieves the best R², with robust cross-validation scores.
- **Recommendations**: Integrate satellite data (e.g., NDVI) or explore time-series models for improved forecasting.

## Visualizations
Below are sample outputs (generated in `figures/`):
- **Yield Distribution**: Histogram showing the spread of crop yields.
- **Country-Level Analysis**: Boxplots of yields for top 10 countries.
- **Crop Trends**: Violin plots and time-series trends by crop type.
- **SHAP Plots**: Feature importance and individual prediction explanations.

## Future Improvements
- Integrate satellite-derived features (e.g., NDVI) using Google Earth Engine.
- Explore deep learning models (e.g., LSTM) for temporal patterns.
- Add spatial analysis for country-specific models.
- Incorporate additional datasets (e.g., soil nutrients, irrigation data).

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

## Acknowledgments
- **Dataset**: [Crop Yield Prediction Dataset](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset) by Patel Ris.
- **Sources**: FAO (yield, pesticides), World Bank (rainfall, temperature).
- **Tools**: Pandas, Scikit-learn, XGBoost, LightGBM, SHAP, Seaborn, Matplotlib.

## Contact
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/prasannbarot/) or open an issue on GitHub for questions or feedback.

---

*Built with passion for data science and sustainable agriculture.*
