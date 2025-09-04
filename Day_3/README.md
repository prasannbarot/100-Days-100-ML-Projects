# Day 3: Short-term Electricity Demand Forecasting (Household Level)

## Project Definition
Forecast next-hour electricity consumption per household using historical smart meter data.
Helps optimize energy distribution, reduce costs, and manage grid load.

---

## Dataset
- Source: [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- Rows: 2,075,259  
- Features: `Global_active_power`, `Voltage`, `Sub_metering_1/2/3`, etc.  
- Target: `Global_active_power` (kW)

---

## Exploratory Data Analysis (EDA)
- Target distribution shows clear daily/weekly cycles.  
- Correlation heatmap highlights strong links between `Global_active_power` and sub-meterings.  
- Rolling statistics reveal trends and variance over hours.

---

## Approach
1. Data preprocessing & hourly resampling  
2. Feature engineering (lags, rolling mean/std, time features)  
3. Regression models: Linear Regression, Random Forest, XGBoost  
4. Ensemble: Voting Regressor  
5. Evaluation: RMSE, MAE, R², plots  
6. SHAP for feature importance

---

## Results
- RMSE: 0.22 kW  
- MAE: 0.15 kW  
- R²: 0.87  
- Ensemble outperforms single models, captures daily/weekly trends

---

## Skills Demonstrated
- Time series feature engineering  
- Regression modeling & ensembles  
- Handling real-world noisy household energy data  
- Model explainability with SHAP

---

## Installation

To get started, install the required packages:

```bash
pip install ucimlrepo
```