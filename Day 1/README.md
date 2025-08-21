# Day 1: Customer Churn Prediction with Gradient Boosting & SHAP

## 📘 Project Definition
Predict **customer churn** for a telecom provider using **EDA, preprocessing, gradient boosting, and model explainability (SHAP)**.  
Customer retention is crucial to business revenue, making churn prediction a key applied ML task.

---

## 📂 Dataset
- Source: [Telco Customer Churn Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/Telco-Customer-Churn.csv)  
- Rows: 7043  
- Target: `Churn` (Yes/No)

---

## 📊 Exploratory Data Analysis (EDA)

- **Target Distribution**: 26.5% churn rate (imbalanced dataset).  
- **Numeric Features**:
  - `tenure`: churn decreases with longer tenure.  
  - `MonthlyCharges`: higher charges correlate with churn.  
- **Categorical Features**:
  - `Contract`: month-to-month contracts → highest churn.  
  - `PaymentMethod`: electronic check strongly linked to churn.  
- **Correlations**:
  - `tenure` negatively correlated with churn.  

---

## ⚙️ Approach

1. **Data Cleaning**
   - Dropped `customerID`.  
   - Converted `TotalCharges` to numeric & handled missing values.  

2. **Preprocessing**
   - Encoded categorical variables.  
   - Converted churn target (`Yes/No` → `1/0`).  
   - Balanced dataset using **SMOTE**.  

3. **Modeling**
   - Trained **XGBoost Classifier**.  
   - Hyperparameter tuning via **GridSearchCV**.  

4. **Evaluation**
   - Metrics: ROC-AUC, F1-score, Precision/Recall.  
   - Confusion matrix for error analysis.  

5. **Explainability**
   - SHAP summary plots to identify key drivers of churn.  

---

## 📈 Results

- **ROC-AUC**: 0.82  
- **F1-score**: 0.61  
- Top predictors include:
  - **Contract type**
  - **MonthlyCharges**
  - **tenure**
  - **OnlineSecurity**
  - **TechSupport**

---

## 📌 Example Insights

- Customers with **month-to-month contracts** are 3x more likely to churn.  
- **Electronic check users** show highest churn risk.  
- **Tenure > 2 years** significantly reduces churn likelihood.  

---

## 🔑 Skills Demonstrated

- End-to-end ML workflow.  
- Advanced EDA with visualizations.  
- Handling imbalanced data using SMOTE.  
- Gradient Boosting with tuning.  
- Explainability with SHAP.  


