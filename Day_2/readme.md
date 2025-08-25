# Day 2: Credit Card Fraud Detection with Anomaly Detection & Ensemble Models

## Project Definition
Detect fraudulent credit card transactions using **EDA, anomaly detection, and ensemble models**.  
The dataset is **highly imbalanced** (~0.17% fraud cases), making this problem challenging and requiring specialized handling.

---

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Rows: 284,807 transactions  
- Features: 30 (PCA transformed for confidentiality)  
- Target: `Class` (1 = Fraud, 0 = Legitimate)

---

## Exploratory Data Analysis (EDA)

- **Target Distribution**:  
  - Fraud cases: 492 (0.172%).  
  - Non-fraud: 284,315.  
  - **Imbalance ratio** ≈ 1:577.  

- **Key Observations**:
  - Transaction amounts for frauds are typically **small to medium**.  
  - PCA components show certain clusters where frauds concentrate.  
  - Time feature shows **no strong seasonal trend**.  

- **Visualization Insights**:
  - Boxplots of `Amount` reveal fraud transactions often have lower amounts.  
  - t-SNE and PCA 2D projections show **fraud transactions form sparse clusters**.  

---

## Approach

1. **Data Preprocessing**
   - Scaled `Amount` and `Time` using **StandardScaler**.  
   - Stratified split into train/test sets. 
   - Perform Feature Engineering and create one more valuable feature `LogAmount`. 

2. **Handling Imbalance**
   - Techniques tested:  
     - **SMOTE** oversampling.  
     - **Undersampling** majority class.  
     - **Class weights** in algorithms.  

3. **Modeling Approaches**
   - **Logistic Regression with class weights**.  
   - **Random Forest** tuned for recall.  
   - **XGBoost with scale_pos_weight**.  
   - **Isolation Forest** (unsupervised anomaly detection).  
   - **Voting Classifier Ensemble** of the best models.  

4. **Evaluation**
   - Metrics: Precision, Recall, F1, ROC-AUC, Precision-Recall AUC.  
   - Special focus on **Recall (Fraud Detection Rate)**.  
   - Used **confusion matrix heatmaps** for interpretability.  

5. **Explainability**
   - SHAP analysis on tree-based models.  
   - Identified PCA components that contribute most to fraud detection.  

---

## Results

- **XGBoost (tuned)**:
  - ROC-AUC: 0.9758
  - PR-AUC: 0.8611
  - Precision: 0.7387 
  - Recall: 0.8367 
  - F1: 0.7847

- **Isolation Forest**:  
  - Detected 70% of fraud cases without labels, showing anomaly potential.  

- **Voting Ensemble** (XGBoost + SMOT + Logistic):  
  - Accuracy: 0.9993153330290369
  - Precision: 0.780952380952381
  - Recall: 0.8367346938775511
  - F1 Score: 0.8078817733990148

---

## Example Insights

- Fraud transactions are usually **smaller in amount** than legitimate ones.  
- Certain PCA-transformed features (V14, V17, V12) strongly separate fraud from normal cases.  
- Ensemble approaches outperform single models due to the rare-event nature.  

---

## Skills Demonstrated

- Advanced EDA on imbalanced datasets.  
- Multiple resampling strategies (SMOTE, undersampling, class weights).  
- Combination of **supervised + unsupervised** approaches.  
- Model explainability via SHAP on PCA components.  
- Ensemble modeling for improved robustness.  
