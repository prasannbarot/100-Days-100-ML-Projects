# Day 4: Job Salary Prediction — Transformer Text Embeddings + Tabular Fusion

## 📘 Project Definition
Predict job salary from job postings by fusing Transformer-based text embeddings of job descriptions with structured tabular features (company, location, experience level, remote ratio). This produces robust, modern salary estimates useful for recruiters, job boards, and compensation analytics.

---

## 📂 Dataset
- Place dataset CSV at `Day4_JobSalaryPrediction/data/jobs.csv`.  
- Example columns required (adapt if dataset differs):  
  - `job_title`, `company`, `location`, `experience_level`, `employment_type`, `remote_ratio`, `company_size`, `job_description`, `salary_in_usd`.

---

## 📊 Exploratory Analysis (brief)
- Salary distribution is right-skewed.  
- Experience level and job seniority strongly predict salary.  
- Remote roles show varied salary dispersions.  
- Text embeddings cluster job descriptions by domain (ML, Data Eng, Analytics).

---

## ⚙️ Approach

1. **Text embeddings**  
   - Use `sentence-transformers` (e.g., `all-mpnet-base-v2`) to convert job descriptions to dense vectors.

2. **Tabular features**  
   - One-hot encode categorical features and standardize numeric fields.

3. **Fusion**  
   - Concatenate text embeddings with tabular features.

4. **Models**  
   - LightGBM and XGBoost trained on fused features.  
   - Stacking ensemble (LightGBM + XGBoost → Ridge meta-model).

5. **Evaluation**  
   - Predict log-salary and invert (`exp`) to produce USD estimates.  
   - Metrics: RMSE (USD), MAE (USD), R² (log).

6. **Explainability**  
   - SHAP (TreeExplainer) to identify top tabular drivers.  
   - Attention / embedding inspection for qualitative checks.

---

## 📈 Results (example run)
- **Stacked model** typically improves over single models.  
- Example metrics (your run may vary):  
  - RMSE: \$8,000 - \$15,000 depending on data coverage.  
  - MAE: \$4,000 - \$9,000.  
---

## 🔑 Skills Demonstrated
- Modern NLP with Transformers for production embeddings.  
- Fusion of unstructured and structured data.  
- Ensemble modeling and stacking.  
- Explainable AI with SHAP for tabular drivers.  
- End-to-end reproducible pipeline and artifact saving.

