# Credit Risk Default Prediction using XGBoost

> An end-to-end Machine Learning project to predict customer **Probability of Default (PD)** using XGBoost, explain predictions with SHAP, and optimize lending decisions through threshold-based business strategy.

---

## Project Overview

Financial institutions evaluate the likelihood that a customer will default before approving loans or credit cards. This project develops a machine learning model to estimate the **Probability of Default (PD)** using customer behavioral and financial attributes.

The project covers the complete machine learning lifecycle including data preprocessing, feature engineering, model development, explainability, hyperparameter tuning, and business decision analysis.

---

## Business Problem

The objective is to accurately identify high-risk customers before extending credit while balancing customer acquisition and portfolio risk.

The solution focuses on:

- Predicting customer Probability of Default (PD)
- Improving credit approval decisions
- Reducing portfolio credit risk
- Explaining model predictions using SHAP
- Evaluating different lending strategies through threshold optimization

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-Learn |
| Explainability | SHAP |
| Model Optimization | GridSearchCV |
| Visualization | Matplotlib |

---

## Project Workflow

```
Raw Customer Data
        │
        ▼
Data Cleaning
        │
        ▼
Missing Value Treatment
        │
        ▼
Feature Engineering
        │
        ▼
Categorical Encoding
        │
        ▼
Train/Test Split
        │
        ▼
XGBoost Model
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
SHAP Explainability
        │
        ▼
Threshold Optimization
        │
        ▼
Business Recommendation
```

---

## Dataset

- Sampled version of the American Express Default Prediction dataset
- 7,000 customer records
- 161 engineered features after preprocessing
- Binary target variable indicating customer default

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed features with more than 50% missing values
- Median imputation for remaining numeric missing values
- Label encoding for categorical features
- Date feature engineering
  - Statement Year
  - Statement Month
  - Statement Quarter
- Train/Test split using Stratified Sampling to preserve class distribution

---

## Machine Learning Model

The final model was developed using **XGBoost Classifier** due to its strong performance on structured/tabular datasets.

### Hyperparameter Tuning

Hyperparameters were optimized using **GridSearchCV** with 3-fold cross-validation.

Best Parameters:

```python
{
'colsample_bytree': 0.8,
'learning_rate': 0.1,
'max_depth': 3,
'n_estimators': 100,
'subsample': 1.0
}
```

---

## Model Performance

| Metric | Score |
|---------|-------|
| ROC-AUC | **0.9000** |
| Accuracy | **82.07%** |
| Precision | **66.57%** |
| Recall | **64.23%** |
| F1 Score | **65.38%** |

---

## Confusion Matrix

| | Predicted Non-Default | Predicted Default |
|---|---:|---:|
| Actual Non-Default | 912 | 119 |
| Actual Default | 132 | 237 |

---

## Feature Importance

The most influential features identified by XGBoost were:

- P_2
- D_51
- R_3
- S_15
- D_129
- B_8
- S_3
- D_75
- B_5
- S_7

Feature importance was further validated using SHAP explainability.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to understand both global and individual model predictions.

The project includes:

- Global Feature Importance
- SHAP Summary Plot
- SHAP Waterfall Plot

These explain how each feature contributes toward increasing or decreasing the predicted Probability of Default.

---

## Business Strategy Analysis

Different Probability of Default thresholds were evaluated to determine the optimal lending strategy.

| Threshold | Approval Rate | Portfolio Default Rate | Default Capture |
|----------:|--------------:|-----------------------:|----------------:|
| 0.30 | 69.57% | 9.65% | 74.53% |
| 0.40 | 71.79% | 11.34% | 69.11% |
| 0.50 | 74.57% | 12.64% | 64.23% |
| 0.60 | 77.14% | 13.61% | 60.16% |
| 0.70 | 79.86% | 15.12% | 54.20% |

---

## Business Recommendation

A threshold of **0.40** provides a balanced trade-off between:

- Customer approval rate
- Portfolio default rate
- Default capture

This demonstrates how machine learning outputs can be translated into practical lending strategies based on an organization's risk appetite.

---

## Repository Structure

```
Credit-Risk-Default-Prediction/
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Development.ipynb
│   ├── 03_Model_Explainability.ipynb
│   └── 04_Business_Strategy.ipynb
│
├── images/
│
├── model/
│
├── data/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Compare XGBoost with LightGBM and CatBoost
- Probability Calibration
- Expected Loss (PD × LGD × EAD)
- MLflow Experiment Tracking
- Model Deployment using FastAPI or Streamlit

---

## Skills Demonstrated

- Machine Learning
- Predictive Analytics
- Credit Risk Modeling
- Probability of Default (PD)
- Feature Engineering
- Hyperparameter Tuning
- Model Evaluation
- SHAP Explainability
- Business Strategy Analysis
- Threshold Optimization
- Python
- Pandas
- Scikit-Learn
- XGBoost

---

## Author

**Rupesh Reddy**

If you found this project useful, feel free to ⭐ the repository.
