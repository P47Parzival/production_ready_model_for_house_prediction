# 🏠 House Price Prediction using Machine Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Sklearn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-red)

---

## 📍 Project Overview

This project aims to build a **machine learning pipeline** to predict **house sale prices** using structured data. The dataset comes from the popular Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

The project covers:
- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Visualization
- Model Building (Linear Regression, Random Forest, XGBoost)
- Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
- Final Model Saving (Production-Ready)

---

## 🚀 Tech Stack

- **Python 3.8+**
- **Pandas, Numpy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **XGBoost**
- **Joblib** (for model persistence)

---

````

---

## 📊 Problem Statement

**Objective:** Predict the `SalePrice` of a house given various numerical and categorical features about the property.

---

## 🛠️ Steps Followed

### ✅ 1. Exploratory Data Analysis (EDA)
- Missing values analysis
- Correlation heatmaps
- Outlier detection
- Distribution analysis (SalePrice, GrLivArea, etc.)

### ✅ 2. Data Cleaning
- Imputed missing values (Median / Mode)
- Removed outliers in GrLivArea

### ✅ 3. Feature Engineering
- Created new features: `TotalSF`, `TotalBath`, `AgeOfHouse`, `AgeSinceRemodel`
- Removed redundant columns
- Applied log1p on skewed features and SalePrice

### ✅ 4. Data Visualization
- Correlation Heatmaps
- Boxplots (OverallQual, Neighborhood)
- Scatterplots (TotalSF vs SalePrice)

### ✅ 5. Modeling
- **Linear Regression (Baseline)**
- **Ridge, Lasso**
- **Random Forest Regressor**
- **XGBoost Regressor**
- Model comparison with RMSE metrics

### ✅ 6. Hyperparameter Tuning
- GridSearchCV for RandomForest
- RandomizedSearchCV for XGBoost

### ✅ 7. Final Model Interpretation
- Feature Importance Visualization
- Error Analysis on worst predictions

### ✅ 8. Deployment Readiness
- Saved final models (`best_xgb_model.pkl`, `scaler.pkl`) for API / app integration

---

## 🔥 Final Model Performance (RMSE)

| Model         | RMSE  |
|---------------|-------|
| Linear        | ~0.17 |
| Ridge         | ~0.16 |
| Lasso         | ~0.17 |
| Random Forest | ~0.13 |
| XGBoost       | **~0.12 (Best)** |

*(Values approximate; vary with random seeds.)*

---

## 📥 How to Run Locally

### 1️⃣ Clone Repo
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model (Optional, Notebooks Provided)

Run through the Jupyter notebooks in order.

### 4️⃣ Predict New Data

```python
import joblib
model = joblib.load('saved_models/best_xgb_model.pkl')
scaler = joblib.load('saved_models/scaler.pkl')

# Scale your new data and predict:
# new_data_scaled = scaler.transform(new_data)
# model.predict(new_data_scaled)
```

---

## 📄 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
```

---

## 🎯 Future Improvements

* Integrate with Streamlit / FastAPI for live predictions
* Automate feature selection pipeline
* Deploy on AWS/GCP as API

---

## 🛡️ License

This project is licensed under the MIT License.

---


## ⭐ Acknowledgment

* Kaggle Dataset: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
