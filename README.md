# 📊 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **AI-powered application to identify telecom customers at risk of churn — with explainability using SHAP values and retention strategy recommendations.**

---

## 🎯 Problem Statement

Customer churn is one of the most costly challenges in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. The objective of this project is to:

- **Predict** which customers are likely to churn
- **Identify** the key drivers behind churn
- **Recommend** data-driven retention strategies
- **Deploy** a live prediction tool for business use

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── Customer_churn_prediction_EDA.ipynb     # Exploratory Data Analysis
├── customer_churn_prediction_model.ipynb   # Feature Engineering, Model Building, SHAP
├── app1.py                                 # Streamlit Web Application
│
├── churn_model.pkl                         # Saved Logistic Regression model
├── scaler.pkl                              # Saved StandardScaler
├── feature_columns.pkl                     # Saved feature column names
├── image_1.png                             # Background image for app
│
└── README.md
```

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** 7,043 customers × 21 features
- **Target Variable:** `Churn` (Yes / No)
- **Class Distribution:** 73.5% Non-Churn | 26.5% Churn (imbalanced)

---

## 🔍 Exploratory Data Analysis — Key Findings

| Factor | Insight |
|--------|---------|
| **Contract Type** | Month-to-month customers churn at ~43%; two-year contracts at only ~3% |
| **Monthly Charges** | Higher charges strongly correlate with churn |
| **Tenure** | Churned customers have median tenure ~10 months vs ~38 months for retained |
| **Payment Method** | Electronic check users show the highest churn rate |
| **Internet Service** | Fiber optic users churn more despite premium pricing |

---

## ⚙️ Technical Pipeline

### 1. Feature Engineering
- Dropped irrelevant columns (`customerID`, `TotalCharges` due to multicollinearity)
- Mapped binary categorical columns (Yes/No → 1/0)
- One-hot encoded `InternetService`, `PaymentMethod`, `Contract`
- Converted `TotalCharges` from object to numeric; imputed zero-tenure customers with 0

### 2. Handling Class Imbalance
- Applied **SMOTETomek** (combination of oversampling + undersampling) for balanced training

### 3. Feature Scaling
- **StandardScaler** fitted only on training data to prevent data leakage
- Same scaler transformation applied to test set

### 4. Models Trained & Evaluated

| Model | Metric Focus |
|-------|-------------|
| Logistic Regression | Recall + ROC-AUC ✅ **Selected** |
| Random Forest | Precision-Recall balance |
| AdaBoost Classifier | Ensemble boosting |
| XGBoost Classifier | Gradient boosting |
| K-Nearest Neighbors | Distance-based |

> **Final Model: Logistic Regression** — selected for best balance of Recall and ROC-AUC score, critical for a churn use case where false negatives (missing a churner) are more costly than false positives.

### 5. Model Explainability — SHAP
- Applied **SHAP (SHapley Additive exPlanations)** with `LinearExplainer`
- Generated SHAP summary plot to identify feature importance

**Top SHAP Features:**
1. 🔴 **Tenure** — Low tenure = highest churn risk (most important feature)
2. 🔴 **Contract_Two year** — Strong negative churn predictor
3. 🔴 **MonthlyCharges** — High charges → churn risk
4. 🔴 **InternetService_Fiber optic** — Positive churn driver
5. 🔴 **PaymentMethod_Electronic check** — High-risk payment pattern

---

## 💡 Retention Strategies (SHAP-Driven)

| Strategy | SHAP Insight | Action |
|----------|-------------|--------|
| **Contract Migration** | Month-to-month = highest churn | Offer discounts for 1–2 year contracts |
| **Early Tenure Protection** | New customers churn most | Proactive onboarding + welcome offers |
| **Service Bundling** | No OnlineSecurity/Backup → churn | Free trials for add-on services |
| **High-Bill Alerts** | High MonthlyCharges → churn | Personalized plan optimization |
| **Payment Method Optimization** | Electronic check = high risk | Incentivize auto-pay methods |

---

## 🚀 Streamlit Web Application

The deployed app allows business users to input customer details and receive an instant churn risk assessment.

### App Features
- **Sidebar inputs** for 10 customer attributes (tenure, monthly charges, contract type, etc.)
- **Color-coded prediction card** — 🔴 High Risk / 🟠 Moderate Risk / 🟢 Low Risk
- **Probability score** with a visual progress bar
- **Expandable feature panel** showing all model inputs
- **Risk thresholds:** >70% = High Risk | 40–70% = Moderate | <40% = Low Risk

### Run the App Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app1.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
shap
streamlit
joblib
matplotlib
seaborn
```

---

## 📈 Results Summary

- ✅ **Final Model:** Logistic Regression
- ✅ **Technique:** SMOTETomek for class imbalance
- ✅ **Explainability:** SHAP LinearExplainer with summary plot
- ✅ **Deployment:** Streamlit web app with styled UI and risk-tiered output
- ✅ **Business Value:** 5 data-driven retention strategies derived from SHAP insights

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Libraries | Scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Model Saving | Joblib |

---

## 👩‍💻 Author

**Nithu Anna Ninan**
- 🔗 LinkedIn: [linkedin.com/in/nithu-ninan](https://linkedin.com/in/nithu-ninan)
- 📧 Email: nithuanna24@gmail.com

---

## ⭐ If you found this project useful, please give it a star!
