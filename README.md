# Customer Churn Prediction

Predicting customer churn for a telecommunications company using machine learning — XGBoost baseline, an improved XGBoost model with advanced feature engineering, and a PyTorch neural network.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Business Recommendations](#business-recommendations)

---

## Project Overview

Customer churn prediction is critical for subscription-based businesses. Identifying at-risk customers early enables proactive retention efforts and reduces revenue loss.

This project uses the **Telco Customer Churn dataset** to:

- Build a baseline XGBoost classifier
- Improve it with advanced feature engineering, SMOTE class-balancing, and threshold optimisation
- Benchmark against a PyTorch neural network
- Surface actionable insights for business stakeholders

---

## Repository Structure

```
customer-churn-prediction/
├── data/
│   └── telco-customer-churn.csv        # Raw dataset
├── src/
│   ├── churn_prediction.py             # Baseline XGBoost model
│   ├── improved_churn_model.py         # Improved XGBoost model
│   ├── compare_models.py               # Side-by-side model comparison
│   ├── analyze_results.py              # Feature importance & error analysis
│   └── predict_new_customers.py        # Inference script for new data
├── models/
│   ├── xgboost_churn_model.json        # Saved baseline model
│   ├── improved_xgboost_churn_model.json
│   ├── optimal_threshold.txt           # Optimal decision threshold
│   ├── best_pytorch_model.pth          # Saved PyTorch model weights
│   ├── pytorch_metrics.txt
│   └── pytorch_optimal_threshold.txt
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── improved_confusion_matrix.png
│   ├── improved_feature_importance.png
│   ├── improved_roc_curve.png
│   ├── pytorch_confusion_matrix.png
│   ├── pytorch_learning_curve.png
│   └── pytorch_roc_curve.png
├── docs/
│   └── Churn_Prediction_DS_ML_Technical_Notes.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The **Telco Customer Churn** dataset (IBM/Kaggle) contains 7,043 customers with 21 features:

| Category | Features |
|---|---|
| Demographics | Gender, senior citizen status, partner, dependents |
| Account info | Tenure, contract type, payment method, monthly/total charges |
| Services | Phone, internet, streaming TV/movies, online security/backup |
| Target | `Churn` (Yes / No) |

The dataset is class-imbalanced (~26.5% churn rate).

---

## Methodology

### Baseline XGBoost Model
- Label encoding of categorical features
- XGBoost with basic hyperparameter tuning
- Evaluated on accuracy and F1 score

### Improved XGBoost Model
- **Feature engineering:** tenure groups, charge buckets, service-count metrics, and interaction terms (tenure × monthly charges)
- **Class imbalance:** SMOTE oversampling applied to training data
- **Scaling:** StandardScaler on numerical features
- **Threshold optimisation:** decision boundary tuned to maximise recall for churners

### PyTorch Neural Network
- Fully connected network trained with binary cross-entropy loss
- Learning curve monitored for overfitting
- Comparable evaluation metrics and ROC-AUC

---

## Results

| Metric | Baseline XGBoost | Improved XGBoost |
|---|---|---|
| Accuracy | 81.62% | 79.79% |
| F1 (Churn class) | 0.61 | 0.66 |
| Recall (Churn class) | 0.55 | 0.77 |
| AUC-ROC | — | 0.858 |
| Class imbalance handling | None | SMOTE |

**Key insight:** Despite a slight drop in overall accuracy, the improved model catches **77% of churners vs 55%** — a 22-percentage-point recall improvement that translates directly to retention opportunities.

> For a company with 100,000 customers and 10% annual churn, this improvement means identifying ~2,200 additional at-risk customers per year.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mina-Rezaei/customer-churn-prediction.git
cd customer-churn-prediction

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Train the baseline model
python src/churn_prediction.py

# Train the improved model
python src/improved_churn_model.py

# Compare both models
python src/compare_models.py

# Analyse feature importance
python src/analyze_results.py

# Run inference on new customers
python src/predict_new_customers.py
```

### Sample Prediction Output

```
===== CUSTOMER RISK ASSESSMENT =====
tenure  Contract         MonthlyCharges  Churn_Probability  Predicted_Churn
6       Month-to-month   70.35           0.44               No
24      One year         89.10           0.19               No
12      Month-to-month   30.20           0.12               No

Customer Risk Levels:
  Low Risk:    3 customers
    Medium Risk: 1 customers
      High Risk:   0 customers
      ```

      ---

      ## Business Recommendations

      **Highest-risk segments to target:**

      1. **New customers (< 12 months tenure)** — highest churn probability; onboarding improvements and loyalty incentives are high ROI
      2. **Month-to-month contract holders** — offer discounts to upgrade to annual plans
      3. **High monthly charge customers** — review pricing and bundle offerings
      4. **Electronic check payment users** — encourage auto-pay setup

      **Retention levers:** early-tenure check-ins, contract upgrade incentives, proactive support for fiber optic users.

      ---

      ## Tech Stack

      ![Python](https://img.shields.io/badge/Python-3.8+-blue)
      ![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-orange)
      ![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)
      ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-green)

      ---

      ## License

      This project is open-source and available under the [MIT License](LICENSE).
