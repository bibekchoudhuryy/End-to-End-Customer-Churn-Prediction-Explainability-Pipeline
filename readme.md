
# End-to-End Customer Churn Prediction & Explainability Pipeline

## Overview
This project implements a **production-style customer churn prediction system** designed to identify customers at high risk of churn and provide **explainable, actionable insights** for retention strategies.

The pipeline covers the **entire ML lifecycle** — from raw data ingestion and preprocessing to model training, evaluation, and explainability — with a modular codebase that can be extended to real-time scoring or deployment.

The project emphasizes:
- Imbalanced classification handling
- Precision–recall–driven evaluation
- Model interpretability using SHAP
- Reproducible training and artifact management

---

## Business Problem
Customer churn is a **highly asymmetric cost problem**:
- **False negatives** (missed churners) lead to revenue loss
- **False positives** lead to unnecessary retention spend

Accuracy alone is misleading. This system focuses on **precision–recall tradeoffs** to align predictions with real-world business decisions.

---

## Dataset
- **Source:** Telco Customer Churn dataset  
- **Target:** `Churn` (binary classification)
- **Features:**  
  - Demographics (gender, partner, dependents)  
  - Contract and billing details  
  - Service subscriptions (internet, security, backup, streaming)  
  - Tenure and monetary variables  

Raw data is stored separately from processed train/test splits to prevent leakage.

---

## Project Structure

```

customer_churn_predictor/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── x_train.csv
│       ├── x_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── src/
│   ├── data_preprocessing.py    # Cleaning, encoding, feature preparation
│   ├── trainer.py               # Model training and selection
│   ├── evaluation.py            # Metrics, plots, threshold analysis
│   └── pipeline.py              # End-to-end pipeline orchestration
│
├── notebooks/
│   └── training_and_explainability.ipynb
│
├── models/
│   └── best_model.joblib        # Persisted trained model
│
├── plots/
│   ├── confusion_metrics.png
│   ├── precision_recall.png
│   └── feature_importances.png
│
├── reports/
│   ├── feature_importances.csv
│   └── shap_summary.png
│
├── logs/
│
└── README.md

````

---

## Modeling Approach

### Preprocessing
- Missing value handling
- Categorical encoding
- Train/test split performed **before modeling**
- Feature preparation isolated to avoid data leakage

### Model Training
- Supervised binary classification
- Class imbalance handled via:
  - Class weighting
  - Threshold tuning
- Final model persisted using `joblib`

### Evaluation Strategy
- Confusion matrix
- Precision–Recall curve (preferred over ROC)
- Error-type analysis (false positives vs false negatives)

Accuracy is intentionally **not** the primary metric.

---

## Explainability & Insights

### Feature Importance
- Permutation-based feature importance to assess global influence

### SHAP Explainability
- SHAP summary plots used to explain:
  - Directional impact of features
  - High-risk churn segments
- Key churn drivers identified:
  - Month-to-month contracts
  - Low tenure customers
  - Lack of online security / backup services
  - Electronic check payment method

These insights directly support **business actionability**, not just model transparency.

---

## Results Summary
- Strong precision–recall tradeoff suitable for retention targeting
- Clear identification of churn-driving features
- Fully reproducible pipeline with saved artifacts and reports

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline

```bash
python src/pipeline.py
```

Artifacts (model, plots, reports) will be generated automatically.

---

## Extensibility

This project is designed to be extended into:

* Real-time churn scoring (FastAPI)
* Cost-sensitive optimization using customer LTV
* Automated retraining workflows
* Deployment with Docker and CI/CD
* Monitoring and drift detection

---

## Key Takeaways

* Demonstrates real-world handling of **imbalanced classification**
* Prioritizes **business-aligned evaluation metrics**
* Integrates **model explainability** as a first-class concern
* Structured for production evolution, not notebooks-only analysis

---

## Author

**Bibek Choudhury**

This project was built as part of a broader effort to develop **production-grade ML systems** aligned with industry standards.

---

## License

MIT License



