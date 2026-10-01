# 🏦 Bank Customer Churn Prediction

Predicting which bank customers are about to leave, using feature engineering, a leak-free preprocessing pipeline and four model families
(Logistic Regression, Random Forest, **XGBoost**, **LightGBM**), tuned with **Optuna** and explained with **SHAP**.

![SHAP summary](images/shap_summary.png)

## Results

**Final model: LightGBM** - evaluated once on a held-out test set of 2,500 customers.

| Metric | Test set |
|--------|----------|
| ROC-AUC | **0.870** |
| Recall | 65.0% |
| Precision | 63.5% |
| F1 | 64.3% |
| Accuracy | 85.3% |

The model finds roughly two thirds of the customers who actually leave (331 of 509) with ~64% precision.
Accuracy alone is misleading here because only 20.4% of customers churn, so recall, precision and ROC-AUC are the main metrics.

![Confusion matrix](images/confusion_matrix_test.png)

### Model comparison

| Model | Evaluated on | ROC-AUC | Recall | Precision | F1 |
|-------|--------------|---------|--------|-----------|-----|
| Logistic Regression | 3-fold CV (train) | 0.808 | 72.1% | 42.2% | 53.3% |
| Random Forest | 3-fold CV (train) | 0.847 | 48.4% | 71.0% | 57.6% |
| XGBoost (tuned) | validation split | 0.869 | 69.9% | 62.2% | 65.8% |
| LightGBM (tuned) | validation split | 0.868 | 66.7% | 65.0% | 65.8% |

XGBoost and LightGBM perform almost identically; LightGBM was kept as the final model.

## Pipeline
1. **EDA** - class balance, distributions, churn rate per segment, correlations
2. **Feature engineering** - age groups, "has balance" flag, tenure-to-age ratio, product-engagement level
3. **Preprocessing pipeline** (`ColumnTransformer`) - imputation, log transform + scaling for skewed numerics, one-hot encoding, fitted on the training split only
4. **Baselines** - Logistic Regression and Random Forest with class weights
5. **Boosting models** - XGBoost and LightGBM with native categorical support, early stopping and `scale_pos_weight` for the 4:1 class imbalance
6. **Optuna tuning** - 100 trials per model, first on validation log-loss only, then with an **overfitting penalty** on the train/validation gap
7. **Explainability** - SHAP summary, feature importance and a single-customer waterfall plot

### Fighting overfitting
The first Optuna study minimised validation loss only and the models overfit badly (train accuracy ≈ 99% for XGBoost, ≈ 98% for LightGBM vs ≈ 86% on validation).
Adding a penalty on the train/validation log-loss gap brought train accuracy down to ≈ 88% / 90% while keeping validation performance.

## What drives churn?

![SHAP feature importance](images/shap_feature_importance.png)

The most influential features are the **number of products**, **age**, **gender**, **activity status** and the engineered **product-engagement** level.

| Churn rate by age group | Churn rate by country |
|---|---|
| ![Age group](images/churn_by_age_group.png) | ![Geography](images/churn_by_geography.png) |

![Waterfall](images/shap_waterfall_example.png)

## Limitations
- Customers with 3-4 products churn at extremely high rates (and every customer with 4 products in this data churned). This looks like a quirk of the dataset rather than natural behaviour and should be checked with the business before relying on it.
- The decision threshold is the default 0.5; tuning it on a validation set would let the bank trade precision for recall.
- Optuna and early stopping use the validation split, so the test-set numbers above are the ones to trust.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Bank_Churn_Prediction.ipynb
```
The notebook expects `Churn_Modelling.csv` (10,000 customers, 14 columns) in the same folder.

## Tech stack
Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · Optuna · SHAP · Matplotlib · Seaborn
