# House Price Prediction

## Overview
This project predicts residential sale prices from property characteristics using the Kaggle House Prices - Advanced Regression Techniques dataset. The notebook compares four regression models with a reproducible, leakage-safe seven-fold evaluation and creates test-set predictions.

## Problem Statement
Estimate `SalePrice` from features describing each home, such as quality, living area, neighborhood, basement, and garage attributes.

## Dataset
The data is the Kaggle House Prices competition dataset for residential homes in Ames, Iowa.
- `train.csv` contains property features and the `SalePrice` target.
- `test.csv` contains corresponding property features without `SalePrice`.
- `SalePrice` is the target variable.
- Supplied features describe property size, quality, location, construction, basement, and garage characteristics.

## Machine Learning Workflow
Data loading -> Data cleaning -> EDA -> Missing value handling -> Feature engineering -> Encoding -> Scaling where required -> Model training -> Model comparison -> Model selection -> Test prediction

Preprocessing is fitted within each cross-validation training fold. Categorical features are imputed and one-hot encoded. Numeric features are median-imputed; scaling is applied to numeric columns for Linear Regression, KNN, and SVR, but not Gradient Boosting. The Kaggle `Id` is retained for output and excluded from model inputs.

## Models Evaluated
### 1. Linear Regression
A baseline regression model that estimates a linear relationship between features and price.

### 2. K-Nearest Neighbors Regression
Predicts price using the target values of nearby training observations.

### 3. Support Vector Regression
Fits a regression function using support-vector-based optimization.

### 4. Gradient Boosting Regressor
Builds an ensemble sequentially, with each tree improving on the preceding model.

## Model Comparison
Because this is a regression task, model fit is measured with R2 rather than classification accuracy. The table reports pooled out-of-fold R2, MAE, MSE, and RMSE for all four models from the same shuffled seven-fold split. CV R2 is the mean validation-fold R2; the selection rule ranks by CV R2, then uses pooled out-of-fold RMSE as a tie-breaker.

| Model | R2 Score | MAE | MSE | RMSE | CV R2 |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 0.6408 | 19,600.82 | 2,265,524,065.88 | 47,597.52 | 0.6003 |
| KNN Regressor | 0.7546 | 21,767.76 | 1,547,839,108.21 | 39,342.59 | 0.7503 |
| SVR | -0.0493 | 55,504.65 | 6,617,932,859.16 | 81,350.68 | -0.0513 |
| Gradient Boosting | 0.8776 | 16,281.88 | 771,996,810.61 | 27,784.83 | 0.8674 |

| Model | CV R2 |
|---|---:|
| Linear Regression | 0.6003 |
| KNN Regressor | 0.7503 |
| SVR | -0.0513 |
| Gradient Boosting | 0.8674 |

## Final Model
Multiple regression models were evaluated, and Gradient Boosting Regressor was selected as the final model based on comparative performance. It had the highest mean seven-fold CV R2 (0.8674) and the lowest pooled out-of-fold MAE and RMSE among the four evaluated models. The selected pipeline is retrained on all training rows before predicting the test data.

## Project Structure
```text
House-Price-Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── model_evaluation_report.txt
└── ML_Model/
    ├── data_set/
    │   ├── train.csv
    │   ├── test.csv
    │   ├── sample_submission.csv
    │   └── data_description.txt
    └── src/
        ├── main.ipynb
        └── submission.csv
```

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Joblib

## How to Run
1. Clone or download the repository.
2. Create a virtual environment with `python -m venv .venv`.
3. Activate it (`.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on macOS/Linux).
4. Install dependencies with `python -m pip install -r requirements.txt`.
5. Open `ML_Model/src/main.ipynb` in Jupyter or VS Code and run all cells.
6. Check `ML_Model/src/submission.csv` and `model_evaluation_report.txt`.

The notebook resolves bundled data and output paths relative to its own location; no local machine paths are required.

## Output
`ML_Model/src/submission.csv` contains 1,459 rows with `Id` and numeric `SalePrice` columns in the same ID order as `test.csv`.

## Future Improvements
- Tune hyperparameters with a nested or repeated cross-validation strategy.
- Compare ensemble methods and target transformations.
- Explore domain-informed feature engineering.
- Deploy an interactive prediction interface.
- Monitor model quality as new sale data becomes available.

