# Indian Urban House Price Prediction

This project analyzes the Indian Urban House Price Prediction dataset. Open `ML_Model/src/main.ipynb` and run all cells to view the data summary, regression metrics, price predictions, correlation matrix, and diagnostic visualizations inline.

## Dataset

The archive supplies 80,000 labeled training properties, 20,000 unlabeled test properties, a sample submission, and column metadata. The continuous target is `Price_INR_Lakhs` (price in Indian rupee lakhs). `Property_ID` is used for row identification and excluded from model features. The notebook does not add synthetic records or use test targets.

| File | Rows | Purpose |
| --- | ---: | --- |
| `ML_Model/data_set/train.csv` | 80,000 | Labeled training records |
| `ML_Model/data_set/test.csv` | 20,000 | Unlabeled prediction records |
| `ML_Model/data_set/metaData.csv` | 18 | Feature and target descriptions |
| `ML_Model/data_set/sample_submission.csv` | 20,000 | Expected prediction ID/target format |

The training file has no missing values, duplicate records, or duplicate property IDs. Train and test contain the same 16 predictors, and their property IDs do not overlap.

## Workflow

The notebook loads the archive data, checks the schema, explores target and city distributions, computes numeric correlations, and compares four regression pipelines. Numeric features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded. Models predict `log1p(Price_INR_Lakhs)` and transform predictions back to lakhs.

An 80/20 stratified split of the labeled training data reserves 16,000 properties for validation. Three-fold cross-validation uses a fixed 9,000-row sample from the training partition only. KNN uses at most 20,000 fitting rows and SVR at most 6,000 for practical runtime; Linear Regression and Gradient Boosting use the full training partition. Model selection uses the lowest mean CV RMSE. The test set is used only to display predictions; it has no target labels, so no test metrics are reported.

## Measured Results

Metrics below are from actual predictions on the 16,000-row validation split. Errors are in INR lakhs (MSE is lakhs squared); lower error is better.

| Model | R² | MAE | MSE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gradient Boosting Regressor | 0.7695 | 33.3775 | 5,082.0551 | 71.2885 | 24.4253 |
| Linear Regression | 0.7484 | 35.4261 | 5,546.8293 | 74.4770 | 26.1307 |
| Support Vector Regression | 0.7013 | 40.8741 | 6,584.4815 | 81.1448 | 29.9635 |
| KNN Regressor | 0.5033 | 53.1531 | 10,949.8005 | 104.6413 | 41.4011 |

Gradient Boosting was selected with mean three-fold CV RMSE **70.0752 lakhs** (SD 2.1217) and mean CV R² **0.7637**. These scores are specific to the supplied dataset and fixed split/seed; they are not guarantees of real-world market accuracy.

## Visualizations

Running the notebook displays seven figures: target distribution and area-price relationship; numeric feature-correlation heatmap; city-wise price distribution; actual versus predicted validation prices; residuals versus predictions; validation RMSE comparison; residual distribution; and Gradient Boosting feature importance. The correlation and feature-importance plots are associations, not causal explanations.

## Run

Install the root requirements, open the notebook in VS Code/Jupyter, and run all cells:

```powershell
python -m pip install -r requirements.txt
```

The notebook displays all tables and figures inline. It does not save or overwrite fitted models, submissions, or reports. Test predictions are shown in the final table and can be exported separately if needed.

## Structure

```text
House-Price-Prediction-master/
├── README.md
├── requirements.txt
└── ML_Model/
    ├── data_set/
    │   ├── metaData.csv
    │   ├── sample_submission.csv
    │   ├── test.csv
    │   └── train.csv
    └── src/
        └── main.ipynb
```

