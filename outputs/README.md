# Evaluation Outputs

This folder contains model evaluation reports and charts generated during training.

## Key files

| File | Description |
|------|-------------|
| `regression_results.csv` | Regression model comparison (MAE, RMSE, R²) |
| `classification_results.csv` | Classification model comparison (Accuracy, F1) |
| `actual_vs_predicted.png` | Actual vs predicted AQI scatter plot |
| `residual_analysis.png` | Residual distribution plot |
| `prediction_error_distribution.png` | Prediction error histogram |
| `confusion_matrix.png` | Classification confusion matrix |
| `feature_importance.png` | Top feature importances |

These files are included in the repository so the **Model Performance** dashboard page works without retraining.

Large per-row prediction CSVs are excluded via `.gitignore`.

Regenerate all outputs by running:

```bash
python scripts/train_all.py
```
