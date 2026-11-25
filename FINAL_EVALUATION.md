# Final Model Evaluation Report

## Executive summary

- **Task:** binary customer churn classification
- **Dataset:** IBM Telco Customer Churn sample dataset
- **Rows:** 7,043
- **Selected estimator:** Logistic Regression
- **Approved operating threshold:** 0.63371099
- **Minimum validation Recall requirement:** 0.65
- **Reported test Recall:** 0.6818

The approved threshold was selected using validation predictions only, by maximizing Precision subject to Recall being at least 0.65. The test metrics below were then calculated at that fixed threshold.

## Evaluation protocol

1. Load and validate the IBM Telco Customer Churn dataset.
2. Create a stratified 60% train / 20% validation / 20% test split with `random_state=42`.
3. Compare Logistic Regression, Random Forest, and Gradient Boosting using validation F1 at threshold 0.50.
4. Select Logistic Regression (validation F1: 0.6243; Random Forest: 0.6156; Gradient Boosting: 0.5693).
5. Search thresholds using validation predictions only, requiring Recall >= 0.65 and maximizing Precision.
6. Refit the selected estimator on the combined train and validation partitions.
7. Calculate the reported test metrics at the fixed threshold 0.63371099.

### Validation threshold result

| Metric | Validation result |
|---|---:|
| Threshold | 0.63371099 |
| Precision | 0.5920 |
| Recall | 0.6711 |
| F1 | 0.6291 |
| False Positives | 173 |
| False Negatives | 123 |
| True Positives | 251 |
| True Negatives | 862 |

This threshold met the required minimum Recall of 0.65 while giving the highest Precision among the eligible thresholds in the fine-grained validation search.

## Final test metrics

The test partition contains 1,409 rows: 1,035 No Churn cases and 374 Churn cases.

| Metric | Test result |
|---|---:|
| Accuracy | 0.7729 |
| Precision | 0.5592 |
| Recall | 0.6818 |
| F1 | 0.6145 |
| ROC-AUC | 0.8413 |

### Confusion matrix

Rows are actual classes and columns are predicted classes.

|  | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 834 (TN) | 201 (FP) |
| Actual: Churn | 119 (FN) | 255 (TP) |

### Class-level report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No Churn | 0.88 | 0.81 | 0.84 | 1,035 |
| Churn | 0.56 | 0.68 | 0.61 | 374 |

## Business interpretation

The business requirement prioritized avoiding false-positive churn alerts while retaining at least 65% Recall on validation.

At the approved threshold:

- 255 of 374 actual churners were identified.
- 119 actual churners were missed (false negatives).
- 201 customers who did not churn received a false alert.
- 55.92% of positive churn alerts were correct.
- Test Recall was 68.18%, above the agreed 65% minimum.

The model is useful as an analytical baseline, but the false-positive and false-negative counts should be evaluated against actual retention-team capacity and intervention costs before operational use.

## Test-set transparency and caveat

The test partition was previously evaluated at an earlier validation-selected threshold of 0.6098, and was subsequently re-evaluated at 0.63371099. The threshold adjustment and selection were based on validation results and the pre-agreed business Recall constraint, not on test metrics. Nevertheless, the test partition was not inspected only once in the overall project history.

For that reason, these numbers should be presented as the project's reported holdout evaluation snapshot, not as a pristine, never-before-inspected estimate. Do not use this test partition for further model selection or threshold tuning. A future independent evaluation should use a newly collected or separately reserved dataset.

## Implemented threshold policy

The training implementation now enforces the agreed validation constraint `Recall >= 0.65`. It searches candidate thresholds using validation predictions only and selects the threshold with the highest Precision among eligible candidates. Ties are resolved deterministically by preferring higher Recall, then higher F1, then the higher threshold. If no candidate meets the minimum Recall, training stops with a clear error rather than silently violating the business requirement.

Model selection remains separate: Logistic Regression, Random Forest, and Gradient Boosting are compared using validation F1 at the default threshold of 0.50. After the model and threshold are selected, the final pipeline is refitted on train plus validation data. The chosen threshold, minimum Recall, and selection rule are persisted in `models/churn_threshold.json`; the prediction CLI reads that threshold for inference.

The test partition is excluded from model and threshold selection. Training reports test metrics after the threshold is locked, but those metrics must not be used to tune the threshold or choose a model. Because this project has already inspected the test partition more than once (as disclosed above), the reported test metrics remain a historical evaluation snapshot rather than a pristine one-time estimate.

## Reproducibility notes

- Split: stratified 60/20/20
- Random seed: 42
- Model selection: validation F1 at threshold 0.50
- Threshold selection: validation Precision maximization subject to Recall >= 0.65
- Final model fit: train + validation
- Reported test threshold: 0.63371099
