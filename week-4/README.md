# Week 4: Tuning Models on Numerical Features

## This week's goals

> Focus on improving your models and earning a higher score than your baseline models from the previous week. Expand your approach to include more advanced forms of preprocessing or modeling.

> To postpone the challenges of categorical features, start by building models that use only the numerical features. Consider logistic regression, decision trees, random forests, and gradient-boosted tree models. Use cross-validation to perform hyperparameter tuning. After identifying the optimal model type and parameters, create a new notebook on Kaggle to submit predictions from this model.

## What to read, in order

1. **[`bm-fraud-numeric-tuning.ipynb`](bm-fraud-numeric-tuning.ipynb)** builds the numerical feature set, drops redundant V columns, and tunes all four model types with cross-validation on time-ordered folds. It ends with a comparison of the best setting for each model type and a summary of what tuning showed.
2. **[`bm-fraud-numeric-submission.ipynb`](bm-fraud-numeric-submission.ipynb)** trains the best model from tuning on all of the training data and writes the Kaggle submission. The last cell has the Kaggle score.
3. **[`tuning-results/`](tuning-results/)** has every setting each model tried, with its scores, so any number in this README can be checked without rerunning the tuning.

## Tuning results files

Each file has one row per setting tried, sorted best first.

| File | Settings tried |
|---|---|
| [`tuning_logistic_regression.csv`](tuning-results/tuning_logistic_regression.csv) | 8 |
| [`tuning_decision_tree.csv`](tuning-results/tuning_decision_tree.csv) | 12 |
| [`tuning_random_forest.csv`](tuning-results/tuning_random_forest.csv) | 6 |
| [`tuning_lightgbm.csv`](tuning-results/tuning_lightgbm.csv) | 10 |

The columns are:
- `params`: the settings tried, as JSON;
- `mean_auc` and `std_auc`: the average AUC across the four folds and how much it varied between them;
- `last_fold_auc`: the AUC on the most recent 20% of training data, the same split as Week 3;
- `fraud_caught_top_1pct`: the share of all fraud caught when reviewing the top 1% of transactions;
- `review_precision_top_1pct`: the share of those reviewed transactions that are fraud;
- `minutes`: how long the setting took to score;
- `mean_trees`: LightGBM only, the average number of trees early stopping chose.

Both notebooks are saved with their outputs, so they don't need to be rerun to see the results.

## Decisions

- **Category codes stored as numbers are held back.** Columns like `card1`–`card6`, `addr1`, `addr2` and `id_12`–`id_38` contain numbers, but they're codes for a card, a billing region or a browser, not measurements. Following the recommendation to postpone categorical features, I left them out until next week's categorical step.
- **Tuning uses the full training set.** Every setting was scored on all 590,540 training transactions, not a sample. The whole tuning notebook took about 80 minutes on Kaggle.

## How the models were tuned

- **Features:** 272 numerical columns: amount, the identity flag, the C, D and dist columns, `id_01`–`id_11`, and 228 V columns, after dropping 111 V columns that were more than 95% correlated with one already kept.
- **Cross-validation:** four time-ordered folds, where each fold trains on earlier transactions and validates on the next 118,108. The last fold uses the same split as Week 3, so its AUC compares directly.
- **Search:** random search over each model's settings: 8 for logistic regression, 12 for the decision tree, 6 for the random forest and 10 for LightGBM.
- **Metrics:** ROC AUC, plus how much fraud is caught when analysts only review the top 1% of transactions.

## Results

| Model | Mean AUC (4 folds) | Last-fold AUC | Fraud caught reviewing top 1% |
|---|---|---|---|
| LightGBM | 0.8926 | 0.8960 | 24.4% |
| Random forest | 0.8691 | 0.8664 | 23.4% |
| Decision tree | 0.8351 | 0.8377 | 18.5% |
| Logistic regression | 0.8317 | 0.8216 | 15.9% |

LightGBM was the best model. It trained on all of the training data with 327 trees and scored **0.8936 private / 0.9115 public** on Kaggle. The private score is only 0.0024 below its last-fold AUC, which tells me the time-ordered validation is a good estimate of how the model does on future data.

Since fraud is 3.4–4.0% of each fold, reviewing the top 1% of transactions can catch at most about 25–29% of the fraud. LightGBM catches 24.4%, and about nine of every ten transactions it puts in the top 1% are fraud.

Compared with Week 3, logistic regression improved from 0.7787 to 0.8216 on the same split because it now has far more columns. LightGBM's last-fold AUC is 0.027 lower than Week 3's 0.9227. That's most likely the cost of holding back the categorical-style columns, several of which were among Week 3's top features, so next week's categorical step should help close that gap.

## Kaggle notebooks

- Tuning: [link]
- Submission: [link]
