# Week 3: Exploring the Data and Building Baselines

## This week's goals

> Perform exploratory data analysis to explore the structure of the data. ... Start identifying the technical challenges associated with working with the dataset for your competition. ... Begin building and evaluating models. Start by building relatively simple baseline models.

## What to read, in order

1. **[`bm-fraud-detection-exploratory.ipynb`](bm-fraud-detection-exploratory.ipynb)** loads and merges the transaction and identity files, then looks at class balance, train vs. test over time, transaction amounts, fraud rate by category, identity coverage, missing values and redundant V columns. Part 2 at the end is a table of the technical challenges I found and how I plan to handle each one.
2. **[`bm-fraud-baseline-models.ipynb`](bm-fraud-baseline-models.ipynb)** trains two baseline models on a time-based split and compares them. The last cell summarizes the results.

Both notebooks are saved with their outputs, so they don't need to be rerun to see the results.

## What I found

- **Fraud is rare.** 20,663 of the 590,540 training transactions (3.50%) are fraud, so I'm scoring models with ROC AUC instead of accuracy.
- **Test comes after train.** The training data covers days 1–182 and the test data covers days 213–395, with a 31-day gap between them. Because of this I validate on the most recent training transactions instead of a random split.
- **Identity data is partial but useful.** Only 24.4% of training transactions have identity data, but those transactions are fraud 7.85% of the time, compared with 2.09% without it. I kept a `has_identity` flag as a feature.
- **The test identity file spells its columns differently** (`id-01` instead of `id_01`), so the loading code renames them before merging.
- **A lot of data is missing or redundant.** The id columns are 85% missing on average, and 634 pairs of V columns are more than 95% correlated.

## Baseline results

Both models were validated on the most recent 20% of training transactions.

| Model | Features | Validation AUC |
|---|---|---|
| Logistic regression | 17 numerical columns | 0.7787 |
| LightGBM | All 432 columns | 0.9227 (277 trees) |

The top LightGBM features by gain were V257, C1, card1, C14 and V258. Card, address and time-delta (D) columns showed up throughout the top 20, which is part of why I want to try grouping transactions by card and address later on. I didn't submit to Kaggle this week.

## Kaggle notebooks

- Exploratory analysis: [link]
- Baseline models: [link]
