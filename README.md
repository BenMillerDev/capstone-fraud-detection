# capstone-fraud-detection
# Card Fraud Detection Under a Review Budget

DSCI 598 Capstone · Maryville University · Ben Miller

For my capstone I'm building a fraud detection model using real e-commerce transactions from Vesta, a payment company. Fraud teams usually can't review every transaction, so they focus on the ones that look the riskiest. I want to find out how much fraud a model can catch when analysts only review a small share of transactions, and have the model explain why it flagged each one.

**Quick summary**
- **Data:** 590,540 labeled card transactions. Only about 3.5% of them are fraud, and most of the 430+ columns are anonymized.
- **Approach:** gradient boosted trees (LightGBM), validated on the most recent transactions, since a fraud model will always be scoring future data.
- **Evaluation:** ROC AUC, plus how much fraud gets caught when only the highest-scoring transactions are reviewed.
- **Explanation:** SHAP values to show which features led to each flag. As a stretch goal, I'd like to add an LLM that turns those results into a plain-English explanation for an analyst.
- **Status:** the data exploration and baseline models are done (Week 3), and feature engineering is next.

## The problem

Fraud is rare in this data, so a model that predicts "not fraud" for every transaction would be 96.5% accurate without catching any fraud at all. Accuracy doesn't tell us much here. What matters is whether the model ranks transactions well, with the fraud near the top of the list, because that's the part a review team would look at.

I'm measuring that in three ways:
- **ROC AUC**, which shows how often a fraud transaction scores higher than a legitimate one.
- **Fraud caught at a review budget**, meaning the share of all fraud found if analysts only review the top 1% (or 2%, or 5%) of transactions.
- **False alarms**, meaning how many legitimate customers get flagged along the way.

## Data

The data comes from the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data), which Vesta and the IEEE Computational Intelligence Society released in 2019.

| File | What it holds |
|---|---|
| `train_transaction.csv` / `test_transaction.csv` | Every transaction, with the amount, product, card, address, email domain, and hundreds of anonymized count (C), time-delta (D), match (M) and Vesta-engineered (V) columns |
| `train_identity.csv` / `test_identity.csv` | Device and network details, which only some transactions have |

The test transactions all happen after the training period, and their labels are hidden.

I haven't included the data in this repo because Kaggle's competition rules don't allow redistributing it. The notebooks read it directly from Kaggle.

## Approach

1. **Explore the data** and identify the technical challenges, like the size of the data, how rare fraud is, the time gap between train and test, partial identity data, missing values and redundant anonymized columns.
2. **Build baseline models.** I started with logistic regression as a simple floor and LightGBM on every column. Both are validated on the most recent 20% of training transactions.
3. **Engineer features.** I plan to drop redundant V columns, add frequency encoding for columns with a lot of unique values, and group transactions by card and address to approximate a customer.
4. **Evaluate each change** against the baseline using AUC and fraud caught at a review budget.
5. **Explain the flagged transactions** with SHAP values, then add the LLM explanation layer if time allows.

## Progress

| Week | Work | Notebook |
|---|---|---|
| 3 | Exploratory analysis and technical challenges | [`fraud-eda.ipynb`](fraud-eda.ipynb) |
| 3 | Logistic regression and LightGBM baselines with time-based validation | [`fraud-baseline-models.ipynb`](fraud-baseline-models.ipynb) |
| 4 | Feature engineering and review-budget metrics | *in progress* |

## Results

I'll add results here as each experiment runs.

| Model | Validation AUC | Fraud caught reviewing top 1% |
|---|---|---|
| Logistic regression (17 columns) | | |
| LightGBM (all columns) | | |

## Running the notebooks

I run the notebooks on Kaggle because it attaches the dataset directly and gives about 30 GB of RAM, which helps since the merged data takes up several GB in memory.

1. On Kaggle, create a new notebook and choose **File → Import Notebook**.
2. Click **Add Input** and add the *IEEE-CIS Fraud Detection* competition. You'll need to accept its rules first.
3. Run all cells. The code reads the data from `/kaggle/input/ieee-fraud-detection`.

## Repository contents

- `fraud-eda.ipynb` loads and merges the data, then looks at class balance, train vs. test over time, transaction amounts, categories, identity coverage, missing values and V-column redundancy. It ends with a table of the technical challenges I found.
- `fraud-baseline-models.ipynb` encodes the text columns, sets up the time-based validation split, trains both baselines, shows feature importance and writes an optional Kaggle submission file.

## AI assistance

I used Claude (Anthropic) to help draft the notebook code and this README. All of the results come from my own runs.

## Sources

- [Kaggle: IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [NVIDIA Developer Blog: Leveraging Machine Learning to Detect Fraud](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution) (dataset size, fraud rate and the first-place team's techniques)

