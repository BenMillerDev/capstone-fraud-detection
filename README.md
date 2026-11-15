# Card Fraud Detection Under a Review Budget

DSCI 598 Capstone · Maryville University · Ben Miller

For my capstone I'm building a fraud detection model using real e-commerce transactions from Vesta, a payment company. Fraud teams usually can't review every transaction, so they focus on the ones that look the riskiest. I want to find out how much fraud a model can catch when analysts only review a small share of transactions, and have the model explain why it flagged each one.

**Quick summary**
- **Data:** 590,540 labeled card transactions. Only about 3.5% of them are fraud, and most of the 430+ columns are anonymized.
- **Approach:** logistic regression, decision trees, random forests and gradient boosted trees (LightGBM), tuned with cross-validation on time-ordered folds, since a fraud model will always be scoring future data.
- **Evaluation:** ROC AUC, plus how much fraud gets caught when only the highest-scoring transactions are reviewed.
- **Explanation:** SHAP values to show which features led to each flag. As a stretch goal, I'd like to add an LLM that turns those results into a plain-English explanation for an analyst.
- **Status:** Week 5's LightGBM, using numerical features plus the most predictive categorical levels, scored 0.9086 on Kaggle's hidden test set, up from 0.8936 in Week 4. Customer-level features are next.

## How to review this repo

Each week of the project has its own folder. Start with that week's README, which says what the week's goals were, which notebooks to read and in what order, and what the results were. The notebooks are saved with their outputs, so nothing needs to be rerun to see the results.

| Week | Focus | Start here |
|---|---|---|
| 3 | Exploring the data, technical challenges and baseline models | [`week-3/README.md`](week-3/README.md) |
| 4 | Tuning four model types on numerical features and the first Kaggle submission | [`week-4/README.md`](week-4/README.md) |
| 5 | Testing which categorical levels are predictive, then retuning all four model types with them | [`week-5/README.md`](week-5/README.md) |

The Results table below summarizes every week in one place.

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
3. **Tune models on numerical features only.** I drop redundant V columns, then tune logistic regression, a decision tree, a random forest and LightGBM with cross-validation on time-ordered folds, and submit the best one.
4. **Add categorical features.** I use chi-square tests to find which levels of each categorical column are most predictive, keep only those, and repeat the same four-model tuning.
5. **Engineer customer-level features** by grouping transactions by card and address to approximate a customer, and add frequency encoding for columns with a lot of unique values.
6. **Evaluate each change** against the earlier models using AUC and fraud caught at a review budget.
7. **Explain the flagged transactions** with SHAP values, then add the LLM explanation layer if time allows.

## Decisions

- **Category codes stored as numbers count as categorical.** Columns like `card1`–`card6`, `addr1`, `addr2` and `id_12`–`id_38` contain numbers, but the numbers are codes for a card, a billing region or a browser, not measurements. I'm holding them back from the numerical-only models and adding them in the categorical step.
- **Tuning uses the full training set.** Every setting is scored on all 590,540 training transactions instead of a sample. It's slower (the Week 4 tuning notebook took about 80 minutes on Kaggle), but for an 8-week project I'd rather be thorough. The only sample is the 100,000 rows used to measure V-column correlations, which is a feature selection step.
- **Categorical levels are chosen inside each fold.** The tests that pick which categorical levels to keep run on each fold's training rows only, so the validation rows never help choose the features they're scored on.

## Progress

| Week | Work | Notebook |
|---|---|---|
| 3 | Exploratory analysis and technical challenges | [`bm-fraud-detection-exploratory.ipynb`](week-3/bm-fraud-detection-exploratory.ipynb) |
| 3 | Logistic regression and LightGBM baselines with time-based validation | [`bm-fraud-baseline-models.ipynb`](week-3/bm-fraud-baseline-models.ipynb) |
| 4 | Logistic regression, decision tree, random forest and LightGBM tuned on numerical features with time-ordered cross-validation | [`bm-fraud-numeric-tuning.ipynb`](week-4/bm-fraud-numeric-tuning.ipynb) |
| 4 | Submission from the best numerical-feature model | [`bm-fraud-numeric-submission.ipynb`](week-4/bm-fraud-numeric-submission.ipynb) |
| 5 | Chi-square tests to find the predictive levels of each categorical column | [`bm-fraud-categorical-tests.ipynb`](week-5/bm-fraud-categorical-tests.ipynb) |
| 5 | The four model types retuned on numerical features plus the kept levels | [`bm-fraud-categorical-tuning.ipynb`](week-5/bm-fraud-categorical-tuning.ipynb) |
| 5 | Submission from the best model with categorical levels | [`bm-fraud-categorical-submission.ipynb`](week-5/bm-fraud-categorical-submission.ipynb) |

## Results

I'll add results here as each experiment runs. Week 3 used a single split on the most recent 20% of training data. The last fold in Weeks 4 and 5 uses that same split, so the "last-20% AUC" column compares directly across weeks.

| Week | Model | Features | Last-20% AUC | Fraud caught reviewing top 1% | Kaggle private / public |
|---|---|---|---|---|---|
| 3 | Logistic regression | 17 numerical columns | 0.7787 | not measured | not submitted |
| 3 | LightGBM | All 432 columns | 0.9227 | not measured | not submitted |
| 4 | Logistic regression, tuned | 272 numerical columns | 0.8216 | 15.9% | not submitted |
| 4 | Decision tree, tuned | 272 numerical columns | 0.8377 | 18.5% | not submitted |
| 4 | Random forest, tuned | 272 numerical columns | 0.8664 | 23.4% | not submitted |
| 4 | LightGBM, tuned | 272 numerical columns | 0.8960 | 24.4% | 0.8936 / 0.9115 |
| 5 | Logistic regression, tuned | 272 numerical + 301 categorical levels | 0.8371 | 17.6% | not submitted |
| 5 | Decision tree, tuned | 272 numerical + 301 categorical levels | 0.8354 | 19.3% | not submitted |
| 5 | Random forest, tuned | 272 numerical + 301 categorical levels | 0.8857 | 23.6% | not submitted |
| 5 | LightGBM, tuned | 272 numerical + 301 categorical levels | 0.9126 | 24.7% | **0.9086 / 0.9326** |

The fraud-caught column for Weeks 4 and 5 is the average across the four time-ordered folds. Since fraud is 3.4–4.0% of each fold, reviewing the top 1% of transactions can catch at most about 25–29% of the fraud. LightGBM is close to that ceiling in both weeks, and over nine of every ten transactions it puts in the top 1% are fraud.

Holding back the categorical-style columns in Week 4 cost about 0.027 AUC compared with Week 3. Adding the kept categorical levels in Week 5 won back about 60% of that and raised the private Kaggle score from 0.8936 to 0.9086. The "301 categorical levels" are the ones kept on the last fold's training rows; the submission model, trained on all of the training data, kept 318. The rest of the gap is most likely `card1`, which has 13,553 values, and only 25 of them are kept here.

## Running the notebooks

I run the notebooks on Kaggle because it attaches the dataset directly and gives about 30 GB of RAM, which helps since the merged data takes up several GB in memory.

1. On Kaggle, create a new notebook and choose **File → Import Notebook**.
2. Click **Add Input** and add the *IEEE-CIS Fraud Detection* competition. You'll need to accept its rules first.
3. Run all cells. The code reads the data from `/kaggle/input/competitions/ieee-fraud-detection`. For the tuning notebooks, use **Save & Run All** instead, since Week 4's runs for about 80 minutes and Week 5's for about two hours.

## Repository contents

```
capstone-fraud-detection/
├── README.md                               project overview and results across all weeks
├── week-3/
│   ├── README.md                           Week 3 goals, findings and results
│   ├── bm-fraud-detection-exploratory.ipynb
│   └── bm-fraud-baseline-models.ipynb
├── week-4/
│   ├── README.md                           Week 4 goals, decisions and results
│   ├── bm-fraud-numeric-tuning.ipynb
│   ├── bm-fraud-numeric-submission.ipynb
│   └── tuning-results/                     every setting each model tried, with its scores
│       ├── tuning_logistic_regression.csv
│       ├── tuning_decision_tree.csv
│       ├── tuning_random_forest.csv
│       └── tuning_lightgbm.csv
└── week-5/
    ├── README.md                           Week 5 goals, decisions and results
    ├── bm-fraud-categorical-tests.ipynb
    ├── bm-fraud-categorical-tuning.ipynb
    ├── bm-fraud-categorical-submission.ipynb
    └── tuning-results/                     every setting each model tried, with its scores
        ├── tuning_logistic_regression.csv
        ├── tuning_decision_tree.csv
        ├── tuning_random_forest.csv
        └── tuning_lightgbm.csv
```

- `week-3/bm-fraud-detection-exploratory.ipynb` loads and merges the data, then looks at class balance, train vs. test over time, transaction amounts, categories, identity coverage, missing values and V-column redundancy. It ends with a table of the technical challenges I found.
- `week-3/bm-fraud-baseline-models.ipynb` encodes the text columns, sets up the time-based validation split, trains both baselines, shows feature importance and writes an optional Kaggle submission file.
- `week-4/bm-fraud-numeric-tuning.ipynb` builds the numerical feature set, drops redundant V columns, and tunes all four model types on time-ordered folds, scoring each setting by AUC and fraud caught reviewing the top 1%. It ends by printing the best model and settings.
- `week-4/bm-fraud-numeric-submission.ipynb` trains that best model on all of the training data and writes the Kaggle submission file.
- `week-4/tuning-results/` holds one CSV per model type with every setting tried and its scores. These back up the Week 4 numbers in the Results table.
- `week-5/bm-fraud-categorical-tests.ipynb` runs chi-square tests on every level of the 49 categorical columns, keeps the ones that are significant and have a much higher or lower fraud rate, and checks how well they carry over to the test transactions.
- `week-5/bm-fraud-categorical-tuning.ipynb` repeats those tests inside each fold and retunes all four model types on the numerical features plus the kept levels.
- `week-5/bm-fraud-categorical-submission.ipynb` trains the best model on all of the training data and writes the Kaggle submission.
- `week-5/tuning-results/` backs up the Week 5 numbers in the Results table.

## AI assistance

I used Claude (Anthropic) to help draft the notebook code and this README. All of the results come from my own runs.

## Sources

- [Kaggle: IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [NVIDIA Developer Blog: Leveraging Machine Learning to Detect Fraud](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution) (dataset size, fraud rate and the first-place team's techniques)

