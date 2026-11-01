# Data

| File | What it is |
|---|---|
| `raw/ai4i2020_original.csv` | The **original, untouched** AI4I 2020 Predictive Maintenance dataset (10,000 rows). It is clean: no missing values, no duplicates (it has 27 label inconsistencies). |
| `raw/ai4i2020_impure.csv` | **Working raw data.** A copy of the original into which impurities were **deliberately injected** (step 0, fixed seed) so that data wrangling could be performed and demonstrated. Not a property of the Kaggle data. |
| `processed/ai4i2020_clean.csv` | The impure file after wrangling (step 2). |
| `processed/ai4i2020_kbs_enriched.csv` | Clean data + *simulated* operator observations + train/test split (step 4). |

**Source / citation:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020 Third International Conference on
Artificial Intelligence for Industries (AI4I), pp. 69-74. Dataset: UCI Machine Learning Repository (id 601) and Kaggle
(`stephanmatzka/predictive-maintenance-dataset-ai4i-2020`). Please check the license terms on the dataset page before redistributing the data.
