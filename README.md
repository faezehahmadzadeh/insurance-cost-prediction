# Insurance Cost Prediction

Predicting annual medical insurance charges from a person's age, BMI, number of children, smoking status, sex, and region — using Python and machine learning. This project connects directly to insurance claims/policy work: the same kind of data an insurer uses for pricing and risk assessment.

## Problem

Insurance companies need to estimate how much medical cost a person is likely to generate. If a model can predict charges well from a few simple features, it can support pricing, risk segmentation, and fraud/outlier review.

**Task:** regression — predict the continuous target `charges` (annual medical charges in USD).

## Dataset

- **Name:** Medical Cost Personal Dataset
- **Source:** https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv
- **Size:** 1,338 people, 7 columns
- **Features:**
  - `age` (years), `bmi` (body mass index), `children` (number covered) — numeric
  - `sex`, `smoker` (yes/no), `region` (northeast / northwest / southeast / southwest) — categorical
- **Target:** `charges`

The script downloads the data on first run and saves a local copy to `data/insurance.csv`. There are no missing values in this dataset.

## Method

1. **EDA** — summary statistics, missing-value check, average charges by smoker status and by region.
2. **Preprocessing** (scikit-learn `ColumnTransformer` + `Pipeline`, so there is no data leakage):
   - Numeric features: median imputation + `StandardScaler`
   - Categorical features: most-frequent imputation + `OneHotEncoder`
3. **Train/test split** — 80/20 (1,070 train / 268 test), `random_state=42`.
4. **Models compared** (see the comparison table below):
   - Linear Regression (baseline)
   - Decision Tree
   - Random Forest (300 trees)
   - Gradient Boosting
   - K-Nearest Neighbors (k=5, distance-based — uses the scaled features from the pipeline)
5. **Evaluation** on the held-out test set: RMSE, MAE, R².
6. **Outputs** — `outputs/metrics.json`, charges distribution chart, actual-vs-predicted chart for the best model, a feature-importance chart, and a model-comparison RMSE chart.

## Machine Learning Algorithms Compared

Five regression algorithms were trained on the same pipeline and evaluated on the same held-out test set (268 people):

| Algorithm | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression | 5,796.28 | 4,181.19 | 0.7836 |
| Decision Tree | 6,668.08 | 3,227.66 | 0.7136 |
| Random Forest | 4,611.37 | 2,525.80 | 0.8630 |
| **Gradient Boosting** | **4,324.57** | **2,400.76** | **0.8795** |
| K-Nearest Neighbors | 5,998.69 | 3,631.59 | 0.7682 |

**How to read this:** Linear Regression assumes a straight-line relationship, so it misses the sharp smoker/non-smoker split. A single Decision Tree captures non-linear patterns but is unstable on its own (highest RMSE here). Random Forest (averaging 300 trees) and Gradient Boosting (building trees one at a time, each correcting the previous trees' errors) are the strongest tree ensembles, and Gradient Boosting wins. K-Nearest Neighbors finds the 5 most similar people (by scaled age/BMI/children and encoded features) and averages their charges; it is simple but sensitive to which features dominate the distance, so it lands close to the linear baseline here.

## Results (actual run)

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression | 5,796.28 | 4,181.19 | 0.7836 |
| Decision Tree | 6,668.08 | 3,227.66 | 0.7136 |
| Random Forest | 4,611.37 | 2,525.80 | 0.8630 |
| **Gradient Boosting** | **4,324.57** | **2,400.76** | **0.8795** |
| K-Nearest Neighbors | 5,998.69 | 3,631.59 | 0.7682 |

Gradient Boosting is the best model: it explains about **88% of the variance** in charges, with a typical error (MAE) of about **$2,401**.

**What drives the cost?** (Gradient Boosting feature importance, grouped)

| Feature | Importance |
|---|---|
| smoker | 0.676 |
| bmi | 0.191 |
| age | 0.117 |
| children | 0.010 |
| region | 0.005 |
| sex | 0.001 |

## Business interpretation

- **Smoking dominates.** In the raw data, smokers average **$32,050/year** vs **$8,434/year** for non-smokers (median $34,456 vs $7,345) — roughly 3.8× higher. The model confirms this: smoking status alone carries about two-thirds of the predictive power.
- **BMI and age** are the next strongest factors — higher BMI and older age both push expected cost up.
- **Region and sex** matter very little in this data. Average charges by region range only from about $12,347 (southwest) to $14,735 (southeast).
- For an insurer, this supports risk-based pricing and targeted prevention programs (for example, smoking-cessation incentives), because one behavioral factor explains most of the cost difference. It also shows why a simple linear model is not enough: the tree models capture the sharp smoker/non-smoker split much better (R² 0.78 → 0.88).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python predict_insurance_cost.py
```

Outputs appear in `outputs/`:
- `metrics.json` — all model scores
- `charges_distribution.png`
- `actual_vs_predicted_gradient_boosting.png`
- `feature_importance_gradient_boosting.png`
- `model_comparison_rmse.png`

## Project structure

```
insurance-cost-prediction/
├── predict_insurance_cost.py   # main script (load → EDA → model → evaluate → charts)
├── requirements.txt
├── data/
│   └── insurance.csv           # saved automatically on first run
├── outputs/
│   ├── metrics.json
│   ├── charges_distribution.png
│   ├── actual_vs_predicted_gradient_boosting.png
│   ├── feature_importance_gradient_boosting.png
│   └── model_comparison_rmse.png
├── README.md
└── WALKTHROUGH_FA.md           # step-by-step explanation in Persian
```

## Skills used

Python, pandas, scikit-learn (pipelines, OneHotEncoder, StandardScaler, Linear Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, RMSE/MAE/R²), matplotlib, regression modeling, model comparison, exploratory data analysis, feature importance interpretation.

## Ideas to extend

- Tune Gradient Boosting hyperparameters with `GridSearchCV`
- Add an interaction feature (for example, smoker × BMI)
- Try predicting `log(charges)`, since the charge distribution is right-skewed
- Compare with a model trained only on SQL-aggregated features, mirroring claims reporting work
