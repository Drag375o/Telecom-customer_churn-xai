# Explainable Machine Learning for Business Customer Churn Prediction

**Comparing Financial and Service-Engagement Factors in Telecommunications Customer Churn**

## Overview

A Bulgarian telecommunications operator wants to identify which of its business
customers are at risk of leaving. This project builds an explainable churn
prediction pipeline over 8,453 real business-customer records and uses it to
test a specific question about *what kind* of customer information carries
predictive value.

**Headline result: churn is close to unpredictable from the available
attributes.** The best model achieves PR-AUC 0.0882 against a no-skill floor of
0.0651, and ROC-AUC 0.5697 against a random baseline of 0.500. This negative
result is reported as the primary finding rather than buried.

## Research Questions

> **RQ1.** Are financial characteristics or service-engagement characteristics
> more useful for predicting business customer churn?

> **RQ2.** Does combining both families improve prediction over the better
> family alone?

> **RQ3.** Which individual attributes contribute most to predictions?

## Positioning

This dataset has existing published research attached to it. Applying machine
learning to it is not a contribution. The contribution here is:

1. a **controlled comparison** of financial vs service-engagement feature
   groups, held constant across three models;
2. **statistically tested** rather than eyeballed, using repeated
   cross-validation with paired significance tests;
3. **honest reporting** of a weak-signal result, including a configuration that
   performed below random.

## Dataset

| Property | Value |
|---|---|
| Customers | 8,453 |
| Feature columns | 13 |
| Target | `CHURN` (Yes/No) |
| Churned | 549 |
| Churn rate | **6.49%** |
| Class imbalance | **14.4 : 1** |

Source: [Mendeley Data, DOI 10.17632/nrb55gr66h.1](https://data.mendeley.com/datasets/nrb55gr66h/1)
(Tokmakov, 2024, CC BY 4.0). See `data/README.md`.

## Methodology

**Structural integrity checks.** Before modelling, three arithmetic identities
were tested and confirmed:

| Identity | Match rate |
|---|---|
| `TotalRevenue = AvgMobileRevenue + AvgFIXRevenue` | 100.0% |
| `Total_SUBs = Active + NotActive + Suspended` | 100.0% |
| `ARPU = TotalRevenue / Active_subscribers` | 98.5% |

These determined the imputation strategy (blank subscriber counts are zeros —
proven, since the identity only closes at 100% under that assumption),
identified two perfectly collinear columns, and revealed that ARPU is a
**hybrid** feature dividing revenue by an engagement variable.

**Cleaning.** No rows removed. Blanks imputed as zeros per Identity 2; one
missing ARPU recomputed via Identity 3; `Sliver` typo merged into `Silver`;
identifier column dropped (26 values were corrupted to scientific notation).

**Leakage audit.** `Suspended_subscribers` was the main leakage suspect —
suspension often accompanies leaving. Tested and cleared: only 31 of 549
churners have any suspension, and the churn-rate lift is 6.39% → 8.81%.
Retained as a weak legitimate predictor.

**Class imbalance.** `class_weight="balanced"` (LR, RF) and
`scale_pos_weight=14.40` (XGBoost), computed from training data only. SMOTE was
rejected: with 439 training churners and near-zero signal, synthesising minority
points risks manufacturing structure that isn't there.

**Evaluation.** Stratified 80/20 split. Feature groups compared by 5-fold
stratified CV repeated 5 times (**225 fits**) on the training set, with paired
t-tests across the 25 matched folds. Test set touched once.

## Experiments

| | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| **Financial** (5 features) | ✓ | ✓ | ✓ |
| **Engagement** (4 features) | ✓ | ✓ | ✓ |
| **Combined** (9 features) | ✓ | ✓ | ✓ |

- **Financial:** `TotalRevenue`, `AvgMobileRevenue`, `AvgFIXRevenue`, `ARPU`, `Value_Segment`
- **Engagement:** `Active_subscribers`, `Not_Active_subscribers`, `Suspended_subscribers`, `Total_SUBs`
- **Excluded:** `PID`, `Billing_ZIP`, `KA_name`, `EffectiveSegment` — none belong to
  either research family, so including them would confound RQ1

## Results

### Held-out test set (1,691 customers, 110 churners)

| Feature Group | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| — none — | **Baseline (predict "No")** | **0.9349** | 0.000 | 0.000 | 0.000 | 0.5000 | 0.0651 |
| Financial | Logistic Regression | 0.5547 | 0.0753 | 0.5182 | 0.1315 | 0.5663 | 0.0778 |
| Financial | Random Forest | 0.8800 | 0.0811 | 0.0818 | 0.0814 | 0.5497 | 0.0792 |
| Financial | XGBoost | 0.7102 | 0.0778 | 0.3182 | 0.1250 | 0.5637 | 0.0862 |
| Engagement | Logistic Regression | 0.6901 | 0.0860 | 0.3909 | **0.1410** | 0.5501 | 0.0822 |
| Engagement | Random Forest | 0.7132 | 0.0629 | 0.2455 | 0.1002 | **0.4878** | **0.0648** |
| Engagement | XGBoost | 0.6582 | 0.0602 | 0.2909 | 0.0997 | 0.5092 | 0.0709 |
| Combined | Logistic Regression | 0.5618 | 0.0731 | 0.4909 | 0.1272 | 0.5648 | 0.0807 |
| Combined | Random Forest | 0.9101 | **0.1379** | 0.0727 | 0.0952 | **0.5697** | 0.0862 |
| Combined | XGBoost | 0.7380 | 0.0848 | 0.3091 | 0.1331 | 0.5629 | **0.0882** |

Three observations:

1. **The do-nothing baseline beats every model on accuracy.** This is why
   accuracy is not used as a primary metric.
2. **`Engagement + Random Forest` scored PR-AUC 0.0648 against a 0.0651 floor** —
   worse than random ranking. Reported, not hidden.
3. **Best precision is 0.138.** At best, roughly 1 in 7 flagged customers churns.

### RQ1 — paired tests across 25 CV folds

**On ROC-AUC:**

| Model | Financial | Engagement | Diff | p | Significant |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.5724 | 0.5646 | +0.0078 | 0.179 | no |
| Random Forest | 0.5267 | 0.4928 | +0.0339 | <0.001 | yes |
| XGBoost | 0.5372 | 0.5092 | +0.0281 | 0.0001 | yes |

**On PR-AUC — the direction reverses for Logistic Regression:**

| Model | Financial | Engagement | Diff | p | Significant |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.0874 | **0.0910** | −0.0036 | 0.244 | no |
| Random Forest | 0.0750 | 0.0718 | +0.0031 | 0.065 | no |
| XGBoost | 0.0766 | 0.0765 | +0.0001 | 0.951 | no |

**RQ1 answer: no robust difference.** The two significant ROC-AUC results come
from the two *worst-performing* models and are driven by Engagement + Random
Forest collapsing below random. The correct reading is **differential
overfitting**, not predictive advantage: financial features win by degrading
less on near-zero signal.

### RQ2 — does combining help?

| Model | Combined | Financial | Diff | p |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.5695 | 0.5724 | −0.0028 | 0.080 |
| Random Forest | 0.5250 | 0.5267 | −0.0017 | 0.634 |
| XGBoost | 0.5387 | 0.5372 | +0.0015 | 0.633 |

**RQ2 answer: no.** All non-significant; two of three differences negative.

### Sensitivity analysis

| Configuration | ROC-AUC | vs baseline | p |
|---|---:|---:|---:|
| Financial (as defined) | 0.5724 | — | — |
| Financial minus ARPU | **0.5780** | +0.0057 | **0.007** |
| Engagement (as defined) | 0.5646 | — | — |
| Engagement minus Suspended | 0.5645 | −0.0001 | 0.853 |

Dropping the hybrid ARPU feature *improves* the financial group, confirming it
does not inflate the RQ1 comparison. Dropping `Suspended_subscribers` changes
nothing — it is neither a leak nor a driver.

### Model selection stability

The CV ranking and the single-split test ranking of the nine configurations
were **almost completely reordered**. CV's top pick fell to 4th on test; the
test's top pick was 4th in CV. Only last place agreed. With 110 test churners
and PR-AUC gaps of ~0.01, single-split model selection is unreliable — which is
why RQ1 rests on the paired CV tests, not the test table.

## Explainability

Four independent methods applied to `Combined + XGBoost`:

| Method | Top feature | Value |
|---|---|---|
| Univariate correlation | `AvgMobileRevenue` | r = +0.065 |
| Permutation importance (Δ test PR-AUC) | `AvgMobileRevenue` | 0.0162 ± 0.0058 |
| Mean \|SHAP\| | `AvgMobileRevenue` | 0.4185 |
| Logistic coefficient (odds ratio) | `AvgMobileRevenue` | 1.094 |

**Consistent across all four:** `AvgMobileRevenue` is the strongest single
predictor. `Suspended_subscribers` and `AvgFIXRevenue` are near-useless — both
show **negative** permutation importance, meaning shuffling them *improves*
test performance.

**Group-level SHAP attribution — the nuance that matters:**

| Group | Total \|SHAP\| | Encoded features | Share | **Per feature** |
|---|---:|---:|---:|---:|
| Financial | 1.2038 | 12 | 74.4% | **0.1003** |
| Engagement | 0.4144 | 4 | 25.6% | **0.1036** |

The raw 74/26 split looks decisive for financial features — but it is an
artefact of one-hot encoding inflating the financial column count. Normalised
per feature, the groups are tied, with engagement marginally ahead. **This
independently corroborates the PR-AUC significance tests.**

## Business Insights

**The model should not be deployed.** At the 0.50 threshold it flags 401
customers to catch 34 of 110 churners — 367 false alarms, 8.5% precision
against a 6.5% base rate. Precision barely moves across the threshold sweep
(0.075 → 0.096), so high-confidence predictions are no more trustworthy than
low-confidence ones.

**The one reliable finding is model-independent:** churn rate rises
monotonically with customer value.

| Segment | n | Churn rate |
|---|---:|---:|
| Platinum | 537 | 8.57% |
| Gold | 1,453 | 8.05% |
| Silver | 2,040 | 7.16% |
| Bronze | 3,820 | 5.37% |
| Iron | 246 | 3.66% |

High-value business customers churn at roughly **1.6×** the rate of low-value
ones. Prioritising retention attention by value tier is defensible on this
descriptive basis alone — no model required.

**The disengagement hypothesis is not supported.** Dormant and suspended line
counts — the intuitive churn warning signs — rank at the bottom of every
importance method. Re-engagement campaigns triggered on dormant-line counts
would not be targeting genuine churn risk in this dataset.

**Recommendation: invest in data collection, not modelling.** The ceiling here
is set by feature availability, not algorithm choice.

## Limitations

- Single operator, single country — no generalisation claimed
- **Point-in-time snapshot with no temporal features** — no tenure, contract
  length or end date, complaints, support tickets, or usage trends. Churn is a
  temporal process; the strongest known predictors from the literature are
  absent. This is the most likely cause of the weak signal
- Only 549 churners (110 in test) — demonstrably too few for stable ranking
- `AvgFIXRevenue` zero for 98.6% of customers; `Suspended_subscribers` zero for 95.8%
- Feature groups unequal in size (5 vs 4 raw; 12 vs 4 encoded)
- Minimal hyperparameter tuning by design
- Six paired tests run without multiple-comparison correction (the two
  significant ROC-AUC results survive Bonferroni at α = 0.0083)
- Observational data — all findings associational, no causal claims
- **A negative result is not proof of absence.** These 13 attributes carry
  little signal; churn may be predictable from data not collected here

## Technologies

```text
Python 3.10+ · pandas · NumPy · scikit-learn · XGBoost
SHAP · SciPy · Matplotlib · Seaborn · Jupyter
```

## How to Run

```bash
git clone https://github.com/<your-username>/telecom-churn-xai.git
cd telecom-churn-xai

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download the CSV into data/ -- see data/README.md
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

Run all cells top to bottom. `RANDOM_STATE = 42` throughout; results are fully
reproducible. Cell 25 takes 1–3 minutes (225 model fits); everything else is
near-instant.

## Licence

MIT (code). Dataset is CC BY 4.0 — see `data/README.md` for attribution.