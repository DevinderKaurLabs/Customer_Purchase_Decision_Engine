# Churn Prediction & Retention Engine — Careem AI Challenge

This is an extension of my [Customer Purchase Decision Engine](https://github.com/DevinderKaurLabs/Customer_Purchase_Decision_Engine).

The original project asks: **who is likely to purchase again?**

This challenge version changes the decision problem to:

> **Who is showing evidence of churn, why, and what should marketing test next?**

## What I changed

- Reused the original point-in-time customer feature engineering.
- Changed the target from next-30-day purchase to **no purchase in the next 60 days**.
- Added a temporal train / validation / test split.
- Compared **Logistic Regression** with **Histogram Gradient Boosting**.
- Evaluated **PR-AUC, ROC-AUC, Brier score, lift and calibration**, rather than accuracy alone.
- Added churn risk tiers and customer states.
- Added a retention decision layer that maps risk + customer value to an action.
- Added deterministic **90% treatment / 10% control** assignment by action cell.
- Added permutation feature importance to explain what drives the churn score.

## Important dataset limitation

The public dataset is **Online Retail II**, not Careem data. It contains retail transactions, not rides, food orders, delivery operations, app sessions, cancellations, support contacts or offer exposure.

Therefore:

- the model demonstrates the **data-science approach**;
- the Careem-oriented actions are **testable marketing hypotheses**;
- the project does **not** claim to model Careem's actual churn rate or customer behaviour.

The public dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

## Why churn is defined this way

For this prototype:

**churn_60d = 1** when the customer has no purchase during the 60 days after a point-in-time snapshot.

This is a behavioural proxy. In a production marketplace, I would redefine churn around the actual product's natural usage cycle — for example, completed transactions over a chosen inactivity window — and validate that definition against reactivation behaviour.

## Modelling approach

### 1. Point-in-time features

The feature set comes from the existing project:

- recency
- frequency
- monetary value
- average order value
- tenure
- recent vs previous activity
- order/revenue velocity
- revenue trend
- return behaviour
- inter-purchase rhythm
- recency relative to the customer's own normal rhythm
- country

No future transactions are allowed into the features.

### 2. Model comparison

I compare:

**Logistic Regression**
- transparent baseline
- useful for directional explanations
- relatively easy to calibrate

**Histogram Gradient Boosting**
- captures nonlinear relationships
- can model interactions such as high value + unusual inactivity
- handles missing numeric values natively

The model is selected on validation PR-AUC. The final test window is kept untouched until the end.

### 3. Why PR-AUC matters

Churn is a minority-class problem in many customer datasets. Accuracy can look impressive while missing the customers marketing actually needs to find.

I therefore report:

- PR-AUC
- ROC-AUC
- Brier score
- lift at different population depths
- calibration
- feature importance

## Retention strategy

The model does not simply say:

**"churn probability > 50% → send discount."**

Instead it creates states:

| State | Marketing hypothesis |
|---|---|
| SERVICE_RISK | Fix service/product friction before promotion |
| NEW_UNPROVEN | Drive the second transaction |
| CRITICAL_CHURN_RISK | Personalised win-back, with incentives tested rather than assumed |
| HIGH_CHURN_RISK | Personalised discovery / cross-category relevance |
| WATCH | Low-frequency nurture and monitoring |
| STABLE | Avoid unnecessary paid retention spend |

For a Careem-shaped production implementation, these could become hypotheses across mobility, food, grocery and other relevant services — but the correct action would be learned from Careem's own experiment data.

## How I would take this into Careem

The production version would add behavioural features that are not available in Online Retail II:

- days since last completed ride/order
- change in weekly transaction frequency
- cancellation rate
- failed payment rate
- delivery/service incidents
- app/session recency
- category mix
- cross-service adoption
- price sensitivity
- offer exposure and redemption
- support contacts
- geography/time-of-day patterns
- acquisition cohort
- tenure
- contribution margin / customer value

Then I would separate **propensity to churn** from **propensity to respond to an intervention**.

That distinction matters. A churn model tells us who is at risk. It does not tell us who will be saved by a discount.

The next step would be a controlled retention experiment:

1. stratify customers by churn risk and value;
2. reserve a control group;
3. test personalised recommendation / service recovery / incentive cells;
4. measure incremental retention and contribution margin;
5. feed the treatment outcomes back into an uplift model.

## Run

Download the public Online Retail II workbook and place it at:

`data/raw/online_retail_II.xlsx`

Then:

```bash
pip install -r requirements.txt
python notebooks/10_churn_prediction.py
```

Outputs:

- `outputs/careem_churn/model_comparison.csv`
- `outputs/careem_churn/churn_summary.csv`
- `outputs/careem_churn/churn_lift.csv`
- `outputs/careem_churn/churn_feature_importance.csv`
- `outputs/careem_churn/retention_plan.csv`

## 100-word submission summary

I extended my existing purchase decision engine into a churn prediction and retention system using public Online Retail II data. The pipeline creates features, defines churn as no purchase in the following 60 days, and compares logistic regression with histogram gradient boosting using a temporal train/validation/test split. I evaluate PR-AUC, ROC-AUC, calibration and lift rather than accuracy alone. The model then converts churn risk into states, prioritises customers by risk and value, and maps each segment to a retention action. I also include treatment/control holdouts so recommendations can be tested for incremental impact instead of assuming correlation means real retention.

## Challenge mapping

**Challenge 1 — Churn Predictor:** completed.

The prototype goes beyond prediction by connecting churn risk to a retention decision layer and an experiment design.

