"""
Careem-style Churn Prediction Challenge
---------------------------------------
Reuses the feature engineering and modelling ideas from the parent
Customer Purchase Decision Engine, but changes the business target:

    churn_60d = 1 when a customer makes NO purchase in the next 60 days.

The public dataset is retail, not Careem data. The retention actions are
therefore framed as hypotheses for a marketplace such as Careem, not claims
about Careem customers or internal performance.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import (
    load_retail_data,
    clean_transactions,
    create_orders,
    create_snapshot,
    build_customer_features,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_logistic_model,
    build_gradient_boosting_model,
    evaluate_predictions,
    ranking_metrics,
    permutation_importance_table,
    GROSS_MARGIN_RATE,
)

from sklearn.metrics import confusion_matrix


HORIZON_DAYS = 60
RANDOM_STATE = 42


def build_churn_dataset(orders, returns, purchases, snapshot_dates, horizon_days=60):
    """Build point-in-time snapshots with a future no-purchase churn label."""
    frames = []

    for date in snapshot_dates:
        past_orders, future_orders, past_returns, past_lines = create_snapshot(
            orders, returns, purchases, date, horizon_days
        )

        if past_orders.empty:
            continue

        # Only use snapshots where the full future horizon is observable.
        max_order_date = orders["order_date"].max()
        if pd.Timestamp(date) + pd.Timedelta(days=horizon_days) > max_order_date:
            continue

        customers = build_customer_features(
            past_orders, past_returns, past_lines, date
        )

        active_future = set(
            future_orders.loc[future_orders["order_value"] > 0, "customer_id"]
        )

        customers["churn_60d"] = (~customers.index.isin(active_future)).astype(int)
        customers["snapshot_date"] = pd.Timestamp(date)
        frames.append(customers.reset_index())

    if not frames:
        raise ValueError("No valid snapshots. Check DATA_PATH and date coverage.")

    return pd.concat(frames, ignore_index=True)


def temporal_split(frame):
    """Split by snapshot date so future observations never train the model."""
    dates = sorted(frame["snapshot_date"].unique())
    if len(dates) < 6:
        raise ValueError("Need at least six snapshot dates for a temporal split.")

    train_end = dates[int(len(dates) * 0.70) - 1]
    val_end = dates[int(len(dates) * 0.85) - 1]

    train = frame[frame["snapshot_date"] <= train_end].copy()
    validation = frame[
        (frame["snapshot_date"] > train_end)
        & (frame["snapshot_date"] <= val_end)
    ].copy()
    test = frame[frame["snapshot_date"] > val_end].copy()

    return train, validation, test


def add_risk_tier(probability):
    return pd.cut(
        probability,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    ).astype(str)


def customer_state(row):
    """Business state: risk plus customer history, not an intervention."""
    if row["return_rate"] >= 0.40 and row["return_orders"] >= 2:
        return "SERVICE_RISK"
    if row["frequency"] == 1:
        return "NEW_UNPROVEN"
    if row["churn_probability"] >= 0.75:
        return "CRITICAL_CHURN_RISK"
    if row["churn_probability"] >= 0.50:
        return "HIGH_CHURN_RISK"
    if row["churn_probability"] >= 0.25:
        return "WATCH"
    return "STABLE"


def retention_action(row):
    """
    Careem-shaped retention hypotheses.

    These are deliberately action hypotheses. The public retail dataset does
    not contain Careem channels, rides, orders, complaints or offer exposure.
    """
    state = row["state"]
    value = row["monetary"]

    if state == "SERVICE_RISK":
        return (
            "SERVICE_RECOVERY",
            "Review recent service/product friction before sending a promotion.",
        )

    if state == "NEW_UNPROVEN":
        return (
            "SECOND_TRANSACTION_JOURNEY",
            "Reduce friction to a second transaction with onboarding, discovery and relevant reminders.",
        )

    if state == "CRITICAL_CHURN_RISK" and value >= row["value_threshold"]:
        return (
            "HIGH_VALUE_WINBACK",
            "Use personalised cross-category or loyalty treatment; reserve incentives for tested segments.",
        )

    if state == "CRITICAL_CHURN_RISK":
        return (
            "REACTIVATION_SEQUENCE",
            "Test a short reactivation sequence based on the customer's prior behaviour.",
        )

    if state == "HIGH_CHURN_RISK":
        return (
            "PERSONALISED_DISCOVERY",
            "Recommend relevant categories/services based on previous behaviour rather than a blanket discount.",
        )

    if state == "WATCH":
        return (
            "LIGHT_NURTURE",
            "Use low-frequency reminders and personalised discovery; monitor before increasing spend.",
        )

    return (
        "NO_PAID_RETENTION",
        "Avoid unnecessary incentives; maintain normal lifecycle communication.",
    )


def build_retention_plan(scored):
    out = scored.copy()
    out["value_threshold"] = out["monetary"].quantile(0.70)
    out["state"] = out.apply(customer_state, axis=1)

    actions = out.apply(retention_action, axis=1, result_type="expand")
    actions.columns = ["recommended_action", "action_rationale"]
    out = pd.concat([out, actions], axis=1)

    # Prioritisation heuristic, not expected profit.
    out["priority_score"] = (
        out["churn_probability"]
        * np.log1p(out["monetary"])
    )

    return out.sort_values("priority_score", ascending=False)


def create_experiment_plan(plan):
    """
    Deterministic 90/10 treatment-control assignment within action cells.
    The control is what lets the next cycle estimate incremental retention.
    """
    out = plan.copy()
    keys = (
        out["customer_id"].astype(str)
        + "|"
        + out["recommended_action"].astype(str)
    )

    buckets = pd.util.hash_pandas_object(keys, index=False) % 10
    out["experiment_arm"] = np.where(buckets == 0, "CONTROL", "TREATMENT")

    return out


def run(data_path, output_dir):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_retail_data(data_path)
    purchases, returns = clean_transactions(raw)
    orders = create_orders(purchases)

    # Monthly point-in-time snapshots. The final 60 days are excluded because
    # the churn outcome would not yet be observable.
    start = orders["order_date"].min() + pd.Timedelta(days=90)
    end = orders["order_date"].max() - pd.Timedelta(days=HORIZON_DAYS)
    snapshot_dates = pd.date_range(start=start, end=end, freq="30D")

    model_data = build_churn_dataset(
        orders, returns, purchases, snapshot_dates, HORIZON_DAYS
    )

    model_data = model_data[
        (model_data["monetary"] > 0) & (model_data["frequency"] > 0)
    ].copy()

    train, validation, test = temporal_split(model_data)

    X_train, y_train = train[FEATURES], train["churn_60d"]
    X_val, y_val = validation[FEATURES], validation["churn_60d"]
    X_test, y_test = test[FEATURES], test["churn_60d"]

    candidates = {
        "logistic_regression": build_logistic_model(),
        "gradient_boosting": build_gradient_boosting_model(),
    }

    results = []
    fitted = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        p_val = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val, p_val)
        metrics["model"] = name
        results.append(metrics)

    comparison = pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    )

    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]

    p_test = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, p_test)

    ranking = ranking_metrics(y_test, p_test)
    ranking.to_csv(output_dir / "churn_lift.csv", index=False)

    importance = permutation_importance_table(
        lambda x: best_model.predict_proba(x)[:, 1],
        X_val,
        y_val,
        n_repeats=5,
    )
    importance.to_csv(output_dir / "churn_feature_importance.csv", index=False)

    scored = test.copy()
    scored["churn_probability"] = p_test
    scored["risk_tier"] = add_risk_tier(p_test)
    scored["value_threshold"] = test["monetary"].quantile(0.70)

    plan = build_retention_plan(scored)
    plan = create_experiment_plan(plan)

    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    plan.to_csv(output_dir / "retention_plan.csv", index=False)

    summary = {
        "best_model": best_name,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "test_churn_rate": test_metrics["base_rate"],
        "customers_scored": len(plan),
        "treatment_share": float((plan["experiment_arm"] == "TREATMENT").mean()),
    }

    pd.DataFrame([summary]).to_csv(output_dir / "churn_summary.csv", index=False)

    print("\nMODEL COMPARISON — validation")
    print(comparison.round(4).to_string(index=False))

    print("\nTEST — untouched final time window")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTOP CHURN DRIVERS")
    print(importance.head(10).round(4).to_string(index=False))

    print("\nRETENTION ACTION MIX")
    print(plan["recommended_action"].value_counts().to_string())

    print("\nFiles written to:", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "raw" / "online_retail_II.xlsx"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "careem_churn"),
    )
    args = parser.parse_args()

    run(args.data, args.output)
