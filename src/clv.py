import pandas as pd


def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple CLV as MonthlyCharges * tenure.
    """
    df = df.copy()
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    return df


def segment_revenue_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate segment-level CLV and revenue at risk summary.
    Requires 'segment', 'Churn', and 'CLV' columns.
    """
    summary = df.groupby("segment").agg(
        total_customers=("segment", "count"),
        churned_customers=("Churn", lambda x: (x == "Yes").sum()),
        avg_monthly_revenue=("MonthlyCharges", "mean"),
        avg_clv=("CLV", "mean")
    )

    summary["revenue_at_risk"] = (
        summary["churned_customers"] *
        summary["avg_clv"]
    )

    return summary


def compute_retention_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute retention priority score.
    Requires 'CLV' and 'churn_probability' columns.
    """
    df = df.copy()

    if "CLV" not in df.columns:
        raise ValueError("CLV column missing. Run compute_clv() first.")

    if "churn_probability" not in df.columns:
        raise ValueError("churn_probability column missing.")

    df["Retention_Priority"] = df["CLV"] * df["churn_probability"]

    return df