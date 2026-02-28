import pandas as pd


def load_data(path):
    """Load dataset from given path."""
    return pd.read_csv(path)


def clean_total_charges(df):
    """Convert TotalCharges to numeric and handle missing values."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def convert_binary_columns(df):
    """Convert binary numeric columns to categorical where appropriate."""
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def create_tenure_group(df):
    """Create tenure lifecycle groups."""
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, 60, 72],
        labels=["0-12 Months", "12-36 Months", "36-60 Months", "60+ Months"]
    )
    return df


def create_service_count(df):
    """Create service count feature based on subscribed services."""
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x in ["No", "No internet service", "No phone service"] else 1)

    df["service_count"] = df[service_cols].sum(axis=1)
    return df


def preprocess_data(path):
    """Full preprocessing pipeline."""
    df = load_data(path)
    df = clean_total_charges(df)
    df = convert_binary_columns(df)
    df = create_tenure_group(df)
    df = create_service_count(df)
    return df