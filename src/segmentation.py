import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


SEGMENT_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "service_count"
]


def get_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select features used for customer segmentation.
    """
    return df[SEGMENT_FEATURES]


def scale_features(X: pd.DataFrame):
    """
    Scale segmentation features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_kmeans(X_scaled, n_clusters: int = 4, random_state: int = 42):
    """
    Train KMeans clustering model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans


def segment_customers(df: pd.DataFrame, n_clusters: int = 4):
    """
    Full segmentation pipeline.
    Returns updated dataframe, trained KMeans model, and scaler.
    """
    X = get_segmentation_features(df)
    X_scaled, scaler = scale_features(X)
    clusters, kmeans = train_kmeans(X_scaled, n_clusters=n_clusters)

    df = df.copy()
    df["segment"] = clusters

    return df, kmeans, scaler