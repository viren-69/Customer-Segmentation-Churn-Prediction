"""Load and clean the IBM Telco customer churn data.

The raw file has 33 columns. Only 19 of them may be used to predict churn:
  - 4 columns LEAK the answer (Churn Label/Value, Churn Score, Churn Reason, CLTV),
  - 3 are the same for every row (Count, Country, State),
  - 7 identify a customer or place (ID, city, zip, coordinates) and would let
    the model memorise customers instead of learning patterns.
"""
import pandas as pd

from src.config import TELCO_FILE

TARGET = "churn"

# Columns that contain (or are derived from) the outcome we want to predict.
LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason", "CLTV"]
CONSTANT_COLUMNS = ["Count", "Country", "State"]
ID_COLUMNS = ["CustomerID", "City", "Zip Code", "Lat Long", "Latitude", "Longitude"]

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def snake_case(name: str) -> str:
    """'Tenure Months' -> 'tenure_months'."""
    return name.strip().lower().replace(" ", "_")


def load_raw(path=TELCO_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per customer with the 19 usable features, the target, and two
    reference columns (customer_id, cltv) that are kept for reporting but never used as features."""
    df = raw.copy()
    reference = df[["CustomerID", "CLTV"]].rename(columns={"CustomerID": "customer_id", "CLTV": "cltv"})

    df = df.drop(columns=LEAKAGE_COLUMNS + CONSTANT_COLUMNS + ID_COLUMNS)
    df.columns = [snake_case(c) for c in df.columns]
    df = df.rename(columns={"churn_value": TARGET})

    # 'Total Charges' is text because 11 brand-new customers (tenure 0) have a blank.
    # They haven't been billed yet, so their total is 0, not missing.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    new_customers = df["total_charges"].isna() & (df["tenure_months"] == 0)
    df.loc[new_customers, "total_charges"] = 0.0

    # "No internet service" / "No phone service" mean the same as "No" for an add-on
    # (the customer doesn't have it); the internet_service / phone_service columns
    # already record whether they have the base service. Merging avoids duplicate signals.
    for col in ["multiple_lines", "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies"]:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    if df[FEATURES].isna().any().any():
        raise ValueError("Unexpected missing values after cleaning:\n" + str(df[FEATURES].isna().sum()))

    return pd.concat([reference, df[FEATURES + [TARGET]]], axis=1)


def load_clean() -> pd.DataFrame:
    return clean(load_raw())
