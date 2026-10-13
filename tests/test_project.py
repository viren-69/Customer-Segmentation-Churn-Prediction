"""Unit tests for the data cleaning, profit logic and RFM helpers.

They use tiny hand-made data, so they run in about a second with no database
or Excel file.
"""
import numpy as np
import pandas as pd
import pytest

from src.churn_model import OFFER_COST, SAVE_RATE, MONTHS_OF_VALUE, best_threshold, campaign_profit
from src.rfm import name_segments, prepare_features
from src.telco_data import FEATURES, LEAKAGE_COLUMNS, TARGET, clean


def raw_row(**overrides) -> dict:
    """One customer in the raw IBM file's format."""
    row = {
        "CustomerID": "0001-A", "Count": 1, "Country": "United States", "State": "California",
        "City": "Los Angeles", "Zip Code": 90001, "Lat Long": "1, 2", "Latitude": 1.0, "Longitude": 2.0,
        "Gender": "Male", "Senior Citizen": "No", "Partner": "No", "Dependents": "No", "Tenure Months": 5,
        "Phone Service": "Yes", "Multiple Lines": "No", "Internet Service": "DSL", "Online Security": "No",
        "Online Backup": "No", "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No",
        "Streaming Movies": "No", "Contract": "Month-to-month", "Paperless Billing": "Yes",
        "Payment Method": "Electronic check", "Monthly Charges": 50.0, "Total Charges": "250",
        "Churn Label": "Yes", "Churn Value": 1, "Churn Score": 90, "CLTV": 3000, "Churn Reason": "Moved",
    }
    row.update(overrides)
    return row


# ---------------- telco cleaning ----------------

def test_clean_removes_leakage_and_ids():
    df = clean(pd.DataFrame([raw_row()]))
    assert set(df.columns) == {"customer_id", "cltv", *FEATURES, TARGET}
    for leak in ["churn_score", "churn_reason", "churn_label"]:
        assert leak not in df.columns
    assert "Churn Score" in LEAKAGE_COLUMNS


def test_blank_total_charges_for_new_customer_becomes_zero():
    df = clean(pd.DataFrame([raw_row(**{"Tenure Months": 0, "Total Charges": " "})]))
    assert df.loc[0, "total_charges"] == 0.0


def test_no_internet_service_is_merged_into_no():
    df = clean(pd.DataFrame([raw_row(**{"Internet Service": "No", "Online Security": "No internet service",
                                        "Multiple Lines": "No phone service"})]))
    assert df.loc[0, "online_security"] == "No"
    assert df.loc[0, "multiple_lines"] == "No"


def test_unexpected_missing_value_fails_loudly():
    with pytest.raises(ValueError):
        clean(pd.DataFrame([raw_row(**{"Tenure Months": 12, "Total Charges": " "})]))  # blank but NOT new


# ---------------- campaign profit ----------------

def test_campaign_profit_by_hand():
    y = np.array([1, 0, 1])
    proba = np.array([0.9, 0.6, 0.1])
    monthly = np.array([100.0, 100.0, 100.0])
    # Threshold 0.5 contacts the first two: one real churner, one loyal customer.
    expected = SAVE_RATE * MONTHS_OF_VALUE * 100.0 - 2 * OFFER_COST
    assert campaign_profit(y, proba, monthly, 0.5) == pytest.approx(expected)
    assert campaign_profit(y, proba, monthly, 0.95) == 0.0  # nobody contacted


def test_best_threshold_contacts_only_profitable_customers():
    # Customers above 0.5 are all churners, below 0.5 all loyal: best threshold must be just above the loyal ones.
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.2])
    t = best_threshold(y, proba, np.full(4, 100.0))
    assert 0.3 < t <= 0.8


# ---------------- RFM ----------------

def test_rfm_features_are_standardised():
    rfm = pd.DataFrame({"recency_days": [10, 100, 300], "frequency": [1, 1, 3], "monetary": [20.0, 200.0, 5000.0]})
    X = prepare_features(rfm)
    assert X.shape == (3, 3)
    assert np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0), 1)


def test_segment_names_follow_the_profile_not_the_cluster_number():
    customers = pd.DataFrame({"recency_days": [10, 20, 400, 410, 200], "monetary": [500, 30, 600, 25, 300],
                              "frequency": [1, 1, 1, 1, 3]})
    prof = pd.DataFrame({"recency_days": [15, 405, 200], "monetary": [265, 312, 300], "frequency": [1, 1, 3]},
                        index=[7, 3, 0])  # arbitrary cluster numbers
    names = name_segments(prof, customers)
    assert names[0][0] == "Repeat buyers"
    assert names[3][0].startswith("Lapsed")
    assert names[7][0].startswith("Recent")


def test_sensitivity_covers_every_scenario_and_respects_assumptions():
    from src.churn_model import sensitivity
    y = np.array([1, 0, 1, 0]); proba = np.array([0.9, 0.2, 0.7, 0.4]); monthly = np.full(4, 80.0)
    table = sensitivity(y, proba, monthly, y, proba, monthly, offer_costs=(10, 1000), save_rates=(0.5,), months=(12,))
    assert len(table) == 2
    # A $1000 offer can never pay off on $80/month customers, so the best plan contacts (almost) nobody.
    expensive = table[table["offer_cost"] == 1000].iloc[0]
    assert expensive["profit_tuned"] >= expensive["profit_contact_everyone"]
