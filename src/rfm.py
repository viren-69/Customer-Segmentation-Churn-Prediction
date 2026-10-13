"""RFM customer segmentation on the Olist marketplace (database from Project 1).

RFM describes each customer with three numbers:
  Recency   - days since their last order (lower = more engaged)
  Frequency - how many orders they placed
  Monetary  - how much they spent in total (item prices, freight excluded)

K-Means then groups customers with similar RFM values into segments.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL, RANDOM_STATE, RFM_SNAPSHOT_DATE

# One row per real person (customer_unique_id, NOT customer_id: see Project 1),
# using the same revenue definition as Project 1.
RFM_SQL = """
SELECT
    c.customer_unique_id                                    AS customer_id,
    MAX(o.order_purchase_timestamp)                         AS last_order,
    COUNT(DISTINCT o.order_id)                              AS frequency,
    SUM(oi.price)                                           AS monetary
FROM orders o
JOIN customers c    ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id   = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
  AND o.order_purchase_timestamp < :snapshot
GROUP BY c.customer_unique_id
"""


def load_rfm(url: str = DATABASE_URL, snapshot: str = RFM_SNAPSHOT_DATE) -> pd.DataFrame:
    engine = create_engine(url)
    with engine.connect() as conn:
        df = pd.read_sql(text(RFM_SQL), conn, params={"snapshot": snapshot})
    df["recency_days"] = (pd.Timestamp(snapshot) - df["last_order"]).dt.days
    df["monetary"] = df["monetary"].astype(float)
    return df[["customer_id", "recency_days", "frequency", "monetary"]]


def prepare_features(rfm: pd.DataFrame) -> np.ndarray:
    """Make the three columns comparable before K-Means.

    K-Means measures plain distances, so:
      - log1p squashes the long right tails (a few customers spend R$ 10,000+;
        without the log they would get a cluster to themselves),
      - StandardScaler puts all three on the same scale, otherwise "monetary" (hundreds)
        would outweigh "frequency" (1-2) just because of its units.
    """
    X = np.column_stack([
        rfm["recency_days"],
        np.log1p(rfm["frequency"]),
        np.log1p(rfm["monetary"]),
    ])
    return StandardScaler().fit_transform(X)


def choose_k(X: np.ndarray, ks=range(2, 9), sample_size: int = 10_000) -> pd.DataFrame:
    """Score each number of clusters k two ways.

    inertia    - total distance of customers to their cluster centre (always falls as k grows;
                 look for the "elbow" where it stops falling fast).
    silhouette - how well separated the clusters are, from -1 to 1 (higher is better).
                 Computed on a random sample because it is slow on 90K+ rows.
    """
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        score = silhouette_score(X, model.labels_, sample_size=sample_size, random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": score})
    return pd.DataFrame(rows)


def fit_segments(rfm: pd.DataFrame, X: np.ndarray, k: int) -> pd.DataFrame:
    out = rfm.copy()
    out["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    return out


def profile(segmented: pd.DataFrame) -> pd.DataFrame:
    """Average R, F, M and size of every cluster, plus a business name and action."""
    total_revenue = segmented["monetary"].sum()
    prof = segmented.groupby("cluster").agg(
        customers=("customer_id", "size"),
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    prof["share_of_customers_pct"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["share_of_revenue_pct"] = 100 * prof["revenue"] / total_revenue
    names = name_segments(prof, segmented)
    prof["segment"] = prof.index.map(lambda c: names[c][0])
    prof["action"] = prof.index.map(lambda c: names[c][1])
    return prof.sort_values("revenue", ascending=False)


def name_segments(prof: pd.DataFrame, segmented: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Turn cluster numbers into names a marketing team understands.

    K-Means numbers its clusters arbitrarily (cluster 2 today may be cluster 0
    after a re-run), so names are given by comparing each cluster's averages
    with the whole customer base, not by cluster number.
    """
    recent = prof["recency_days"] < segmented["recency_days"].median()
    big_spender = prof["monetary"] > segmented["monetary"].median()
    names = {}
    for c in prof.index:
        if prof.loc[c, "frequency"] >= 1.5:
            names[c] = ("Repeat buyers", "Loyalty rewards, early access; ask for reviews and referrals")
        elif recent[c] and big_spender[c]:
            names[c] = ("Recent big spenders", "Second-purchase offer within 30 days; cross-sell related categories")
        elif recent[c]:
            names[c] = ("Recent small spenders", "Low-cost nudges: free-shipping threshold, bundles")
        elif big_spender[c]:
            names[c] = ("Lapsed big spenders", "Win-back campaign with a personal offer; ask why they left")
        else:
            names[c] = ("Lapsed small spenders", "Cheap automated emails only; don't spend much here")
    return names
