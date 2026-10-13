"""Train, compare and evaluate churn models.

Workflow (each step is one function below):
  1. split()            80% train / 20% test, stratified so both keep the 26.5% churn rate.
                        The test set is locked away until the very end.
  2. build_pipelines()  preprocessing + model in one sklearn Pipeline, so the exact same
                        transformations are applied in training, cross-validation and scoring.
  3. tune()             5-fold cross-validation ON THE TRAINING SET picks each model's
                        settings and tells us which model is better.
  4. evaluate()         final, one-time check on the untouched test set.
  5. best_threshold()   chooses who to contact by expected PROFIT, not by the default 0.5.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.telco_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---- Business assumptions for the retention campaign (illustrative; change to real figures) ----
OFFER_COST = 50.0        # cost per contacted customer: discount + agent time, in $
SAVE_RATE = 0.30         # share of real churners who stay because of the offer
MONTHS_OF_VALUE = 12     # a saved customer pays roughly 12 more months of their monthly charge


def split(df: pd.DataFrame, test_size: float = 0.2):
    return train_test_split(df, df[TARGET], test_size=test_size, stratify=df[TARGET], random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """Numbers -> standardised (mean 0, std 1); categories -> one 0/1 column per value.

    drop="if_binary" keeps a single column for yes/no features (e.g. partner_Yes)
    instead of two perfectly opposite ones.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def build_pipelines() -> dict[str, tuple[Pipeline, dict]]:
    """Each model with the settings grid that cross-validation will search."""
    return {
        "Baseline (predict nobody churns)": (
            Pipeline([("prep", build_preprocessor()), ("model", DummyClassifier(strategy="prior"))]),
            {},
        ),
        "Logistic Regression": (
            Pipeline([("prep", build_preprocessor()),
                      ("model", LogisticRegression(max_iter=2000))]),
            # C = 1 / regularisation strength: small C = simpler model with smaller coefficients.
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            Pipeline([("prep", build_preprocessor()),
                      ("model", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))]),
            # Bigger leaves and limited depth stop the trees memorising individual customers.
            {"model__min_samples_leaf": [1, 5, 10, 20], "model__max_depth": [None, 8, 12]},
        ),
    }


@dataclass
class TunedModel:
    name: str
    pipeline: Pipeline       # refitted on the whole training set with the best settings
    best_params: dict
    cv_roc_auc: float        # mean ROC-AUC over the 5 validation folds
    cv_roc_auc_std: float


def tune(X_train: pd.DataFrame, y_train: pd.Series) -> list[TunedModel]:
    results = []
    for name, (pipeline, grid) in build_pipelines().items():
        search = GridSearchCV(pipeline, grid or {}, scoring="roc_auc", cv=CV, n_jobs=-1)
        search.fit(X_train[FEATURES], y_train)
        i = search.best_index_
        results.append(TunedModel(name, search.best_estimator_, search.best_params_,
                                  float(search.cv_results_["mean_test_score"][i]),
                                  float(search.cv_results_["std_test_score"][i])))
    return results


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """All the metrics that matter for an imbalanced yes/no problem.

    ROC-AUC  - how well the model RANKS churners above non-churners (0.5 = random, 1 = perfect).
    PR-AUC   - like ROC-AUC but focused on the churners; the baseline is the churn rate (0.27).
    Precision - of the customers we flag, how many really churn.
    Recall    - of the customers who really churn, how many we flag.
    """
    proba = pipeline.predict_proba(X[FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "threshold": threshold,
    }


def out_of_fold_proba(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Churn probability for every TRAINING customer, each predicted by a model that didn't see it.

    Used to choose the threshold without touching the test set.
    """
    return cross_val_predict(pipeline, X_train[FEATURES], y_train, cv=CV, method="predict_proba")[:, 1]


def campaign_profit(y: np.ndarray, proba: np.ndarray, monthly_charges: np.ndarray, threshold: float,
                    offer_cost: float = OFFER_COST, save_rate: float = SAVE_RATE,
                    months_of_value: float = MONTHS_OF_VALUE) -> float:
    """Expected profit of contacting every customer whose churn probability >= threshold.

    For each contacted customer we pay offer_cost. If they were really going to churn,
    the offer keeps them with probability save_rate, and keeping them is worth
    months_of_value x their monthly charge. Non-churners we contact cost money and gain nothing.
    """
    contacted = proba >= threshold
    saved_value = (np.asarray(y)[contacted] * save_rate * months_of_value * np.asarray(monthly_charges)[contacted]).sum()
    return float(saved_value - offer_cost * contacted.sum())


# Rounded so each candidate is exactly 0.05, 0.06, ...: np.arange alone gives values like
# 0.30000000000000004, and the threshold we report must be the one we evaluated.
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def profit_curve(y, proba, monthly_charges, thresholds=THRESHOLDS, **assumptions) -> pd.DataFrame:
    """Profit at every candidate threshold. **assumptions can override offer_cost, save_rate, months_of_value."""
    return pd.DataFrame({
        "threshold": thresholds,
        "profit": [campaign_profit(y, proba, monthly_charges, t, **assumptions) for t in thresholds],
        "contacted_share": [(proba >= t).mean() for t in thresholds],
    })


def best_threshold(y, proba, monthly_charges, **assumptions) -> float:
    curve = profit_curve(y, proba, monthly_charges, **assumptions)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def sensitivity(y_train, oof_proba, train_monthly, y_test, test_proba, test_monthly,
                offer_costs=(25, 50, 100), save_rates=(0.2, 0.3, 0.5), months=(6, 12, 24)) -> pd.DataFrame:
    """Repeat the whole threshold choice under many different business assumptions.

    For each scenario: choose the threshold on the TRAINING out-of-fold predictions,
    then compare three strategies on the TEST set. Shows whether "use the model with
    a tuned threshold" stays the best plan when the assumptions are wrong.
    """
    rows = []
    for cost in offer_costs:
        for rate in save_rates:
            for m in months:
                a = {"offer_cost": cost, "save_rate": rate, "months_of_value": m}
                t = best_threshold(y_train, oof_proba, train_monthly, **a)
                rows.append({**a, "best_threshold": t,
                             "profit_contact_everyone": campaign_profit(y_test, test_proba, test_monthly, 0.0, **a),
                             "profit_default_0.5": campaign_profit(y_test, test_proba, test_monthly, 0.5, **a),
                             "profit_tuned": campaign_profit(y_test, test_proba, test_monthly, t, **a)})
    return pd.DataFrame(rows)


def logistic_odds_ratios(pipeline: Pipeline) -> pd.DataFrame:
    """For logistic regression: how each feature multiplies the odds of churning.

    odds_ratio 2.0 = doubles the odds; 0.5 = halves them. Numeric features are
    standardised, so their ratio is per +1 standard deviation (e.g. ~24 months of tenure).
    """
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
