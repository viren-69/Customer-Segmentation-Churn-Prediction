"""SHAP explanations for a fitted churn pipeline.

SHAP answers "why did the model give THIS customer a high churn risk?" It
splits each prediction into one contribution per feature, measured against
the average prediction. Positive = pushes towards churn, negative = towards staying.
Averaging the absolute contributions over many customers gives a global
feature-importance ranking that, unlike Random Forest's built-in importance,
also shows the DIRECTION of each effect.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.telco_data import FEATURES


def shap_explanation(pipeline: Pipeline, X_explain: pd.DataFrame, X_background: pd.DataFrame) -> shap.Explanation:
    """Return SHAP values (in log-odds for LR, probability for RF) for the rows of X_explain.

    X_background is the "typical customer" reference the contributions are measured against
    (a sample of the training set).
    """
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    names = list(prep.get_feature_names_out())
    Xe = pd.DataFrame(prep.transform(X_explain[FEATURES]), columns=names)
    Xb = pd.DataFrame(prep.transform(X_background[FEATURES]), columns=names)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, Xb)       # exact and fast for linear models
        explanation = explainer(Xe)
    elif isinstance(model, RandomForestClassifier):
        explainer = shap.TreeExplainer(model, Xb)         # exact algorithm for tree ensembles
        explanation = explainer(Xe, check_additivity=False)[:, :, 1]  # keep the "churn" class
    else:
        raise TypeError(f"No SHAP explainer configured for {type(model).__name__}")
    return explanation


def mean_abs_shap(explanation: shap.Explanation) -> pd.Series:
    """Global importance: average size of each feature's contribution, largest first."""
    values = np.abs(explanation.values).mean(axis=0)
    return pd.Series(values, index=explanation.feature_names).sort_values(ascending=False)
