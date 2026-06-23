"""
Causal ML model wrappers for the Causal Climate Policy project.

Provides high-level interfaces for:
- Double/Debiased Machine Learning (DML)
- Causal Forest (Generalized Random Forest)
- Traditional OLS / Fixed Effects baselines
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class CausalEstimate:
    """Container for a causal effect estimate."""
    method: str
    treatment: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float = np.nan
    n_obs: int = 0
    n_treated: int = 0
    first_stage_r2_y: float = np.nan
    first_stage_r2_t: float = np.nan
    additional_info: dict = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if the 95% CI excludes zero."""
        return not (self.ci_lower <= 0 <= self.ci_upper)

    def to_dict(self) -> dict:
        """Convert to dictionary for tabulation."""
        return {
            "Method": self.method,
            "Treatment": self.treatment,
            "Estimate": f"{self.estimate:.4f}",
            "Std. Error": f"{self.std_error:.4f}",
            "95% CI": f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            "p-value": f"{self.p_value:.4f}" if not np.isnan(self.p_value) else "",
            "N": self.n_obs,
            "Significant": "***" if self.is_significant else "",
        }


def within_transformation(
    df: pd.DataFrame, 
    cols: list, 
    entity_col: str = "iso3", 
    time_col: str = "year"
) -> pd.DataFrame:
    """
    Applies within-transformation (demeaning) for Two-Way Fixed Effects.
    This mathematically isolates the within-country variation over time.
    """
    transformed = df.copy()
    for col in cols:
        entity_mean = df.groupby(entity_col)[col].transform("mean")
        time_mean = df.groupby(time_col)[col].transform("mean")
        overall_mean = df[col].mean()
        transformed[col] = df[col] - entity_mean - time_mean + overall_mean
    return transformed


def run_ols_baseline(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    fe_cols: list = None,
) -> CausalEstimate:
    """
    Run OLS regression as a baseline (non-causal) estimate.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    outcome_col : str
        Dependent variable column.
    treatment_col : str
        Treatment variable column.
    covariate_cols : list
        Control variable columns.
    fe_cols : list, optional
        Columns to use for fixed effects dummies.

    Returns
    -------
    CausalEstimate
        OLS coefficient estimate with standard errors.
    """
    import statsmodels.api as sm

    data = df.dropna(subset=[outcome_col, treatment_col] + covariate_cols).copy()

    X = data[[treatment_col] + covariate_cols].copy()

    # Add fixed effects if specified
    if fe_cols:
        for fe_col in fe_cols:
            dummies = pd.get_dummies(data[fe_col], prefix=fe_col, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    X = X.astype(float)
    y = data[outcome_col].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC1")  # Heteroskedasticity-robust SEs

    coef = model.params[treatment_col]
    se = model.bse[treatment_col]
    ci = model.conf_int().loc[treatment_col]

    return CausalEstimate(
        method="OLS",
        treatment=treatment_col,
        estimate=coef,
        std_error=se,
        ci_lower=ci[0],
        ci_upper=ci[1],
        p_value=model.pvalues[treatment_col],
        n_obs=int(model.nobs),
        n_treated=int(data[treatment_col].sum()),
    )


def run_dml(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    n_folds: int = 5,
    first_stage_model: str = "xgboost",
) -> CausalEstimate:
    """
    Run Double/Debiased Machine Learning estimation.

    Uses the econml library for the core DML procedure with
    cross-fitting (sample splitting) for valid inference.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel (no missing values in specified columns).
    outcome_col : str
        Outcome variable (e.g., 'log_co2').
    treatment_col : str
        Treatment variable (e.g., 'treat_carbon_pricing').
    covariate_cols : list
        Confounder columns.
    n_folds : int
        Number of cross-fitting folds.
    first_stage_model : str
        First-stage ML model: 'xgboost', 'random_forest', or 'gradient_boosting'.

    Returns
    -------
    CausalEstimate
        DML treatment effect estimate.
    """
    from econml.dml import LinearDML

    data = df.dropna(subset=[outcome_col, treatment_col] + covariate_cols).copy()

    # Apply Within-Transformation for TWFE
    transform_cols = [outcome_col, treatment_col] + covariate_cols
    data = within_transformation(data, transform_cols)

    Y = data[outcome_col].values
    T = data[treatment_col].values.reshape(-1, 1)
    X = None  # No effect modifiers for ATE (use W only)
    W = data[covariate_cols].values

    # Select first-stage models
    if first_stage_model == "xgboost":
        try:
            from xgboost import XGBRegressor
            model_y = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            model_t = XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            logger.warning("XGBoost not available, falling back to GradientBoosting")
            model_y = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
            model_t = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    elif first_stage_model == "random_forest":
        model_y = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    else:
        model_y = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        model_t = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)

    # Fit DML
    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        cv=n_folds,
        random_state=42,
    )

    dml.fit(Y, T, X=X, W=W)

    # Extract estimates
    ate = dml.ate()
    ate_interval = dml.ate_interval(alpha=0.05)

    # Inference
    se = float((ate_interval[1] - ate_interval[0]) / (2 * 1.95996))
    from scipy.stats import norm
    p_val = float(norm.sf(abs(float(ate) / se)) * 2) if se > 0 else np.nan

    return CausalEstimate(
        method=f"DML ({first_stage_model})",
        treatment=treatment_col,
        estimate=float(ate),
        std_error=se,
        ci_lower=float(ate_interval[0]),
        ci_upper=float(ate_interval[1]),
        p_value=p_val,
        n_obs=len(data),
        n_treated=int(data[treatment_col].sum()),
        additional_info={"n_folds": n_folds, "first_stage": first_stage_model},
    )


def run_causal_forest(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    effect_modifier_cols: list = None,
    n_estimators: int = 1000,
) -> tuple:
    """
    Run Causal Forest for heterogeneous treatment effect estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    outcome_col : str
        Outcome variable.
    treatment_col : str
        Treatment variable (binary).
    covariate_cols : list
        Confounder columns (used in both stages).
    effect_modifier_cols : list, optional
        Variables that may modify the treatment effect.
        If None, uses covariate_cols.
    n_estimators : int
        Number of trees in the forest.

    Returns
    -------
    tuple
        (CausalEstimate for ATE, pd.DataFrame of individual CATEs)
    """
    from econml.grf import CausalForest

    if effect_modifier_cols is None:
        effect_modifier_cols = covariate_cols

    data = df.dropna(
        subset=[outcome_col, treatment_col] + covariate_cols + effect_modifier_cols
    ).copy()

    # Apply Within-Transformation for TWFE
    transform_cols = list(set([outcome_col, treatment_col] + covariate_cols + effect_modifier_cols))
    data = within_transformation(data, transform_cols)

    Y = data[outcome_col].values
    T = data[treatment_col].values
    # Combine effect modifiers and covariates for GRF
    all_x_cols = list(set(effect_modifier_cols + covariate_cols))
    X_all = data[all_x_cols].values

    # Fit Causal Forest
    cf = CausalForest(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    cf.fit(X_all, T, Y)

    # Individual CATEs (predict only on the effect modifiers or all?)
    # GRF expects the same X shape for prediction, so we must predict with X_all
    cate_pred = cf.predict(X_all)
    cate_intervals = cf.predict_interval(X_all, alpha=0.05)

    # Build CATE dataframe
    cate_df = data[["iso3", "year"] + effect_modifier_cols].copy()
    cate_df["cate"] = cate_pred.flatten()
    cate_df["cate_lower"] = cate_intervals[0].flatten()
    cate_df["cate_upper"] = cate_intervals[1].flatten()

    # ATE from Causal Forest (mean of CATEs)
    ate = float(np.mean(cate_pred))
    ate_se = float(np.std(cate_pred) / np.sqrt(len(cate_pred)))

    from scipy.stats import norm
    p_val = float(norm.sf(abs(ate / ate_se)) * 2) if ate_se > 0 else np.nan

    ate_estimate = CausalEstimate(
        method="Causal Forest",
        treatment=treatment_col,
        estimate=ate,
        std_error=ate_se,
        ci_lower=float(np.mean(cate_intervals[0])),
        ci_upper=float(np.mean(cate_intervals[1])),
        p_value=p_val,
        n_obs=len(data),
        n_treated=int(T.sum()),
        additional_info={"n_estimators": n_estimators},
    )

    return ate_estimate, cate_df


def compare_methods(estimates: list[CausalEstimate]) -> pd.DataFrame:
    """
    Create a comparison table of causal estimates from different methods.

    Parameters
    ----------
    estimates : list of CausalEstimate
        Estimates to compare.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    rows = [e.to_dict() for e in estimates]
    return pd.DataFrame(rows)
