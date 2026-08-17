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

    Used only by `run_ols_baseline`, where the Frisch-Waugh-Lovell theorem
    guarantees demean-then-regress is exactly equivalent to including full
    fixed-effect dummy sets in a single linear regression. NOT used by
    `run_dml`/`run_causal_forest` (see `_add_fe_dummies`): FWL does not
    extend to nonlinear/ML nuisance estimation, and because this function
    demeans using each row's own group mean computed over the *entire*
    sample, applying it before DML's cross-fitting would leak information
    from a fold into the transformation applied to observations held out
    from that same fold.
    """
    transformed = df.copy()
    for col in cols:
        entity_mean = df.groupby(entity_col)[col].transform("mean")
        time_mean = df.groupby(time_col)[col].transform("mean")
        overall_mean = df[col].mean()
        transformed[col] = df[col] - entity_mean - time_mean + overall_mean
    return transformed


def _add_fe_dummies(
    df: pd.DataFrame, entity_col: str = "iso3", time_col: str = "year"
) -> pd.DataFrame:
    """
    Build one-hot country/year fixed-effect indicator columns to be passed
    as nuisance controls (W) to DML/Causal Forest, in place of linear
    within-transformation demeaning.

    Rationale: `within_transformation` is a linear projection, and by
    Frisch-Waugh-Lovell is exact only for a subsequent *linear* regression
    (as in `run_ols_baseline`). DML's nuisance functions are flexible/
    nonlinear ML models (XGBoost etc.), so pre-demeaning and then fitting a
    nonlinear model to the residuals is not guaranteed equivalent to fitting
    that nonlinear model with fixed effects included directly, and the
    demeaning statistics (computed over the full sample before any
    cross-fitting split) would leak fold-held-out information into every
    row's transformed value. Passing country/year as indicator columns
    lets the cross-fitted ML nuisance models absorb country/year
    heterogeneity nonparametrically, with no leakage, since each dummy
    column is a static structural indicator independent of the outcome.
    """
    entity_dummies = pd.get_dummies(df[entity_col], prefix=entity_col, drop_first=True, dtype=float)
    time_dummies = pd.get_dummies(df[time_col], prefix=time_col, drop_first=True, dtype=float)
    return pd.concat([entity_dummies, time_dummies], axis=1)


def run_ols_baseline(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    fe_cols: list = None,
    cluster_col: str = "iso3",
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
    cluster_col : str, optional
        Panel unit column to cluster standard errors on (default 'iso3').
        Country-year panels have serially correlated errors within a country;
        heteroskedasticity-robust (HC1) SEs ignore this and understate
        uncertainty. If the column is absent, falls back to HC1 with a
        logged warning rather than failing silently.

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

    if cluster_col and cluster_col in data.columns:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    else:
        logger.warning(f"cluster_col '{cluster_col}' not found; falling back to HC1 robust SEs")
        model = sm.OLS(y, X).fit(cov_type="HC1")

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
        additional_info={"r2": float(model.rsquared)},
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
    n_treated = int(data[treatment_col].sum())

    # Country/year fixed effects are passed to DML's cross-fitted nuisance
    # models as indicator controls in W, not removed via linear
    # within-transformation beforehand -- see `_add_fe_dummies` docstring
    # for why: FWL-based demeaning is not valid ahead of a nonlinear
    # nuisance model, and doing so leaks fold information into the
    # transformed values before cross-fitting ever splits the sample.
    fe_dummies = _add_fe_dummies(data)
    W_cols = covariate_cols
    W = np.hstack([data[W_cols].values, fe_dummies.values])

    Y = data[outcome_col].values
    T = data[treatment_col].values.reshape(-1, 1)
    X = None  # No effect modifiers for ATE (use W only)

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
        n_treated=n_treated,
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
    # NOTE: previously used econml.grf.CausalForest directly and computed the
    # ATE's std_error as std(individual CATEs)/sqrt(n) while ci_lower/ci_upper
    # were the mean of each unit's *individual* 95% CATE interval bounds. Those
    # are two different, mutually inconsistent quantities (a population-ATE SE
    # vs. the average width of unit-level prediction intervals), which produced
    # CIs ~50x wider than the SE implied and could cross zero while p<<0.05.
    # econml.dml.CausalForestDML exposes ate()/ate_interval() that are
    # internally consistent (interval derived from the same asymptotic
    # variance as the SE), matching the approach already used in run_dml.
    from econml.dml import CausalForestDML

    if effect_modifier_cols is None:
        effect_modifier_cols = covariate_cols

    data = df.dropna(
        subset=[outcome_col, treatment_col] + covariate_cols + effect_modifier_cols
    ).copy()
    n_treated = int(data[treatment_col].sum())

    # Country/year fixed effects are passed as nuisance-only controls (W),
    # not removed via linear within-transformation beforehand -- see
    # `_add_fe_dummies` and the matching note in `run_dml` for why linear
    # demeaning ahead of a nonlinear/cross-fitted nuisance model is not
    # valid and leaks fold information into the transformed values.
    fe_dummies = _add_fe_dummies(data)
    W_fe = fe_dummies.values

    Y = data[outcome_col].values
    T = data[treatment_col].values
    # Combine effect modifiers and covariates for GRF (X: used for CATE
    # heterogeneity splitting; W_fe: nuisance-only country/year controls)
    all_x_cols = list(set(effect_modifier_cols + covariate_cols))
    X_all = data[all_x_cols].values

    # Fit Causal Forest via the DML wrapper, which provides proper ATE inference
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        model_t=GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    cf.fit(Y, T, X=X_all, W=W_fe)

    # Individual CATEs and their per-unit intervals (for heterogeneity plots)
    cate_pred = cf.effect(X_all)
    cate_intervals = cf.effect_interval(X_all, alpha=0.05)

    cate_df = data[["iso3", "year"] + effect_modifier_cols].copy()
    cate_df["cate"] = np.asarray(cate_pred).flatten()
    cate_df["cate_lower"] = np.asarray(cate_intervals[0]).flatten()
    cate_df["cate_upper"] = np.asarray(cate_intervals[1]).flatten()

    # Population ATE with an internally-consistent SE/CI (not derived by
    # averaging individual CATE intervals)
    ate = float(cf.ate(X_all))
    ate_lb, ate_ub = cf.ate_interval(X_all, alpha=0.05)
    ate_lb, ate_ub = float(ate_lb), float(ate_ub)
    ate_se = float((ate_ub - ate_lb) / (2 * 1.95996))

    from scipy.stats import norm
    p_val = float(norm.sf(abs(ate / ate_se)) * 2) if ate_se > 0 else np.nan

    ate_estimate = CausalEstimate(
        method="Causal Forest",
        treatment=treatment_col,
        estimate=ate,
        std_error=ate_se,
        ci_lower=ate_lb,
        ci_upper=ate_ub,
        p_value=p_val,
        n_obs=len(data),
        n_treated=n_treated,
        additional_info={"n_estimators": n_estimators},
    )

    return ate_estimate, cate_df


def run_models_by_sector(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list,
    sectors: list = None,
    methods: list = ("ols", "dml"),
    **method_kwargs,
) -> pd.DataFrame:
    """
    Run the same OLS/DML/Causal-Forest estimators used for the national
    analysis, separately for each sector's outcome column.

    Sector outcomes are expected as `log_co2_<sector>` (created by
    `create_outcome_variables` when sector columns like `co2_power` are
    present in the panel). Deliberately reuses `run_ols_baseline`/`run_dml`/
    `run_causal_forest` directly (the same functions used for the national
    ATE) instead of a separate implementation, so sector estimates share one
    sign/outcome convention with the national results -- this is what the
    prior `sector_estimates.csv` (generated by an untraceable, since-lost
    ad-hoc script) failed to do, producing sign-inconsistent results.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel with sector outcome columns present.
    treatment_col : str
        Treatment column (national-level; policy treatment is not
        sector-specific in the underlying Climate Policy Database, so the
        same treatment indicator applies across all sector regressions for
        a given country-year -- sector heterogeneity is captured via the
        sector-specific outcome, not a sector-specific treatment).
    covariate_cols : list
        Confounder columns.
    sectors : list, optional
        Sector names to run (default: power, transport, industry,
        agriculture -- whichever have a matching log_co2_<sector> column).
    methods : list
        Subset of ("ols", "dml", "causal_forest") to run per sector.

    Returns
    -------
    pd.DataFrame
        One row per (sector, method) with the same columns as
        `compare_methods` output, plus a 'Sector' column.
    """
    if sectors is None:
        sectors = ["power", "transport", "industry", "agriculture"]

    rows = []
    for sector in sectors:
        outcome_col = f"log_co2_{sector}"
        if outcome_col not in df.columns:
            logger.warning(f"Sector outcome column '{outcome_col}' not found, skipping sector '{sector}'")
            continue

        if "ols" in methods:
            est = run_ols_baseline(df, outcome_col, treatment_col, covariate_cols,
                                    fe_cols=method_kwargs.get("fe_cols"))
            row = est.to_dict()
            row["Sector"] = sector
            rows.append(row)

        if "dml" in methods:
            est = run_dml(df, outcome_col, treatment_col, covariate_cols,
                           n_folds=method_kwargs.get("n_folds", 5),
                           first_stage_model=method_kwargs.get("first_stage_model", "xgboost"))
            row = est.to_dict()
            row["Sector"] = sector
            rows.append(row)

        if "causal_forest" in methods:
            est, _ = run_causal_forest(df, outcome_col, treatment_col, covariate_cols,
                                        effect_modifier_cols=method_kwargs.get("effect_modifier_cols"),
                                        n_estimators=method_kwargs.get("n_estimators", 1000))
            row = est.to_dict()
            row["Sector"] = sector
            rows.append(row)

    return pd.DataFrame(rows)


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
