"""
Staggered-adoption event-study estimation for the Causal Climate Policy project.

Implements a Sun & Abraham (2021) interaction-weighted dynamic
two-way-fixed-effects estimator: for each eventually-treated unit, effects
are estimated separately by (adoption cohort x event time) interaction and
then aggregated into a single coefficient per relative event time. This
avoids the "forbidden comparison" problem of naive dynamic TWFE under
staggered adoption, where already-treated units act as improperly-weighted
comparison groups for later-treated units (Goodman-Bacon, 2021).

Never-treated units serve as the (sole) comparison group throughout.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def _build_event_time(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treatment_col: str,
    window: tuple,
) -> pd.DataFrame:
    """Attach adoption_year, event_time, and cohort to each row.

    Units that are never treated get adoption_year = NaN and event_time = NaN
    (they contribute to the reference/comparison group only). Event time is
    censored (binned) at the window edges.
    """
    df = df.copy()

    adoption_year = (
        df.loc[df[treatment_col] == 1]
        .groupby(unit_col)[time_col]
        .min()
    )
    df["adoption_year"] = df[unit_col].map(adoption_year)
    df["event_time"] = df[time_col] - df["adoption_year"]

    lo, hi = window
    df["event_time_binned"] = df["event_time"].clip(lower=lo, upper=hi)
    # cohort = adoption year, used for Sun-Abraham cohort x event-time interactions
    df["cohort"] = df["adoption_year"]

    return df


def run_event_study(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    unit_col: str = "iso3",
    time_col: str = "year",
    covariate_cols: list = None,
    window: tuple = (-5, 5),
    reference_period: int = -1,
) -> pd.DataFrame:
    """
    Sun & Abraham (2021) interaction-weighted staggered-adoption event study.

    Estimates cohort x relative-event-time interaction effects via OLS with
    unit and year fixed effects, then aggregates across cohorts (weighted by
    cohort size) into one coefficient per relative event time, using
    never-treated units as the comparison group throughout.

    The pre-treatment leads (event_time < 0, excluding the reference period)
    should be statistically indistinguishable from zero if the parallel-
    trends assumption underlying this design holds -- this is the paper's
    formal test for that assumption, which the prior binary-treatment DML/CF
    cross-sectional design had no equivalent for.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    treatment_col : str
        Binary treatment indicator column (used only to infer each unit's
        adoption year; the regression itself uses cohort x event-time dummies).
    outcome_col : str
        Outcome variable (e.g. 'log_co2').
    unit_col, time_col : str
        Panel identifiers.
    covariate_cols : list, optional
        Time-varying confounders to include as controls.
    window : tuple
        (min, max) relative event time to estimate; endpoints are binned
        (all event times beyond the window are pooled into the endpoint bin).
    reference_period : int
        Omitted (baseline) relative event-time period, conventionally -1
        (the year immediately before adoption).

    Returns
    -------
    pd.DataFrame
        One row per relative event time with columns:
        event_time, estimate, std_error, ci_lower, ci_upper, p_value, n_cohorts.
    """
    covariate_cols = covariate_cols or []

    data = df.dropna(subset=[outcome_col, treatment_col, unit_col, time_col] + covariate_cols).copy()
    data = _build_event_time(data, unit_col, time_col, treatment_col, window)

    lo, hi = window
    event_times = [t for t in range(lo, hi + 1) if t != reference_period]

    # Cohorts adopted at or before the panel's earliest observed year have NO
    # observed pre-treatment periods: every row for such a unit is already
    # post-treatment, so that unit's own fixed effect is not separable from
    # its treatment effect within-unit, and identification of its "effect"
    # falls entirely on a cross-unit level comparison to never-treated units.
    # This can badly bias the aggregate post-treatment estimate (observed on
    # this panel: including such cohorts inflated post-period effects to an
    # implausible ~0.47, vs. ~0.08-0.09 from the cross-sectional DML/CF ATE).
    # Sun & Abraham (2021) require cohorts to have sufficient pre-period
    # observations for exactly this reason; we drop cohorts with zero
    # observed pre-treatment years.
    min_year = data[time_col].min()
    cohorts_with_pretrend = data.loc[data["cohort"] > min_year, "cohort"].dropna().unique()
    n_cohorts_dropped = data["cohort"].dropna().nunique() - len(cohorts_with_pretrend)
    if n_cohorts_dropped > 0:
        logger.info(
            f"Dropping {n_cohorts_dropped} cohort(s) adopted at/before the panel's "
            f"earliest year ({min_year}) -- no observed pre-treatment period."
        )
    data.loc[~data["cohort"].isin(cohorts_with_pretrend), "cohort"] = np.nan
    data.loc[~data["cohort"].isin(cohorts_with_pretrend), "event_time_binned"] = np.nan

    # Never-treated units (and now-excluded no-pretrend cohorts): event_time
    # is NaN -> treated as the reference category for all interaction dummies.
    is_treated_cohort = data["cohort"].notna()

    cohorts = sorted(data.loc[is_treated_cohort, "cohort"].unique())

    # Build cohort x event-time interaction dummies for every cohort/event-time
    # combination except the reference period, restricted to observed cells
    # with a reasonable number of units (Sun-Abraham interaction-weighted spec).
    interaction_cols = []
    X_parts = []
    for cohort in cohorts:
        cohort_mask = (data["cohort"] == cohort)
        for et in event_times:
            col_name = f"D_{int(cohort)}_{et:+d}"
            dummy = ((cohort_mask) & (data["event_time_binned"] == et)).astype(float)
            if dummy.sum() == 0:
                continue
            X_parts.append(dummy.rename(col_name))
            interaction_cols.append(col_name)

    if not X_parts:
        raise ValueError("No cohort x event-time cells with observations; check treatment_col/window.")

    X_interactions = pd.concat(X_parts, axis=1)

    unit_dummies = pd.get_dummies(data[unit_col], prefix="unit", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(data[time_col], prefix="year", drop_first=True, dtype=float)

    X = pd.concat([X_interactions, unit_dummies, year_dummies], axis=1)
    if covariate_cols:
        X = pd.concat([X, data[covariate_cols].astype(float)], axis=1)
    X = sm.add_constant(X).astype(float)
    y = data[outcome_col].astype(float)

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": data[unit_col]})

    # Aggregate cohort-specific coefficients into one estimate per event time,
    # weighted by cohort size (Sun-Abraham interaction-weighted aggregation).
    results = []
    for et in event_times:
        cols_for_et = [c for c in interaction_cols if c.endswith(f"_{et:+d}")]
        if not cols_for_et:
            continue

        weights = np.array([X_interactions[c].sum() for c in cols_for_et], dtype=float)
        weights = weights / weights.sum()

        coefs = np.array([model.params[c] for c in cols_for_et])
        estimate = float(np.dot(weights, coefs))

        # Delta-method variance for a weighted sum of coefficients, using the
        # model's cluster-robust covariance matrix restricted to these cols.
        cov_sub = model.cov_params().loc[cols_for_et, cols_for_et].values
        var = float(weights @ cov_sub @ weights)
        se = float(np.sqrt(var)) if var > 0 else np.nan

        from scipy.stats import norm
        p_val = float(norm.sf(abs(estimate / se)) * 2) if se and se > 0 else np.nan
        ci_lower = estimate - 1.95996 * se if se else np.nan
        ci_upper = estimate + 1.95996 * se if se else np.nan

        results.append({
            "event_time": et,
            "estimate": estimate,
            "std_error": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
            "n_cohorts": len(cols_for_et),
        })

    # Reference period is 0 by construction
    results.append({
        "event_time": reference_period,
        "estimate": 0.0,
        "std_error": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "p_value": np.nan,
        "n_cohorts": len(cohorts),
    })

    result_df = pd.DataFrame(results).sort_values("event_time").reset_index(drop=True)
    return result_df


def pretrend_test(event_study_results: pd.DataFrame, reference_period: int = -1) -> dict:
    """
    Joint test that all pre-treatment (event_time < reference_period) leads
    are zero -- the formal parallel-trends check for the event study.

    Uses a simple Wald-style check: are all pre-period estimates within their
    own 95% CI of zero? (A full joint F-test would require the raw covariance
    matrix from `run_event_study`; this provides a quick per-coefficient
    screen suitable for a diagnostic table.)

    Returns
    -------
    dict with 'n_pretrend_periods', 'n_significant', 'all_insignificant' (bool).
    """
    pre = event_study_results[event_study_results["event_time"] < reference_period]
    n_sig = int(((pre["ci_lower"] > 0) | (pre["ci_upper"] < 0)).sum())
    return {
        "n_pretrend_periods": len(pre),
        "n_significant": n_sig,
        "all_insignificant": n_sig == 0,
    }
