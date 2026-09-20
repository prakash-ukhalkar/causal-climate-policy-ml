"""
Robustness checks and sensitivity analyses for the Causal Climate Policy project.

Implements:
- Placebo tests (temporal and outcome)
- Leave-one-out stability analysis
- Alternative specification tests
- Sensitivity to unobserved confounding
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def placebo_test_time(
    df: pd.DataFrame,
    estimation_func: Callable,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list,
    unit_col: str = "iso3",
    time_col: str = "year",
    placebo_shifts: list = None,
    **estimation_kwargs,
) -> pd.DataFrame:
    """
    Temporal placebo test: artificially shift each unit's own adoption year backward.

    If the causal effect is real, pseudo-treatments in pre-policy periods
    should yield estimates close to zero.

    Reconstructs each unit's actual adoption year from its treatment history
    (the first year `treatment_col` switches to 1) and shifts THAT year
    backward per unit, then rebuilds the binary treatment indicator from the
    shifted adoption year. Never-treated units remain untreated in every
    placebo draw. This replaces a prior implementation that called
    `df[treatment_col].shift(-shift)` directly on the pooled, unsorted
    treatment column — which shifted across country boundaries (mixing one
    country's treatment status into another's row) and did not respect each
    country's own staggered adoption timing.

    Each unit-year with an actual adoption year is restricted to its
    pre-actual-adoption window (year < actual adoption year) before the
    placebo indicator is applied. Without this restriction, the placebo
    indicator `year >= placebo_adoption_year` is absorbing and never turns
    back off, so it also covers every year from the *true* adoption year
    onward — i.e. the "placebo-treated" group is mostly genuinely treated
    observations with a few extra pre-years mixed in, not a pre-treatment
    pseudo-effect at all. On this project's real panel, an unrestricted
    shift of -3 put 87% of "placebo-treated" observations after the true
    adoption date, which mechanically produced large, significant "placebo"
    estimates that were really just the true effect leaking through.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    estimation_func : callable
        Causal estimation function (e.g., run_dml).
    treatment_col : str
        Actual (binary) treatment column.
    outcome_col : str
        Outcome column.
    covariate_cols : list
        Confounder columns.
    unit_col : str
        Panel unit identifier (default 'iso3').
    time_col : str
        Panel time identifier (default 'year').
    placebo_shifts : list
        Years to shift backward (e.g., [-5, -3, -1]). 0 = actual treatment.

    Returns
    -------
    pd.DataFrame
        Placebo estimates for each shift.
    """
    if placebo_shifts is None:
        placebo_shifts = [-5, -4, -3, -2, -1, 0]

    # Reconstruct each unit's actual adoption year (first year treated==1)
    adoption_year = (
        df.loc[df[treatment_col] == 1]
        .groupby(unit_col)[time_col]
        .min()
    )

    results = []

    for shift in placebo_shifts:
        df_placebo = df.copy()

        if shift != 0:
            label = f"Placebo (t{shift:+d})"
            # Shift is expressed as "years earlier"; a shift of -3 means the
            # placebo adoption year is 3 years before the true one.
            placebo_adoption_year = adoption_year + shift
            df_placebo["_placebo_adoption_year"] = df_placebo[unit_col].map(placebo_adoption_year)
            df_placebo["_actual_adoption_year"] = df_placebo[unit_col].map(adoption_year)

            # Restrict to each unit's pre-actual-adoption window so the
            # placebo indicator cannot pick up genuinely treated years
            # (never-adopting units have no actual adoption year and are
            # kept in full as the untreated comparison group).
            pre_period_mask = (
                df_placebo["_actual_adoption_year"].isna()
                | (df_placebo[time_col] < df_placebo["_actual_adoption_year"])
            )
            df_placebo = df_placebo.loc[pre_period_mask].copy()

            df_placebo[treatment_col] = (
                (df_placebo["_placebo_adoption_year"].notna())
                & (df_placebo[time_col] >= df_placebo["_placebo_adoption_year"])
            ).astype(int)
            df_placebo = df_placebo.drop(columns=["_placebo_adoption_year", "_actual_adoption_year"])
        else:
            label = "Actual"

        try:
            estimate = estimation_func(
                df_placebo, outcome_col, treatment_col, covariate_cols,
                **estimation_kwargs,
            )
            results.append({
                "shift": shift,
                "label": label,
                "estimate": estimate.estimate,
                "ci_lower": estimate.ci_lower,
                "ci_upper": estimate.ci_upper,
                "std_error": estimate.std_error,
                "n_obs": estimate.n_obs,
            })
        except Exception as e:
            logger.warning(f"Placebo test failed for shift={shift}: {e}")
            results.append({
                "shift": shift,
                "label": label,
                "estimate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "std_error": np.nan,
                "n_obs": 0,
            })

    return pd.DataFrame(results)


def placebo_test_outcome(
    df: pd.DataFrame,
    estimation_func: Callable,
    treatment_col: str,
    placebo_outcomes: list,
    covariate_cols: list,
    **estimation_kwargs,
) -> pd.DataFrame:
    """
    Outcome placebo test: estimate treatment effects on unrelated outcomes.

    If the policy has a true causal effect on emissions, it should NOT
    significantly affect unrelated outcomes like population growth.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    estimation_func : callable
        Causal estimation function.
    treatment_col : str
        Treatment column.
    placebo_outcomes : list
        Column names of placebo outcomes (e.g., 'population_growth').
    covariate_cols : list
        Confounder columns (excluding the placebo outcomes).

    Returns
    -------
    pd.DataFrame
        Estimates for each placebo outcome.
    """
    results = []

    for outcome in placebo_outcomes:
        if outcome not in df.columns:
            logger.warning(f"Placebo outcome '{outcome}' not found in data")
            continue

        covars = [c for c in covariate_cols if c != outcome]

        try:
            estimate = estimation_func(
                df, outcome, treatment_col, covars, **estimation_kwargs,
            )
            results.append({
                "outcome": outcome,
                "estimate": estimate.estimate,
                "ci_lower": estimate.ci_lower,
                "ci_upper": estimate.ci_upper,
                "std_error": estimate.std_error,
                "significant": estimate.is_significant,
            })
        except Exception as e:
            logger.warning(f"Placebo outcome test failed for '{outcome}': {e}")

    return pd.DataFrame(results)


def leave_one_out_analysis(
    df: pd.DataFrame,
    estimation_func: Callable,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: list,
    group_col: str = "iso3",
    top_n: int = 10,
    **estimation_kwargs,
) -> pd.DataFrame:
    """
    Leave-one-out stability: re-estimate after dropping one country at a time.

    Checks if results are driven by any single influential country.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    estimation_func : callable
        Causal estimation function.
    group_col : str
        Column to iterate over (typically 'iso3').
    top_n : int
        Number of largest groups to test (by row count).

    Returns
    -------
    pd.DataFrame
        Estimates for each leave-one-out iteration.
    """
    # Identify largest groups
    group_sizes = df.groupby(group_col).size().sort_values(ascending=False)
    groups_to_test = group_sizes.head(top_n).index.tolist()

    results = []

    # Full sample estimate
    try:
        full_est = estimation_func(
            df, outcome_col, treatment_col, covariate_cols, **estimation_kwargs,
        )
        results.append({
            "dropped": "None (Full Sample)",
            "estimate": full_est.estimate,
            "ci_lower": full_est.ci_lower,
            "ci_upper": full_est.ci_upper,
            "n_obs": full_est.n_obs,
        })
    except Exception as e:
        logger.error(f"Full sample estimation failed: {e}")
        return pd.DataFrame()

    # Leave-one-out
    for group in groups_to_test:
        df_loo = df[df[group_col] != group]
        try:
            est = estimation_func(
                df_loo, outcome_col, treatment_col, covariate_cols, **estimation_kwargs,
            )
            results.append({
                "dropped": group,
                "estimate": est.estimate,
                "ci_lower": est.ci_lower,
                "ci_upper": est.ci_upper,
                "n_obs": est.n_obs,
            })
        except Exception as e:
            logger.warning(f"LOO failed for {group}: {e}")

    return pd.DataFrame(results)


def compute_e_value(
    estimate: float,
    ci_lower: float,
    ci_upper: float,
    outcome_is_log_linear: bool = True,
) -> dict:
    """
    E-value for sensitivity to unobserved confounding (VanderWeele & Ding, 2017).

    The E-value is the minimum strength of association (on the risk-ratio
    scale) that an unmeasured confounder would need to have with both the
    treatment and the outcome, above and beyond the measured covariates, to
    fully explain away the observed treatment-effect estimate. It is also
    computed for the CI bound closest to the null, which gives the minimum
    confounding strength needed to move the CI to include the null.

    Since our treatment effects are estimated on a log outcome (log_co2),
    the coefficient is already an approximate log risk-ratio, so we exponentiate
    it directly to obtain a risk-ratio-scale effect (`outcome_is_log_linear=True`).
    For estimates on a different (non-log) scale, this approximation does not
    apply and the caller must pre-convert to a risk-ratio scale.

    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect (on the log-outcome scale).
    ci_lower, ci_upper : float
        95% CI bounds of the estimate (same scale as `estimate`).
    outcome_is_log_linear : bool
        If True, treat exp(estimate) as an approximate risk ratio (valid for
        our log_co2 outcome). If False, `estimate`/`ci_lower`/`ci_upper` are
        assumed to already be on the risk-ratio scale.

    Returns
    -------
    dict
        {'rr': risk ratio, 'e_value_point': E-value for the point estimate,
         'e_value_ci': E-value for the CI bound nearest the null}
    """
    def _e_value_from_rr(rr: float) -> float:
        rr = abs(rr)
        if rr < 1:
            rr = 1 / rr
        return rr + np.sqrt(rr * (rr - 1))

    if outcome_is_log_linear:
        rr = float(np.exp(estimate))
        rr_lo = float(np.exp(ci_lower))
        rr_hi = float(np.exp(ci_upper))
    else:
        rr, rr_lo, rr_hi = estimate, ci_lower, ci_upper

    e_value_point = _e_value_from_rr(rr)

    # CI bound closest to the null (RR=1)
    if rr_lo <= 1 <= rr_hi:
        e_value_ci = 1.0  # CI already includes the null
    else:
        rr_near_null = rr_lo if rr_lo > 1 else rr_hi
        e_value_ci = _e_value_from_rr(rr_near_null)

    return {
        "rr": rr,
        "e_value_point": e_value_point,
        "e_value_ci": e_value_ci,
    }


def oster_delta(
    df: pd.DataFrame,
    estimation_func: Callable,
    outcome_col: str,
    treatment_col: str,
    restricted_covariates: list,
    full_covariates: list,
    r_max: float = 1.3,
    beta_target: float = 0.0,
    **estimation_kwargs,
) -> dict:
    """
    Oster's (2019) delta: robustness of a treatment effect to omitted-variable bias,
    inferred from how much the coefficient and R² move when moving from a
    restricted (few-covariate) to a full (richly-covariate) specification.

    delta is the relative degree of selection on unobservables (vs.
    observables) that would be needed to shrink the treatment coefficient to
    `beta_target` (default 0), given the assumed maximum R² (`r_max`) an
    infeasible "all confounders included" regression would reach. Larger
    |delta| (conventionally |delta| >= 1) indicates the observed relationship
    is unlikely to be driven entirely by omitted-variable bias.

    Uses OLS (via `estimation_func`, expected to be `run_ols_baseline` or
    equivalent returning an object with `.estimate` and `.additional_info`
    containing an R²) for both specifications, matching the existing
    Minimal/Baseline/Extended specification ladder already used in
    `alternative_specifications`.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    estimation_func : callable
        OLS estimation function; must accept `return_r2=True` in
        `estimation_kwargs` or the caller must otherwise obtain R² alongside
        the estimate (see notes in `causal_models.run_ols_baseline`).
    restricted_covariates, full_covariates : list
        Covariate sets for the restricted and full (extended) specifications.
        `full_covariates` should be a superset of `restricted_covariates`.
    r_max : float
        Assumed R² of a hypothetical regression on all (including
        unobserved) confounders. Oster (2019) recommends 1.3x the full
        model's R² as a conservative default; pass a fixed value if r_max is
        computed by the caller as 1.3 * r2_full.
    beta_target : float
        The coefficient value being tested against (0 = "no effect").

    Returns
    -------
    dict with beta_restricted, beta_full, r2_restricted, r2_full, delta.
    """
    est_restricted = estimation_func(
        df, outcome_col, treatment_col, restricted_covariates, **estimation_kwargs,
    )
    est_full = estimation_func(
        df, outcome_col, treatment_col, full_covariates, **estimation_kwargs,
    )

    beta_r = est_restricted.estimate
    beta_f = est_full.estimate
    r2_r = est_restricted.additional_info.get("r2")
    r2_f = est_full.additional_info.get("r2")

    if r2_r is None or r2_f is None:
        raise ValueError(
            "Oster's delta requires R^2 for both specifications; "
            "ensure estimation_func populates additional_info['r2']."
        )

    denom = (r_max - r2_f) * (beta_r - beta_f)
    if denom == 0:
        delta = np.nan
    else:
        delta = ((beta_f - beta_target) * (r2_f - r2_r)) / denom

    return {
        "beta_restricted": beta_r,
        "beta_full": beta_f,
        "r2_restricted": r2_r,
        "r2_full": r2_f,
        "r_max": r_max,
        "delta": float(delta) if not np.isnan(delta) else np.nan,
    }


def alternative_specifications(
    df: pd.DataFrame,
    estimation_func: Callable,
    outcome_col: str,
    treatment_col: str,
    covariate_specs: dict,
    **estimation_kwargs,
) -> pd.DataFrame:
    """
    Test alternative covariate specifications.

    Parameters
    ----------
    covariate_specs : dict
        Named covariate specifications, e.g.:
        {
            "Baseline": ["gdp_pc", "population", "energy_pc"],
            "Extended": ["gdp_pc", "population", "energy_pc", "governance", "trade"],
            "Minimal": ["gdp_pc", "population"],
        }

    Returns
    -------
    pd.DataFrame
        Estimates for each specification.
    """
    results = []

    for spec_name, covariates in covariate_specs.items():
        available = [c for c in covariates if c in df.columns]
        if len(available) < len(covariates):
            missing = set(covariates) - set(available)
            logger.warning(f"Spec '{spec_name}' missing columns: {missing}")

        try:
            est = estimation_func(
                df, outcome_col, treatment_col, available, **estimation_kwargs,
            )
            results.append({
                "specification": spec_name,
                "n_covariates": len(available),
                "estimate": est.estimate,
                "ci_lower": est.ci_lower,
                "ci_upper": est.ci_upper,
                "std_error": est.std_error,
                "n_obs": est.n_obs,
            })
        except Exception as e:
            logger.warning(f"Specification '{spec_name}' failed: {e}")

    return pd.DataFrame(results)
