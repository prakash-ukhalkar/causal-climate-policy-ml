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
    policy_first_year_col: str = None,
    placebo_shifts: list = None,
    **estimation_kwargs,
) -> pd.DataFrame:
    """
    Temporal placebo test: artificially shift policy adoption dates backward.

    If the causal effect is real, pseudo-treatments in pre-policy periods
    should yield estimates close to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel.
    estimation_func : callable
        Causal estimation function (e.g., run_dml).
    treatment_col : str
        Actual treatment column.
    outcome_col : str
        Outcome column.
    covariate_cols : list
        Confounder columns.
    placebo_shifts : list
        Years to shift backward (e.g., [-5, -3, -1]).

    Returns
    -------
    pd.DataFrame
        Placebo estimates for each shift.
    """
    if placebo_shifts is None:
        placebo_shifts = [-5, -4, -3, -2, -1, 0]

    results = []

    for shift in placebo_shifts:
        df_placebo = df.copy()

        if shift != 0:
            # Create placebo treatment: pretend policy was adopted `shift` years earlier
            # This requires access to the actual adoption year
            # For binary treatment, we shift the treatment indicator
            label = f"Placebo (t{shift:+d})"
            df_placebo[treatment_col] = pd.to_numeric(df_placebo[treatment_col].shift(-shift), errors='coerce')
            df_placebo = df_placebo.dropna(subset=[treatment_col])
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
