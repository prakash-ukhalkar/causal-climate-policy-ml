"""
Callaway & Sant'Anna (2021) doubly-robust staggered-adoption DiD.

Primary identification strategy for the rebuilt analysis, replacing the
OLS-vs-DML-vs-Causal-Forest "comparable estimators" framing that caused the
project's second journal rejection (GroupKFold cross-fitting combined with
one-hot country-dummy nuisance controls meant DML/Causal Forest learned no
country-specific control at all for held-out countries, contradicting the
claimed comparability to OLS with country fixed effects). A single,
well-identified staggered-adoption estimator has no such mismatch to defend.

Implemented via the `differences` package (doubly-robust ATT(g,t) per
Callaway & Sant'Anna 2021), feasibility-checked against synthetic data with
a known injected effect before use on real data (pre-treatment event-time
estimates recovered as small/insignificant, post-treatment estimates
recovered as large/significant/growing, matching the injected effect).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from differences import ATTgt

logger = logging.getLogger(__name__)


@dataclass
class DiDResult:
    """Container for a Callaway & Sant'Anna ATT(g,t) estimation result."""
    overall_att: float
    overall_se: float
    overall_ci_lower: float
    overall_ci_upper: float
    overall_p_value: float
    att_gt: pd.DataFrame
    event_study: pd.DataFrame
    n_obs: int
    n_treated_units: int
    n_control_units: int
    additional_info: dict = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        return not (self.overall_ci_lower <= 0 <= self.overall_ci_upper)


def run_callaway_santanna(
    df: pd.DataFrame,
    outcome_col: str,
    cohort_col: str = "adoption_year",
    unit_col: str = "iso3",
    time_col: str = "year",
    covariate_cols: list = None,
    control_group: str = "never_treated",
    cluster_var: str = None,
    alpha: float = 0.05,
) -> DiDResult:
    """
    Run Callaway & Sant'Anna (2021) doubly-robust ATT(g,t) estimation.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis panel, one row per (unit_col, time_col).
    outcome_col : str
        Outcome variable (e.g. 'log_co2').
    cohort_col : str
        Column giving each unit's first-treated period (NaN for
        never-treated units). Matches this project's `adoption_year`
        convention from `src.preprocessing.build_carbon_price_treatment`.
    unit_col, time_col : str
        Panel identifiers.
    covariate_cols : list, optional
        Time-varying covariates for the doubly-robust outcome-regression/
        propensity-score components (passed as the formula's RHS).
    control_group : str
        'never_treated' (default) or 'not_yet_treated'.
    cluster_var : str, optional
        Column to cluster standard errors on. Defaults to `unit_col` if
        not given, since country-years are not independent within a
        country (the same concern that motivated GroupKFold in the old
        DML pipeline, handled here at the inference stage instead).
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    DiDResult
    """
    cluster_var = cluster_var or unit_col

    data = df.dropna(subset=[outcome_col] + (covariate_cols or [])).copy()
    panel = data.set_index([unit_col, time_col])

    formula = f"{outcome_col} ~ " + (" + ".join(covariate_cols) if covariate_cols else "1")

    att_gt_model = ATTgt(data=panel, cohort_column=cohort_col)
    fitted = att_gt_model.fit(formula=formula, control_group=control_group)

    att_gt_df = fitted.to_pandas()

    overall = fitted.aggregate("simple", cluster_var=cluster_var, alpha=alpha)
    event_study = fitted.aggregate("event", cluster_var=cluster_var, alpha=alpha)

    # `differences` returns a 3-level MultiIndex column frame, e.g.
    # ('SimpleAggregation', '', 'ATT') and ('SimpleAggregation', 'analytic',
    # 'std_error') -- locate the ATT/std_error columns by their last level
    # rather than hardcoding the top-level aggregation-type name, since it
    # differs between 'simple' and 'event' aggregations.
    overall_row = overall.iloc[0]
    att_key = next(c for c in overall.columns if c[-1] == "ATT")
    se_key = next(c for c in overall.columns if c[-1] == "std_error")
    overall_att = float(overall_row[att_key])
    overall_se = float(overall_row[se_key])

    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    ci_lower = overall_att - z * overall_se
    ci_upper = overall_att + z * overall_se
    p_value = float(2 * norm.sf(abs(overall_att / overall_se))) if overall_se > 0 else np.nan

    n_treated_units = int(data.loc[data[cohort_col].notna(), unit_col].nunique())
    n_control_units = int(data.loc[data[cohort_col].isna(), unit_col].nunique())

    logger.info(
        f"Callaway-Sant'Anna: overall ATT={overall_att:.4f} (SE={overall_se:.4f}, p={p_value:.4f}), "
        f"{n_treated_units} treated units, {n_control_units} control units, N={len(data)}"
    )

    return DiDResult(
        overall_att=overall_att,
        overall_se=overall_se,
        overall_ci_lower=ci_lower,
        overall_ci_upper=ci_upper,
        overall_p_value=p_value,
        att_gt=att_gt_df,
        event_study=event_study,
        n_obs=len(data),
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        additional_info={"control_group": control_group, "cluster_var": cluster_var, "formula": formula},
    )
