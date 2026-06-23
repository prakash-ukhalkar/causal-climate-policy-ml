"""
Feature engineering for the Causal Climate Policy project.

Constructs the master analysis panel (country × year × sector),
defines treatment and outcome variables, and engineers confounders.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Policy Type Categories ───────────────────────────────────────────────────

POLICY_TYPE_MAPPING = {
    # Carbon pricing
    "Carbon tax": "Carbon Pricing",
    "Emissions trading": "Carbon Pricing",
    "carbon tax": "Carbon Pricing",
    "emissions trading": "Carbon Pricing",
    "Cap and trade": "Carbon Pricing",

    # Renewable mandates
    "Renewable energy target": "Renewable Mandate",
    "Feed-in tariff": "Renewable Mandate",
    "Renewable portfolio standard": "Renewable Mandate",
    "Renewable energy auction": "Renewable Mandate",
    "Net metering": "Renewable Mandate",

    # Efficiency standards
    "Energy efficiency target": "Efficiency Standard",
    "Building codes": "Efficiency Standard",
    "Appliance standards": "Efficiency Standard",
    "Vehicle emissions standard": "Efficiency Standard",
    "Fuel economy standard": "Efficiency Standard",

    # Subsidies & fiscal
    "Fossil fuel subsidy reform": "Fiscal Policy",
    "Green bond": "Fiscal Policy",
    "Tax incentive": "Fiscal Policy",
}

SECTOR_MAPPING = {
    "electricity-and-heat": "Power",
    "power": "Power",
    "energy": "Power",
    "transport": "Transport",
    "transportation": "Transport",
    "industry": "Industry",
    "manufacturing": "Industry",
    "industrial": "Industry",
    "agriculture": "Agriculture",
    "forestry": "Agriculture",
    "land-use": "Agriculture",
    "buildings": "Buildings",
    "residential": "Buildings",
    "commercial": "Buildings",
}


def create_treatment_variables(
    policies_df: pd.DataFrame,
    emissions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create binary and continuous treatment variables from policy data.

    For each country-year-sector combination, determines:
    - Whether a policy of each type was in effect (binary)
    - Years since policy adoption (continuous intensity)
    - Count of active policies (dosage)

    Parameters
    ----------
    policies_df : pd.DataFrame
        Cleaned policy database with columns:
        ['iso3', 'year_adopted', 'policy_category', 'sector']
    emissions_df : pd.DataFrame
        Panel data defining the country-year combinations to populate.

    Returns
    -------
    pd.DataFrame
        Treatment matrix aligned to the emissions panel.
    """
    panel = emissions_df[["iso3", "year"]].drop_duplicates()

    policy_types = ["Carbon Pricing", "Renewable Mandate", "Efficiency Standard", "Fiscal Policy"]

    for ptype in policy_types:
        col_base = ptype.lower().replace(" ", "_")
        type_policies = policies_df[policies_df["policy_category"] == ptype]

        # Binary: any policy of this type adopted on or before this year
        def _has_policy(row, policies=type_policies):
            matches = policies[
                (policies["iso3"] == row["iso3"]) &
                (policies["year_adopted"] <= row["year"])
            ]
            return 1 if len(matches) > 0 else 0

        # For efficiency, use merge-based approach instead of row-wise apply
        first_adoption = (
            type_policies
            .groupby("iso3")["year_adopted"]
            .min()
            .reset_index()
            .rename(columns={"year_adopted": f"{col_base}_first_year"})
        )

        panel = panel.merge(first_adoption, on="iso3", how="left")

        # Binary treatment
        panel[f"treat_{col_base}"] = (
            panel["year"] >= panel[f"{col_base}_first_year"]
        ).astype(int)
        panel.loc[panel[f"{col_base}_first_year"].isna(), f"treat_{col_base}"] = 0

        # Years since adoption (continuous)
        panel[f"years_since_{col_base}"] = np.where(
            panel[f"treat_{col_base}"] == 1,
            panel["year"] - panel[f"{col_base}_first_year"],
            0,
        )

        # Clean up
        panel = panel.drop(columns=[f"{col_base}_first_year"])

    logger.info(f"Treatment matrix: {len(panel)} rows, "
                f"treatments: {[c for c in panel.columns if c.startswith('treat_')]}")

    return panel


def create_outcome_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer outcome variables from raw emissions data.

    Creates:
    - log_co2: Natural log of CO₂ emissions
    - co2_intensity: CO₂ per unit GDP
    - co2_growth: Year-over-year percentage change
    - co2_per_capita: Emissions per person
    """
    df = df.copy()

    # Log emissions (handle zeros)
    df["log_co2"] = np.log(df["co2"].clip(lower=0.001))

    # Emissions intensity
    if "gdp_per_capita" in df.columns:
        # Use gdp_per_capita as the preferred robust metric
        df["co2_intensity"] = df["co2"] / df["gdp_per_capita"].replace(0, np.nan)
    elif "gdp" in df.columns:
        df["co2_intensity"] = df["co2"] / df["gdp"].replace(0, np.nan)

    # Year-over-year growth
    df = df.sort_values(["iso3", "year"])
    df["co2_growth"] = df.groupby("iso3")["co2"].pct_change() * 100
    
    # Clean up infinities created by division by zero or pct_change from zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def create_lagged_features(
    df: pd.DataFrame,
    group_col: str = "iso3",
    time_col: str = "year",
    value_cols: list = None,
    lags: list = None,
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns within panel groups.

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe.
    group_col : str
        Grouping column.
    time_col : str
        Time column.
    value_cols : list
        Columns to lag. Default: all numeric except time.
    lags : list
        Lag periods. Default: [1, 3, 5].

    Returns
    -------
    pd.DataFrame
        Dataframe with lagged columns appended.
    """
    if lags is None:
        lags = [1, 3, 5]

    if value_cols is None:
        value_cols = ["co2", "gdp", "population"]

    df = df.sort_values([group_col, time_col]).copy()

    for col in value_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby(group_col)[col].shift(lag)

    return df


def build_analysis_panel(
    emissions_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    covariates_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Merge all components into the final analysis panel.

    Parameters
    ----------
    emissions_df : pd.DataFrame
        Outcome data (CO₂ emissions by country-year).
    energy_df : pd.DataFrame
        Energy mix data.
    treatment_df : pd.DataFrame
        Treatment indicators.
    covariates_df : pd.DataFrame, optional
        Additional covariates (World Bank indicators, governance).

    Returns
    -------
    pd.DataFrame
        Master analysis panel ready for causal estimation.
    """
    # Start with emissions
    panel = emissions_df.copy()

    # Add energy mix
    energy_cols = [c for c in energy_df.columns
                   if c not in ["country"] and c in energy_df.columns]
    panel = panel.merge(
        energy_df[energy_cols],
        on=["iso3", "year"],
        how="left",
        suffixes=("", "_energy"),
    )

    # Add treatment variables
    treat_cols = ["iso3", "year"] + [c for c in treatment_df.columns
                                      if c.startswith(("treat_", "years_since_"))]
    panel = panel.merge(
        treatment_df[treat_cols],
        on=["iso3", "year"],
        how="left",
    )

    # Add additional covariates if available
    if covariates_df is not None:
        panel = panel.merge(
            covariates_df,
            on=["iso3", "year"],
            how="left",
            suffixes=("", "_wb"),
        )

    # Fill treatment NAs with 0 (no policy)
    treat_cols_final = [c for c in panel.columns if c.startswith(("treat_", "years_since_"))]
    panel[treat_cols_final] = panel[treat_cols_final].fillna(0)

    logger.info(f"Analysis panel: {panel.shape[0]} rows × {panel.shape[1]} columns, "
                f"{panel['iso3'].nunique()} countries, "
                f"years {panel['year'].min()}-{panel['year'].max()}")

    return panel
