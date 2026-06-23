"""
Data preprocessing and harmonization for the Causal Climate Policy project.

Cleans individual datasets, harmonizes country identifiers (ISO 3166-1 alpha-3),
standardizes time dimensions, and handles missing values.
"""

import logging

import numpy as np
import pandas as pd
import pycountry

logger = logging.getLogger(__name__)


# ── Country Name Harmonization ───────────────────────────────────────────────

# Manual overrides for country names that pycountry can't resolve
COUNTRY_NAME_OVERRIDES = {
    "Bolivia": "BOL",
    "Bonaire Sint Eustatius and Saba": "BES",
    "Brunei": "BRN",
    "Cape Verde": "CPV",
    "Congo": "COG",
    "Cote d'Ivoire": "CIV",
    "Czechia": "CZE",
    "Czech Republic": "CZE",
    "Democratic Republic of Congo": "COD",
    "East Timor": "TLS",
    "Eswatini": "SWZ",
    "Falkland Islands": "FLK",
    "Hong Kong": "HKG",
    "Iran": "IRN",
    "Ivory Coast": "CIV",
    "Kosovo": "XKX",
    "Laos": "LAO",
    "Macao": "MAC",
    "Micronesia (country)": "FSM",
    "Moldova": "MDA",
    "Netherlands Antilles": "ANT",
    "North Korea": "PRK",
    "North Macedonia": "MKD",
    "Palestine": "PSE",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "Turkey": "TUR",
    "Turkiye": "TUR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Wallis and Futuna": "WLF",
}

# Aggregates and non-country entities to exclude
NON_COUNTRY_ENTITIES = {
    "World", "Africa", "Asia", "Europe", "European Union (27)",
    "European Union (28)", "High-income countries", "Low-income countries",
    "Lower-middle-income countries", "Upper-middle-income countries",
    "North America", "Oceania", "South America", "Asia (excl. China and India)",
    "Europe (excl. EU-27)", "Europe (excl. EU-28)",
    "International transport", "Kuwaiti Oil Fires",
}


def resolve_country_iso3(name: str) -> str | None:
    """
    Resolve a country name to ISO 3166-1 alpha-3 code.

    Parameters
    ----------
    name : str
        Country name (various formats accepted).

    Returns
    -------
    str or None
        ISO alpha-3 code, or None if unresolvable.
    """
    if name in NON_COUNTRY_ENTITIES:
        return None

    # Check manual overrides first
    if name in COUNTRY_NAME_OVERRIDES:
        return COUNTRY_NAME_OVERRIDES[name]

    # Try pycountry lookup
    try:
        country = pycountry.countries.lookup(name)
        return country.alpha_3
    except LookupError:
        # Try fuzzy search
        results = pycountry.countries.search_fuzzy(name)
        if results:
            return results[0].alpha_3
    except Exception:
        pass

    logger.warning(f"Could not resolve country: '{name}'")
    return None


def harmonize_country_codes(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """
    Add standardized ISO-3 codes and filter out non-country entities.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a country name column.
    country_col : str
        Name of the column containing country names.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'iso3' column added, non-countries removed.
    """
    df = df.copy()

    # If column already contains valid ISO-3 codes, use directly
    if df[country_col].str.len().mode().iloc[0] == 3 and df[country_col].str.isupper().all():
        df["iso3"] = df[country_col]
    else:
        df["iso3"] = df[country_col].map(resolve_country_iso3)

    n_before = len(df)
    df = df.dropna(subset=["iso3"])
    n_after = len(df)

    if n_before != n_after:
        logger.info(f"Dropped {n_before - n_after} rows with unresolvable countries")

    return df


def clean_owid_co2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the OWID CO₂ dataset.

    Steps:
    - Keep only country-level rows (drop aggregates)
    - Harmonize ISO codes
    - Select relevant columns
    - Handle missing values
    """
    df = df.copy()

    # Filter out non-country entities
    df = df[~df["country"].isin(NON_COUNTRY_ENTITIES)]

    # Harmonize country codes
    if "iso_code" in df.columns:
        df["iso3"] = df["iso_code"]
        df = df.dropna(subset=["iso3"])
        df = df[df["iso3"].str.len() == 3]
    else:
        df = harmonize_country_codes(df, "country")

    # Select key columns
    key_cols = [
        "iso3", "country", "year",
        "co2", "co2_per_capita", "co2_per_gdp",
        "coal_co2", "oil_co2", "gas_co2", "cement_co2",
        "co2_growth_prct", "co2_growth_abs",
        "share_global_co2",
        "population", "gdp",
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    df = df[available_cols]

    # Basic validation
    df = df[df["year"] >= 1990]  # Focus on modern era
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    logger.info(f"Cleaned OWID CO₂: {len(df)} rows, {df['iso3'].nunique()} countries, "
                f"years {df['year'].min()}-{df['year'].max()}")

    return df


def clean_owid_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the OWID Energy dataset.

    Extracts energy mix variables relevant for confounders:
    - Renewables share, fossil share
    - Per capita energy use
    - Electricity generation mix
    """
    df = df.copy()

    # Filter non-countries
    df = df[~df["country"].isin(NON_COUNTRY_ENTITIES)]

    if "iso_code" in df.columns:
        df["iso3"] = df["iso_code"]
        df = df.dropna(subset=["iso3"])
        df = df[df["iso3"].str.len() == 3]
    else:
        df = harmonize_country_codes(df, "country")

    # Select key energy variables
    key_cols = [
        "iso3", "country", "year",
        "primary_energy_consumption",
        "renewables_share_energy", "fossil_share_energy",
        "renewables_share_elec", "fossil_share_elec",
        "solar_share_elec", "wind_share_elec", "hydro_share_elec",
        "nuclear_share_elec",
        "energy_per_capita", "energy_per_gdp",
        "electricity_generation",
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    df = df[available_cols]

    df = df[df["year"] >= 1990]
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    logger.info(f"Cleaned OWID Energy: {len(df)} rows, {df['iso3'].nunique()} countries")

    return df


def interpolate_panel_gaps(
    df: pd.DataFrame,
    group_col: str = "iso3",
    time_col: str = "year",
    value_cols: list = None,
    method: str = "linear",
    max_gap: int = 3,
) -> pd.DataFrame:
    """
    Interpolate missing values within panel groups, limited by max_gap.

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataframe.
    group_col : str
        Column defining panel groups (e.g., 'iso3').
    time_col : str
        Time column (e.g., 'year').
    value_cols : list
        Columns to interpolate. If None, interpolates all numeric columns.
    method : str
        Interpolation method ('linear', 'ffill', etc.).
    max_gap : int
        Maximum consecutive NaNs to fill.

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated values.
    """
    df = df.copy()

    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in value_cols if c != time_col]

    def _interpolate_group(group):
        for col in value_cols:
            if col in group.columns:
                group[col] = group[col].interpolate(
                    method=method, limit=max_gap
                )
        return group

    df = df.sort_values([group_col, time_col])
    df = df.groupby(group_col, group_keys=False).apply(_interpolate_group)

    return df.reset_index(drop=True)
