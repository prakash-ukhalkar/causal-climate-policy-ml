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
    "Bahamas, The": "BHS",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Egypt, Arab Rep.": "EGY",
    "Gambia, The": "GMB",
    "Hong Kong SAR, China": "HKG",
    "Iran, Islamic Rep.": "IRN",
    "Korea, Rep.": "KOR",
    "Korea, Dem. Rep.": "PRK",
    "Kosovo": "XKX",
    "Lao PDR": "LAO",
    "Laos": "LAO",
    "Macao SAR, China": "MAC",
    "St. Kitts and Nevis": "KNA",
    "St. Lucia": "LCA",
    "St. Vincent and the Grenadines": "VCT",
    "Swaziland": "SWZ",
    "Taiwan, China": "TWN",
    "Venezuela, RB": "VEN",
    "West Bank and Gaza": "PSE",
    "Yemen, Rep.": "YEM",
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
        # Try fuzzy search; search_fuzzy itself raises LookupError (not just
        # returning an empty list) when nothing matches, so this inner
        # lookup needs its own guard rather than falling through to the
        # outer handler.
        try:
            results = pycountry.countries.search_fuzzy(name)
            if results:
                return results[0].alpha_3
        except LookupError:
            pass
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


def clean_sector_emissions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the sector-disaggregated CO2 dataset (data/raw/wdi/wdi_sector_emissions.csv).

    This is a country x year panel of CO2 emissions broken out by sector
    (power, transport, industry, agriculture), covering 1990-2023. It backs
    the manuscript's "sector-level" analysis; prior to this fix it existed on
    disk but was never loaded by any notebook, so downstream sector results
    referencing it (e.g. sector_estimates.csv) were not reproducible from the
    tracked pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sector emissions dataframe with columns
        ['iso3', 'year', 'co2_power', 'co2_transport', 'co2_industry', 'co2_agriculture'].

    Returns
    -------
    pd.DataFrame
        Cleaned sector panel, restricted to iso3/year plus sector columns,
        years >= 1990, deduplicated on (iso3, year).
    """
    df = df.copy()

    expected_sector_cols = ["co2_power", "co2_transport", "co2_industry", "co2_agriculture"]
    sector_cols = [c for c in expected_sector_cols if c in df.columns]

    df = df.dropna(subset=["iso3", "year"])
    df = df[df["year"] >= 1990]
    df = df.drop_duplicates(subset=["iso3", "year"])
    df = df[["iso3", "year"] + sector_cols].sort_values(["iso3", "year"]).reset_index(drop=True)

    logger.info(f"Cleaned sector emissions: {len(df)} rows, {df['iso3'].nunique()} countries, "
                f"years {df['year'].min()}-{df['year'].max()}, sectors: {sector_cols}")

    return df


def build_carbon_price_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the sector-fuel-level WCPD panel to a single national
    carbon-pricing treatment indicator per country-year.

    Replaces the earlier keyword-matched "Carbon Pricing" treatment variable
    (free-text matching against the Climate Policy Database's instrument
    field), which was the root cause of the project's first journal
    rejection and a contributing factor in its second. WCPD's `tax`/`ets`
    columns are explicit, structured coverage flags set by the dataset's own
    curators from primary legal/regulatory sources (Dolphin & Xiahou 2022),
    not inferred from free text, so no keyword-matching step exists here to
    get wrong.

    Aggregation rule (reported verbatim in the manuscript's data section):
    a country-year is treated (`treat_carbon_price = 1`) if ANY IPCC
    sector-fuel record for that jurisdiction-year has `tax == 1` or
    `ets == 1` (i.e. at least one emissions source is covered by an
    explicit carbon tax or emissions trading scheme that year). A
    jurisdiction's adoption year is the first year this condition holds.
    Once treated, a jurisdiction remains treated in all subsequent years
    (absorbing treatment), matching the staggered-adoption designs used
    downstream (Callaway & Sant'Anna 2021; Sun & Abraham 2021).

    Parameters
    ----------
    df : pd.DataFrame
        Raw WCPD panel as returned by `data_loader.load_wcpd_national()`
        (one row per jurisdiction-year-IPCC_sector-fuel record).

    Returns
    -------
    pd.DataFrame
        One row per (iso3, year) with columns:
        - treat_carbon_price : int, 1 if treated that year, else 0
        - adoption_year : float, this jurisdiction's first treated year
          (NaN for never-treated jurisdictions)
        - any_tax, any_ets : int, instrument-type breakdown of the same
          country-year (for transparency / robustness checks only)
    """
    df = df.copy()

    # Aggregate to jurisdiction-year BEFORE harmonizing country codes: WCPD's
    # raw panel has ~2M sector-fuel-year rows across only 201 jurisdictions,
    # and harmonize_country_codes' per-row pycountry lookup is too slow to
    # run at that row count. Collapsing to ~7k jurisdiction-year rows first
    # (same result, since harmonization only depends on the jurisdiction
    # name) makes the lookup run on 201 distinct names instead of 2M rows.
    jurisdiction_year = (
        df.groupby(["jurisdiction", "year"])
        .agg(any_tax=("tax", "max"), any_ets=("ets", "max"))
        .reset_index()
    )
    country_year = harmonize_country_codes(jurisdiction_year, "jurisdiction")
    country_year["treat_carbon_price"] = (
        (country_year["any_tax"] > 0) | (country_year["any_ets"] > 0)
    ).astype(int)

    adoption_year = (
        country_year.loc[country_year["treat_carbon_price"] == 1]
        .groupby("iso3")["year"]
        .min()
        .rename("adoption_year")
    )
    country_year = country_year.merge(adoption_year, on="iso3", how="left")

    n_treated_countries = country_year.loc[country_year["treat_carbon_price"] == 1, "iso3"].nunique()
    logger.info(
        f"WCPD treatment: {len(country_year)} country-year rows, "
        f"{country_year['iso3'].nunique()} jurisdictions, "
        f"{n_treated_countries} ever-treated by {country_year['year'].max()}"
    )

    return country_year.sort_values(["iso3", "year"]).reset_index(drop=True)


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
