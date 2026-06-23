# Causal ML for Climate Policy Effectiveness

[![License](https://img.shields.io/github/license/prakash-ukhalkar/causal-climate-policy-ml?style=flat-square&color=blue)](https://github.com/prakash-ukhalkar/causal-climate-policy-ml/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Causal ML](https://img.shields.io/badge/Methodology-Causal%20ML-orange?style=flat-square)](https://github.com/microsoft/EconML)


**Quantifying the Heterogeneous Effects of National Climate Policies on Sectoral CO₂ Emissions Using Causal Machine Learning**

> **Target Journal:** **Nexus** (ScienceDirect, eISSN: 2950-1601)  
> **Methodology:** Double/Debiased ML, Causal Forests, Panel Econometrics  
> **Status:** In Development  
> **Author:** Prakash Ukhalkar ([@prakash-ukhalkar](https://github.com/prakash-ukhalkar))  
> **Repository:** [causal-climate-policy-ml](https://github.com/prakash-ukhalkar/causal-climate-policy-ml)

---

## Table of Contents

1. [Research Overview](#research-overview)
2. [Core Research Question](#core-research-question)
3. [Project Structure](#project-structure)
4. [Notebook Pipeline](#notebook-pipeline)
5. [Datasets](#datasets)
6. [Methodology](#methodology)
7. [Environment Setup](#environment-setup)
8. [Development Workflow](#development-workflow)
9. [Expected Outputs](#expected-outputs)
10. [Citation & License](#citation--license)

---

## Research Overview

This project applies **causal machine learning** methods to estimate the heterogeneous treatment effects of national climate policies (carbon taxes, renewable energy mandates, efficiency standards) on sector-level CO₂ emissions. Unlike traditional econometric approaches (difference-in-differences, fixed effects), causal ML methods can:

- Handle **high-dimensional confounders** without manual variable selection
- Estimate **heterogeneous treatment effects** — how policy impacts vary by country income, governance quality, and sector
- Provide **non-parametric, flexible** first-stage models that capture non-linear relationships

This work fills a critical gap: while the causal ML literature has exploded (2021–2026), it has barely been applied to sector-level climate policy evaluation at a global scale.

---

## Core Research Question

> *Can causal machine learning methods (Double/Debiased ML, Causal Forests) estimate the heterogeneous treatment effects of specific national climate policies on sector-level CO₂ emissions, controlling for confounding economic and demographic trends?*

### Sub-Questions

1. **Average Treatment Effect (ATE):** What is the average causal effect of adopting a carbon pricing mechanism on power-sector emissions across countries?
2. **Heterogeneity (CATE):** How does this effect vary by GDP per capita, governance quality, and pre-existing renewable energy share?
3. **Policy Comparison:** Which policy instruments (carbon taxes vs. renewable mandates vs. efficiency standards) yield the largest emission reductions per sector?
4. **Temporal Dynamics:** How do policy effects evolve over time (immediate vs. lagged impacts)?
5. **Robustness:** Do causal estimates survive placebo tests, sensitivity analyses, and alternative specifications?

---

## Project Structure

```
causal-climate-policy/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory (gitignored, downloaded locally)
│   ├── raw/                           # Original downloaded datasets
│   │   ├── climate_policy_db/         # Climate Policy Database exports
│   │   ├── climate_trace/             # Climate TRACE sector emissions
│   │   ├── owid/                      # Our World in Data CO₂ + Energy
│   │   ├── world_bank/                # World Bank Development Indicators
│   │   └── oecd/                      # OECD effective carbon rates
│   ├── interim/                       # Intermediate processed data
│   │   ├── policies_cleaned.csv
│   │   ├── emissions_panel.csv
│   │   └── covariates_merged.csv
│   └── processed/                     # Final analysis-ready datasets
│       ├── analysis_panel.csv         # Master panel: country × year × sector
│       └── treatment_matrix.csv       # Binary/continuous treatment indicators
│
├── notebooks/                         # Jupyter notebooks (numbered pipeline)
│   ├── 01_data_acquisition.ipynb      # Download & organize raw data
│   ├── 02_data_cleaning.ipynb         # Clean, validate, harmonize
│   ├── 03_eda.ipynb                   # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb   # Panel construction & treatment coding
│   ├── 05_causal_analysis.ipynb       # DML, Causal Forest, ATEs & CATEs
│   ├── 06_robustness_checks.ipynb     # Placebo tests, sensitivity analysis
│   ├── 07_results_visualization.ipynb # Publication-quality figures & tables
│   └── 08_paper_tables.ipynb         # Formatted regression/results tables
│
├── src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset download & caching utilities
│   ├── preprocessing.py               # Cleaning & harmonization functions
│   ├── feature_engineering.py         # Panel construction, treatment coding
│   ├── causal_models.py               # DML, Causal Forest wrappers
│   ├── robustness.py                  # Placebo tests, sensitivity analysis
│   └── visualization.py              # Publication-quality plotting functions
│
├── outputs/                           # Generated outputs
│   ├── figures/                       # Publication-quality figures (PNG/PDF)
│   ├── tables/                        # LaTeX/CSV formatted tables
│   └── models/                        # Saved model artifacts
│
└── paper/                             # Paper drafting resources
    ├── outline.md                     # Paper structure outline
    ├── references.bib                 # BibTeX references
    └── supplementary/                 # Supplementary materials
```

---

## Notebook Pipeline

Each notebook is self-contained but follows a strict sequential pipeline. Below is the detailed scope of each notebook.

### `01_data_acquisition.ipynb` — Data Download & Organization

**Purpose:** Programmatically download all required datasets and organize them into `data/raw/`.

| Step | Description |
|------|-------------|
| 1.1 | Download Our World in Data CO₂ dataset from GitHub (CSV, direct URL) |
| 1.2 | Download Our World in Data Energy dataset from GitHub |
| 1.3 | Download Climate TRACE country-level emissions by sector |
| 1.4 | Download Climate Policy Database export (CSV) |
| 1.5 | Download World Bank Development Indicators (API or bulk CSV) |
| 1.6 | Download/document OECD effective carbon rate data |
| 1.7 | Validate all downloads — file existence, row counts, date ranges |
| 1.8 | Generate a data manifest (JSON) logging sources, dates, checksums |

**Output:** Populated `data/raw/` directory + `data/raw/data_manifest.json`

---

### `02_data_cleaning.ipynb` — Cleaning & Harmonization

**Purpose:** Clean each dataset individually, then harmonize country names, ISO codes, date formats, and units.

| Step | Description |
|------|-------------|
| 2.1 | Clean OWID CO₂ data — handle missing values, validate country codes |
| 2.2 | Clean Climate TRACE — standardize sector names, resolve unit inconsistencies |
| 2.3 | Clean Climate Policy Database — parse policy types, adoption years, sectors |
| 2.4 | Clean World Bank indicators — interpolate sparse series, validate ranges |
| 2.5 | Harmonize country identifiers across all datasets (ISO 3166-1 alpha-3) |
| 2.6 | Standardize time dimension (annual, calendar year) |
| 2.7 | Document cleaning decisions, dropped records, and imputation choices |
| 2.8 | Export cleaned datasets to `data/interim/` |

**Output:** `data/interim/policies_cleaned.csv`, `emissions_panel.csv`, `covariates_merged.csv`

---

### `03_eda.ipynb` — Exploratory Data Analysis

**Purpose:** Understand distributions, trends, correlations, and data quality before modeling.

| Step | Description |
|------|-------------|
| 3.1 | Global emission trends by sector (time-series plots) |
| 3.2 | Policy adoption timeline — cumulative policy count by type and region |
| 3.3 | Geographic distribution of policies (choropleth maps) |
| 3.4 | Correlation matrix: emissions × covariates (GDP, energy mix, governance) |
| 3.5 | Treatment vs. control group balance diagnostics |
| 3.6 | Missing data patterns and coverage analysis |
| 3.7 | Distributional analysis of key variables (histograms, box plots, QQ plots) |
| 3.8 | Key findings summary — what patterns suggest about policy effects |

**Output:** Exploratory figures in `outputs/figures/eda/`, summary statistics tables

---

### `04_feature_engineering.ipynb` — Panel Construction & Treatment Coding

**Purpose:** Build the master analysis panel and encode treatment variables.

| Step | Description |
|------|-------------|
| 4.1 | Construct country × year × sector panel (long format) |
| 4.2 | Define treatment variables — binary (policy adopted: 0/1) and continuous (carbon price level, years since adoption) |
| 4.3 | Define outcome variable — log emissions, emissions intensity (CO₂/GDP), year-over-year change |
| 4.4 | Engineer confounders/controls — GDP per capita, population, urbanization, energy mix, governance indices, trade openness |
| 4.5 | Create lagged variables — 1-year, 3-year, 5-year lags for dynamic effects |
| 4.6 | Handle panel structure — country and year fixed effects indicators |
| 4.7 | Propensity score diagnostics — check overlap between treated/control |
| 4.8 | Export final analysis panel to `data/processed/analysis_panel.csv` |

**Output:** `data/processed/analysis_panel.csv`, `treatment_matrix.csv`, panel diagnostics

---

### `05_causal_analysis.ipynb` — Core Causal ML Estimation

**Purpose:** Estimate average and heterogeneous treatment effects using causal ML.

| Step | Description |
|------|-------------|
| 5.1 | **Baseline OLS/FE:** Traditional two-way fixed effects regression (country + year FE) |
| 5.2 | **Double/Debiased ML (DML):** Estimate ATE using `econml.dml.LinearDML` and `NonParamDML` with XGBoost/Random Forest first stages |
| 5.3 | **Causal Forest:** Estimate CATEs using `econml.grf.CausalForest` — how effects vary by country characteristics |
| 5.4 | **DML with multiple treatments:** Separate estimates for carbon taxes, renewable mandates, efficiency standards |
| 5.5 | **Temporal heterogeneity:** Estimate effects by policy age (1yr, 3yr, 5yr post-adoption) |
| 5.6 | **Sector-specific analysis:** Run separate models for power, transport, industry, agriculture |
| 5.7 | Collect all estimates with confidence intervals |
| 5.8 | Compare DML vs. OLS vs. Causal Forest estimates — method sensitivity |

**Output:** Estimated ATEs, CATEs, confidence intervals, model comparison tables

---

### `06_robustness_checks.ipynb` — Validation & Sensitivity Analysis

**Purpose:** Ensure causal estimates are credible and not artifacts of modeling choices.

| Step | Description |
|------|-------------|
| 6.1 | **Placebo test (time):** Estimate treatment effects in pre-policy periods (should be ≈ 0) |
| 6.2 | **Placebo test (outcome):** Test policy effect on unrelated outcomes (e.g., population growth) |
| 6.3 | **Leave-one-out:** Sequentially drop largest countries, check estimate stability |
| 6.4 | **Alternative treatment coding:** Binary vs. continuous, different adoption year thresholds |
| 6.5 | **Alternative ML first stages:** Swap XGBoost for Random Forest, Lasso, Ridge in DML |
| 6.6 | **Sensitivity to unobserved confounding:** Rosenbaum bounds / partial R² approach |
| 6.7 | **Sample restrictions:** OECD-only, non-OECD-only, post-2000 only |
| 6.8 | Summary table of all robustness checks |

**Output:** Robustness tables, placebo plots, sensitivity analysis figures

---

### `07_results_visualization.ipynb` — Publication-Quality Figures

**Purpose:** Generate all figures needed for the paper manuscript.

| Step | Description |
|------|-------------|
| 7.1 | **Figure 1:** Global policy adoption map (choropleth) |
| 7.2 | **Figure 2:** Average treatment effects by policy type (forest plot with CIs) |
| 7.3 | **Figure 3:** CATE heterogeneity — effect × GDP per capita (scatter + LOWESS) |
| 7.4 | **Figure 4:** CATE heterogeneity — effect × governance quality |
| 7.5 | **Figure 5:** Sector-specific treatment effects (grouped bar chart with CIs) |
| 7.6 | **Figure 6:** Temporal dynamics — effect by years since policy adoption |
| 7.7 | **Figure 7:** Placebo test results (coefficient plot) |
| 7.8 | **Figure 8:** SHAP summary of confounders in first-stage models |
| 7.9 | Apply consistent journal-quality styling (fonts, colors, resolution) |

**Output:** All figures saved as PDF + PNG (300 DPI) in `outputs/figures/`

---

### `08_paper_tables.ipynb` — Formatted Results Tables

**Purpose:** Generate LaTeX-formatted tables for direct inclusion in the manuscript.

| Step | Description |
|------|-------------|
| 8.1 | **Table 1:** Summary statistics (outcome, treatment, covariates) |
| 8.2 | **Table 2:** Policy typology & coverage (count by type × sector × region) |
| 8.3 | **Table 3:** Main results — ATE estimates (OLS, DML, Causal Forest) |
| 8.4 | **Table 4:** Heterogeneous effects (CATEs by subgroup) |
| 8.5 | **Table 5:** Sector-specific estimates |
| 8.6 | **Table 6:** Robustness checks summary |
| 8.7 | **Table 7:** Model comparison & diagnostics |
| 8.8 | Export all tables as LaTeX (.tex) and CSV |

**Output:** Formatted tables in `outputs/tables/`

---

## Datasets

| Dataset | Source | URL | Key Variables |
|---------|--------|-----|---------------|
| **Climate Policy Database** | NewClimate Institute | [climatepolicydatabase.org](https://www.climatepolicydatabase.org/) | Policy type, adoption year, sector, country, stringency |
| **Climate TRACE** | Climate TRACE Coalition | [climatetrace.org/downloads](https://climatetrace.org/downloads) | Sector-specific CO₂ emissions by country (2015–2024) |
| **Our World in Data — CO₂** | OWID / Global Carbon Project | [github.com/owid/co2-data](https://github.com/owid/co2-data) | Historical emissions, per capita, by source |
| **Our World in Data — Energy** | OWID | [github.com/owid/energy-data](https://github.com/owid/energy-data) | Energy mix, renewables share, consumption |
| **World Bank WDI** | World Bank | [databank.worldbank.org](https://databank.worldbank.org/) | GDP, population, urbanization, trade |
| **WGI Governance** | World Bank | [info.worldbank.org/governance/wgi](https://info.worldbank.org/governance/wgi/) | Regulatory quality, rule of law, gov. effectiveness |

---

## Methodology

### Causal Framework

```
                    ┌─────────────────────────┐
                    │    Confounders (X)       │
                    │  GDP, population, energy │
                    │  mix, governance, trade  │
                    └─────┬──────────┬────────┘
                          │          │
                          ▼          ▼
               ┌──────────────┐  ┌──────────────────┐
               │ Treatment (T) │  │   Outcome (Y)     │
               │ Climate policy│──│ Sectoral CO₂      │
               │ adoption      │  │ emissions          │
               └──────────────┘  └──────────────────┘
```

### Double/Debiased ML (DML)

The DML procedure (Chernozhukov et al., 2018) proceeds in two stages:

1. **First Stage:** Use flexible ML models (XGBoost) to predict both:
   - Treatment from confounders: `T̂ = g(X)`
   - Outcome from confounders: `Ŷ = f(X)`

2. **Second Stage:** Regress the residualized outcome on the residualized treatment:
   - `(Y - Ŷ) = θ · (T - T̂) + ε`
   - The coefficient `θ` is the **debiased causal estimate**

3. **Cross-fitting:** Use K-fold sample splitting to avoid overfitting bias.

### Causal Forest (Generalized Random Forest)

- Extends random forests to estimate **conditional average treatment effects (CATEs)**
- Each leaf of the forest targets a local treatment effect estimate
- Enables discovery of **which subpopulations** benefit most from policies

---

## Environment Setup

```bash
# 1. Navigate to project directory
cd f:\Nexus_Projects\causal-climate-policy

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Register Jupyter kernel
python -m ipykernel install --user --name=causal-climate --display-name="Causal Climate Policy"

# 6. Launch Jupyter
jupyter lab
```

---

## Development Workflow

### Phase 1: Data Foundation (Notebooks 01–02)
Build the data pipeline — download, clean, and harmonize all datasets into a unified panel.

### Phase 2: Exploration (Notebook 03)
Understand the data through visualization and statistical summaries. Identify potential issues before modeling.

### Phase 3: Analysis Setup (Notebook 04)
Construct the formal econometric panel, define treatments and outcomes, and verify identifying assumptions.

### Phase 4: Core Analysis (Notebooks 05–06)
Run causal ML estimation and validate with robustness checks. This is the intellectual core of the paper.

### Phase 5: Communication (Notebooks 07–08)
Transform results into publication-quality figures and tables ready for the manuscript.

---

## Expected Outputs

### Key Findings (Anticipated)
- **Carbon pricing** reduces power-sector emissions by ~8–15% (causal estimate with CI)
- **Renewable mandates** show strongest effects in countries with GDP > $15K/capita
- **Efficiency standards** have delayed effects (significant only after 3+ years)
- **Agricultural emissions** are largely unresponsive to current policy instruments
- **Governance quality** is a significant moderator of policy effectiveness

### Deliverables
- [ ] Clean, reproducible Jupyter notebook pipeline (8 notebooks)
- [ ] Publication-quality figures (8+ figures)
- [ ] Formatted results tables (7+ tables)
- [ ] Manuscript-ready methodology description
- [ ] Supplementary materials with full robustness checks

---

## Citation & License

### Suggested Citation (Draft)
```
Ukhalkar, P. (2026). Quantifying the Heterogeneous Effects of National Climate
Policies on Sectoral CO₂ Emissions Using Causal Machine Learning. Nexus.
URL: https://github.com/prakash-ukhalkar/causal-climate-policy-ml
```

### Key References
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Athey, S., & Imbens, G. W. (2019). Machine learning methods that economists should know about. *Annual Review of Economics*, 11, 685–725.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, 47(2), 1148–1178.
- Eskander, S. M., & Fankhauser, S. (2020). Reduction in greenhouse gas emissions from national climate legislation. *Nature Climate Change*, 10(8), 750–756.

### License
This project is for academic research purposes. Datasets are used under their respective licenses (see `data/raw/data_manifest.json`).
