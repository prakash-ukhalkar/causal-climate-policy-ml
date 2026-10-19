# Carbon Pricing and CO2 Emissions: A Sensitivity-Analysis Case Study

[![Author](https://img.shields.io/badge/Author-Prakash%20Ukhalkar-blue?style=flat-square&logo=github)](https://github.com/prakash-ukhalkar)
[![Repository](https://img.shields.io/badge/Repository-causal--climate--policy--ml-blueviolet?style=flat-square&logo=github)](https://github.com/prakash-ukhalkar/causal-climate-policy-ml)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Methodology](https://img.shields.io/badge/Methodology-Causal%20Inference-brightgreen?style=flat-square)](https://en.wikipedia.org/wiki/Causal_inference)

**A worked demonstration of why cross-estimator agreement and informal pre-trend checks are not sufficient evidence that a staggered-adoption difference-in-differences design is valid — using carbon pricing's estimated effect on CO2 emissions as the case study.**

---

## Table of Contents

1. [Overview](#overview)
2. [Core Question](#core-question)
3. [Key Finding](#key-finding)
4. [Project Structure](#project-structure)
5. [Notebook Pipeline](#notebook-pipeline)
6. [Data Sources](#data-sources)
7. [Methodology](#methodology)
8. [Environment Setup](#environment-setup)
9. [Reproducing the Results](#reproducing-the-results)
10. [License](#license)

---

## Overview

This project evaluates the causal effect of national carbon pricing on CO2 emissions using a 1990–2022 panel of 193 countries. Treatment is built from the World Carbon Pricing Database (Dolphin & Xiahou, 2022) — a peer-reviewed, structured record of carbon tax and emissions-trading-scheme coverage — rather than free-text policy-database keyword matching.

Two independent, modern staggered-adoption estimators (Callaway & Sant'Anna's doubly-robust ATT(g,t), and a Sun & Abraham interaction-weighted event study) agree on a significant, growing negative effect of carbon pricing on emissions. That agreement is not, on its own, credible evidence of a real causal effect. The project's actual contribution is showing why, and providing a reusable diagnostic workflow for catching this failure mode in similar staggered-adoption evaluations.

## Core Question

Does carbon pricing's apparent effect on CO2 emissions survive formal scrutiny of the parallel-trends assumption it depends on — or is a seemingly well-identified result from two agreeing estimators still consistent with a violated identifying assumption?

## Key Finding

- A naive specification finds a significant negative average treatment effect (-0.289, 95% CI [-0.497, -0.081]), corroborated by an independent event study showing a growing post-adoption reduction.
- The same event study's pre-treatment leads jointly, decisively violate the parallel-trends assumption both designs require — a violation that is not explained by cohort composition or richer covariates.
- A formal relative-magnitude sensitivity bound (Rambachan & Roth, 2023) shows the post-treatment estimate is not robust to even a modest continuation of this violation.
- An independent check — unit-specific linear time trends — corroborates this directly: absorbing the pre-trend also absorbs the effect.
- A mechanism check rules out income convergence as the explanation, and the same failure mode replicates on a materially different treatment definition (carbon tax adoption alone) drawn from the same data.

## Project Structure

```
causal-climate-policy/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data directory (gitignored, downloaded locally)
│   ├── raw/                           # Original downloaded datasets (OWID, WDI, WCPD, WGI)
│   ├── interim/                       # Intermediate cleaned data
│   └── processed/                     # Final analysis-ready panel + treatment matrix
│
├── notebooks/                         # Jupyter notebooks (numbered pipeline)
│   ├── 01_data_acquisition.ipynb      # Download & organize raw data
│   ├── 02_data_cleaning.ipynb         # Clean, validate, build WCPD treatment
│   ├── 03_eda.ipynb                   # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb   # Panel construction
│   ├── 05_causal_analysis.ipynb       # Callaway-Sant'Anna + Sun-Abraham naive results
│   ├── 05b_event_study.ipynb          # Event-study detail
│   ├── 06_robustness_checks.ipynb     # Rambachan-Roth bounds, unit-specific trends
│   ├── 07_results_visualization.ipynb # Figures
│   ├── 08_table_export.ipynb          # Formatted results tables
│   └── 09_generalization_checks.ipynb # Mechanism check + carbon-tax-only replication
│
├── src/                                # Reusable Python modules
│   ├── data_loader.py                  # Dataset download & caching utilities
│   ├── preprocessing.py                # Cleaning, harmonization, treatment construction
│   ├── feature_engineering.py          # Panel construction
│   ├── causal_models.py                # Baseline OLS estimator
│   ├── did_cs.py                       # Callaway-Sant'Anna ATT(g,t) wrapper
│   ├── event_study.py                  # Sun-Abraham event study + sensitivity-bound inputs
│   ├── robustness.py                   # Placebo tests, leave-one-out, specification checks
│   └── visualization.py                # Plotting utilities
│
└── outputs/
    ├── figures/                        # Generated figures (PNG/PDF)
    └── tables/                         # Generated tables (CSV)
```

## Notebook Pipeline

| Notebook | Purpose |
|---|---|
| `01_data_acquisition` | Download OWID CO2/Energy, World Bank WDI/WGI, and the World Carbon Pricing Database |
| `02_data_cleaning` | Clean each source; build the carbon-pricing treatment indicator from WCPD's structured tax/ETS coverage flags, validated against known early adopters (Finland 1990, Sweden & Norway 1991) |
| `03_eda` | Exploratory analysis of the panel |
| `04_feature_engineering` | Build the final country-year analysis panel and treatment matrix |
| `05_causal_analysis` | Primary estimation: Callaway-Sant'Anna overall ATT and a naive Sun-Abraham event study |
| `05b_event_study` | Event-study detail and pre-trend testing |
| `06_robustness_checks` | The paper's central diagnostics: ruling out composition/covariate explanations, Rambachan-Roth relative-magnitude sensitivity bounds, unit-specific linear trends |
| `07_results_visualization` | Figures for the write-up |
| `08_table_export` | Formatted summary, coverage, results, and sensitivity tables |
| `09_generalization_checks` | Income-convergence mechanism check; replication of the full diagnostic workflow on carbon-tax-only adoption |

## Data Sources

| Dataset | Source | Role |
|---|---|---|
| **World Carbon Pricing Database** | Dolphin & Xiahou (2022), *Scientific Data* | Treatment: carbon tax / ETS coverage, structured and auditable |
| **Our World in Data — CO2** | OWID / Global Carbon Project | Outcome: national CO2 emissions |
| **Our World in Data — Energy** | OWID | Energy-mix covariates |
| **World Bank WDI** | World Bank | GDP, population, urbanization |
| **World Bank WGI** | World Bank | Governance indicators (robustness) |

## Methodology

- **Callaway & Sant'Anna (2021)** doubly-robust group-time average treatment effect, implemented via the `differences` package, with never-treated countries as the comparison group and country-clustered inference.
- **Sun & Abraham (2021)** interaction-weighted event study, estimated via OLS with unit and year fixed effects, aggregated across adoption cohorts.
- **Joint pre-trend testing** via a cluster-robust Wald test on the full event-time covariance matrix, not a per-coefficient screen.
- **Rambachan & Roth (2023)** relative-magnitude sensitivity bounds, implemented via `honestdid`, to formally quantify how much a parallel-trends violation of a given size could be driving the post-treatment estimate.
- **Unit-specific linear trends** as an independent corroborating check on the same conclusion.

## Environment Setup

```bash
# 1. Navigate to project directory
cd causal-climate-policy

# 2. Create and activate a virtual environment
python -m venv venv_ccp
venv_ccp\Scripts\activate      # Windows
# source venv_ccp/bin/activate # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Register Jupyter kernel
python -m ipykernel install --user --name=venv_ccp --display-name="venv_ccp"

# 5. Launch Jupyter
jupyter lab
```

## Reproducing the Results

Run the notebooks in order, 01 through 09. Each notebook reads from and writes to `data/` and `outputs/` as documented above; the full pipeline is reproducible end to end with no manual intervention required beyond the initial environment setup.

## License

This project is for academic research purposes. Datasets are used under their respective licenses — see `data/raw/data_manifest.json` for source attribution and version/commit provenance.
