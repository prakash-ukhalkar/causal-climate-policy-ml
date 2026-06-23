"""
Data loading and download utilities for the Causal Climate Policy project.

Handles programmatic downloading, caching, and loading of all required datasets
from public sources (OWID, Climate TRACE, World Bank, Climate Policy Database).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# ── Dataset URLs ─────────────────────────────────────────────────────────────

DATASET_URLS = {
    "owid_co2": {
        "url": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
        "filename": "owid-co2-data.csv",
        "subdir": "owid",
        "description": "Our World in Data — CO₂ and Greenhouse Gas Emissions",
    },
    "owid_co2_codebook": {
        "url": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-codebook.csv",
        "filename": "owid-co2-codebook.csv",
        "subdir": "owid",
        "description": "Our World in Data — CO₂ data codebook",
    },
    "owid_energy": {
        "url": "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
        "filename": "owid-energy-data.csv",
        "subdir": "owid",
        "description": "Our World in Data — Energy dataset",
    },
}


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """
    Download a file from a URL with progress bar and caching.

    Parameters
    ----------
    url : str
        The URL to download from.
    dest_path : Path
        Local file path to save to.
    force : bool
        If True, re-download even if file exists.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        logger.info(f"File already exists (skipping): {dest_path.name}")
        return dest_path

    logger.info(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"Saved: {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def download_all_datasets(force: bool = False) -> dict:
    """
    Download all datasets defined in DATASET_URLS.

    Returns
    -------
    dict
        Manifest of downloaded files with paths and checksums.
    """
    manifest = {}

    for key, info in DATASET_URLS.items():
        dest = RAW_DIR / info["subdir"] / info["filename"]
        download_file(info["url"], dest, force=force)
        manifest[key] = {
            "path": str(dest),
            "url": info["url"],
            "description": info["description"],
            "sha256": _compute_sha256(dest),
            "size_bytes": dest.stat().st_size,
            "downloaded_at": datetime.now().isoformat(),
        }

    return manifest


def save_manifest(manifest: dict, path: Path = None) -> None:
    """Save data manifest to JSON."""
    if path is None:
        path = RAW_DIR / "data_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {path}")


def load_owid_co2() -> pd.DataFrame:
    """Load the OWID CO₂ dataset from raw directory."""
    path = RAW_DIR / "owid" / "owid-co2-data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run download first.")
    return pd.read_csv(path)


def load_owid_energy() -> pd.DataFrame:
    """Load the OWID Energy dataset from raw directory."""
    path = RAW_DIR / "owid" / "owid-energy-data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run download first.")
    return pd.read_csv(path)


def load_interim(filename: str) -> pd.DataFrame:
    """Load a cleaned interim dataset."""
    path = INTERIM_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Interim file not found: {path}")
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a final processed dataset."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_csv(path)
