"""Loading helpers for the consolidated tea NIR dataset."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

META_COLS = ['campaign','campaign_name','sample_id','sample_code','form','batch','grade','scan',
             'Brightness','Briskness','Strength','Colour','Quality','Average','L','I','Q',
             'TasterScore','TenPoint','qc_flag']
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "tea_master.parquet"


def load_master(path: str | Path | None = None, drop_flagged: bool = True) -> pd.DataFrame:
    """Load the master table (one row per scan). Accepts .parquet, .csv or the master .xlsx.

    drop_flagged removes the 128 Tocklai-2023 scans whose absorbance is negative
    across the whole range (white reference mis-taken at acquisition).
    """
    path = Path(path or DEFAULT_PATH)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name="Spectra_long")
    else:
        df = pd.read_csv(path)
    if drop_flagged and "qc_flag" in df.columns:
        df = df[df.qc_flag == "ok"].reset_index(drop=True)
    return df


def wavelengths(df: pd.DataFrame) -> np.ndarray:
    return np.array([float(c) for c in df.columns if c not in META_COLS])


def spectra(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c not in META_COLS]
    return df[cols].to_numpy(float)


def sample_means(df: pd.DataFrame):
    """Average the replicate scans of each physical sample.

    Returns (X, meta) where meta holds one row per sample. Averaging replicates is
    the only honest unit of analysis here: the 16 scans of one sample are not
    independent observations.
    """
    cols = [c for c in df.columns if c not in META_COLS]
    g = df.groupby("sample_id", sort=False)
    X = g[cols].mean().to_numpy(float)
    meta = g.first().reset_index()[[c for c in META_COLS if c in df.columns]]
    return X, meta
