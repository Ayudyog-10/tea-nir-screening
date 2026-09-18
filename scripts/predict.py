"""Command-line screening: feed a CSV of spectra, get band verdicts.

    python scripts/predict.py my_scans.csv --model "TK2026|granule|TasterScore"

The CSV must have one row per scan, a `sample_id` column and 256 absorbance
columns in the same wavelength order as the master file (892.43-1709.86 nm).
Replicate scans of the same sample_id are averaged before screening.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BANNER = ("RESEARCH PREVIEW - these models separate the top third from the bottom third of a\n"
          "reference range at AUC 0.55-0.75. They are not a substitute for a tea taster and\n"
          "must not be used to accept, reject or price a consignment.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--model", default="TK2026|granule|TasterScore")
    ap.add_argument("--bundle", default=str(Path(__file__).resolve().parents[1] / "models" / "tea_screening_models.joblib"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    bundle = joblib.load(a.bundle)
    if a.model not in bundle:
        raise SystemExit(f"unknown model. available: {list(bundle)}")
    entry = bundle[a.model]; model, card = entry["model"], entry["card"]
    df = pd.read_csv(a.csv)
    id_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    spec_cols = [c for c in df.columns if c != id_col]
    X = df.groupby(id_col, sort=False)[spec_cols].mean()
    res = model.screen(X.to_numpy(float))
    out = [dict(sample_id=s, **r) for s, r in zip(X.index, res)]
    if a.json:
        print(json.dumps(dict(model=a.model, card=card, results=out), indent=2)); return
    print(BANNER); print()
    print(f"model {a.model}  |  trained on {card['n_samples']} samples  |  AUC {card['auc_cv']:.2f}")
    print(f"upper band = reference above {card['cut_high']:.2f}, lower band = below {card['cut_low']:.2f}")
    print()
    for r in out:
        print(f"{r['sample_id']:20s} P(upper)={r['probability_upper']:.2f}  {r['verdict']}")


if __name__ == "__main__":
    main()
