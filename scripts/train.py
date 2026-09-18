"""Train the screening models and write models/tea_screening_models.joblib + model cards."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from tea_nir.data import load_master, sample_means
from tea_nir.model import TeaScreeningModel, ModelCard

TASKS = [("TK2023", "powder", t) for t in ["Strength", "Briskness", "Quality", "Brightness", "Average"]] + \
        [("TK2026", "granule", t) for t in ["TasterScore", "TenPoint", "Quality"]] + \
        [("DH2023", "granule", t) for t in ["L", "I", "Q"]] + \
        [("DH2023", "powder", t) for t in ["L", "I", "Q"]]
N_PERM = 60


def build_task(df, campaign, form, target):
    sub = df[(df.campaign == campaign) & (df.form == form)]
    X, meta = sample_means(sub)
    y = pd.to_numeric(meta[target], errors="coerce").to_numpy(float)
    ok = ~np.isnan(y)
    X, y, meta = X[ok], y[ok], meta[ok].reset_index(drop=True)
    lo, hi = np.quantile(y, [1 / 3, 2 / 3])
    if lo == hi:
        # discrete, heavily tied reference (e.g. L, I, Q take only 4 values):
        # tertiles collapse, so fall back to a split at the median with the tied
        # value kept in the lower class, and say so in the model card.
        lo = hi = float(np.median(y))
        sel = np.ones(len(y), bool)
        cls = (y > hi).astype(int)
        return X, cls, lo, hi, meta
    sel = (y <= lo) | (y >= hi)
    return X[sel], (y[sel] >= hi).astype(int), float(lo), float(hi), meta[sel]


def evaluate(X, y, seeds=5):
    aucs, accs = [], []
    for s in range(seeds):
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        p = cross_val_predict(TeaScreeningModel()._build(), X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p)); accs.append(accuracy_score(y, p >= 0.5))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs))


def permutation_p(X, y, observed, n=N_PERM):
    rng = np.random.RandomState(0); null = []
    for i in range(n):
        yp = rng.permutation(y)
        cv = StratifiedKFold(5, shuffle=True, random_state=i % 3)
        p = cross_val_predict(TeaScreeningModel()._build(), X, yp, cv=cv, method="predict_proba")[:, 1]
        null.append(roc_auc_score(yp, p))
    return float((np.sum(np.array(null) >= observed) + 1) / (n + 1))


def main(data_path=None, out_dir="models"):
    df = load_master(data_path)
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    bundle, cards = {}, []
    for campaign, form, target in TASKS:
        X, y, lo, hi, meta = build_task(df, campaign, form, target)
        auc, auc_sd, acc = evaluate(X, y)
        p = permutation_p(X, y, auc)
        model = TeaScreeningModel().fit(X, y)
        split = ("top third vs bottom third" if lo < hi else
                 f"above {hi:g} vs {hi:g} and below (tied reference, tertiles collapse)")
        card = ModelCard(target=target, campaign=campaign, form=form, n_samples=int(len(y)),
                         cut_low=lo, cut_high=hi, auc_cv=auc, auc_sd=auc_sd, permutation_p=p,
                         accuracy_cv=acc, preprocessing=model.preprocessing,
                         n_components=model.n_components,
                         notes=[f"Class definition: {split}.",
                                "Trained on the top and bottom thirds of the reference range only." if lo < hi else
                                "All samples used; the reference takes only a few discrete values.",
                                "Samples in the middle third were never shown to the model.",
                                "Replicate scans averaged per sample; validation grouped by sample.",
                                "Eighteen target/dataset combinations were screened, so a nominal "
                                "p below 0.05 here is not evidence of a usable calibration."])
        key = f"{campaign}|{form}|{target}"
        bundle[key] = dict(model=model, card=card.to_dict())
        cards.append(card.to_dict())
        print(f"{key:32s} n={len(y):3d} AUC={auc:.3f}±{auc_sd:.3f} acc={acc:.3f} perm-p={p:.3f}", flush=True)
    joblib.dump(bundle, out_dir / "tea_screening_models.joblib")
    json.dump(cards, open(out_dir / "model_cards.json", "w"), indent=2)
    print("saved", out_dir / "tea_screening_models.joblib")


if __name__ == "__main__":
    main(*sys.argv[1:])
