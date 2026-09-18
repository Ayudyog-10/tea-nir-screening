"""Reproduce the audit: leakage demonstration, honest regression benchmark, positive controls."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, cross_val_predict, cross_val_score
from tea_nir.data import load_master, sample_means, spectra
from tea_nir.preprocess import Preprocessor

TARGETS = {("TK2023","powder"): ["Brightness","Briskness","Strength","Quality","Average"],
           ("TK2026","granule"): ["Brightness","Briskness","Strength","Colour","Quality","TasterScore","TenPoint"],
           ("DH2023","powder"): ["L","I","Q"], ("DH2023","granule"): ["L","I","Q"]}


def honest_regression(df):
    rows = []
    for (camp, form), targets in TARGETS.items():
        sub = df[(df.campaign == camp) & (df.form == form)]
        X, meta = sample_means(sub)
        Xp = Preprocessor("snv_sg1").fit_transform(X)
        for t in targets:
            y = pd.to_numeric(meta[t], errors="coerce").to_numpy(float)
            ok = ~np.isnan(y); yy = y[ok]; Xv = Xp[ok]; g = meta.sample_code.to_numpy()[ok]
            pls = GridSearchCV(PLSRegression(scale=False), {"n_components": list(range(1, 16))},
                               cv=KFold(5, shuffle=True, random_state=0),
                               scoring="neg_root_mean_squared_error")
            preds = np.zeros((3, len(yy)))
            for r in range(3):
                ug = np.unique(g); rng = np.random.RandomState(r)
                perm = dict(zip(ug, rng.permutation(len(ug))))
                gi = np.array([perm[x] for x in g])
                for tr, te in GroupKFold(5).split(Xv, yy, gi):
                    preds[r, te] = np.ravel(pls.fit(Xv[tr], yy[tr]).predict(Xv[te]))
            rmse = np.sqrt(((preds - yy) ** 2).mean(1)).mean()
            r2 = np.mean([1 - ((p - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum() for p in preds])
            rows.append(dict(campaign=camp, form=form, target=t, n=int(ok.sum()), sd=float(yy.std(ddof=1)),
                             R2=float(r2), RMSECV=float(rmse), RPD=float(yy.std(ddof=1) / rmse)))
    return pd.DataFrame(rows)


def leakage_demo(df):
    sub = df[df.campaign == "TK2023"]
    Xp = Preprocessor("snv_sg1").fit_transform(spectra(sub))
    g = sub.sample_code.to_numpy()
    out = []
    for t in ["Quality", "Average"]:
        y = sub[t].to_numpy(float)
        for name, est in [("KNN (k=3)", KNeighborsRegressor(3)), ("PLSR (10 LV)", PLSRegression(10))]:
            leaky = np.ravel(cross_val_predict(est, Xp, y, cv=KFold(10, shuffle=True, random_state=0)))
            honest = np.ravel(cross_val_predict(est, Xp, y, cv=GroupKFold(10), groups=g))
            r2 = lambda p: 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            out.append(dict(target=t, model=name, R2_scans_split_randomly=float(r2(leaky)),
                            R2_samples_held_out=float(r2(honest))))
    return pd.DataFrame(out)


def positive_controls(df):
    res = {}
    sub = df[df.campaign == "TK2023"]
    X, meta = sample_means(sub)
    Xp = Preprocessor("snv_sg1").fit_transform(X)
    pipe = make_pipeline(StandardScaler(), PCA(15), LogisticRegression(max_iter=5000))
    res["batch_accuracy"] = float(cross_val_score(pipe, Xp, meta.batch, cv=5).mean())
    res["batch_chance"] = float(meta.batch.value_counts(normalize=True).max())
    res["grade_accuracy"] = float(cross_val_score(pipe, Xp, meta.grade, cv=5).mean())
    res["grade_chance"] = float(meta.grade.value_counts(normalize=True).max())
    dh = df[df.campaign == "DH2023"]
    Xd, md = sample_means(dh.assign(sample_id=dh.form + "|" + dh.sample_id))
    Xd = Preprocessor("snv_sg1").fit_transform(Xd)
    form = md.sample_id.str.split("|").str[0]; grp = md.sample_id.str.split("|").str[1]
    res["form_accuracy"] = float(cross_val_score(pipe, Xd, form, cv=GroupKFold(5), groups=grp).mean())
    # repeatability
    scans = df[df.campaign == "TK2023"]
    Xs = pd.DataFrame(Preprocessor("snv_sg1").fit_transform(spectra(scans)))
    within = Xs.groupby(scans.sample_code.to_numpy()).transform(lambda v: v - v.mean()).to_numpy()
    between = Xs.groupby(scans.sample_code.to_numpy()).mean().to_numpy()
    res["within_scan_sd"] = float(within.std(0).mean())
    res["between_sample_sd"] = float(between.std(0).mean())
    res["signal_to_repeatability_ratio"] = res["between_sample_sd"] / res["within_scan_sd"]
    return res


if __name__ == "__main__":
    df = load_master(sys.argv[1] if len(sys.argv) > 1 else None)
    out = Path("reports"); out.mkdir(exist_ok=True)
    reg = honest_regression(df); reg.to_csv(out / "honest_regression.csv", index=False)
    leak = leakage_demo(df); leak.to_csv(out / "leakage_demo.csv", index=False)
    pc = positive_controls(df); json.dump(pc, open(out / "positive_controls.json", "w"), indent=2)
    print(reg.round(3).to_string(index=False)); print(); print(leak.round(3).to_string(index=False)); print(); print(pc)
