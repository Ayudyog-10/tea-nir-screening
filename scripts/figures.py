"""Generate the figures used in the white paper."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.decomposition import PCA
from tea_nir.data import load_master, sample_means, spectra, wavelengths
from tea_nir.preprocess import Preprocessor

OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
C = {"TK2023|powder": "#1f77b4", "DH2023|powder": "#2ca02c", "DH2023|granule": "#98df8a",
     "TK2026|granule": "#d62728"}
LBL = {"TK2023|powder": "Tocklai 2023 (powder)", "DH2023|powder": "Dhunsree 2023 (powder)",
       "DH2023|granule": "Dhunsree 2023 (granule)", "TK2026|granule": "Tocklai 2026 (granule)"}


def fig1(df, wl):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, c in C.items():
        camp, form = k.split("|")
        sub = df[(df.campaign == camp) & (df.form == form)]
        X, _ = sample_means(sub)
        m, s = X.mean(0), X.std(0)
        ax[0].plot(wl, m, color=c, label=LBL[k]); ax[0].fill_between(wl, m - s, m + s, color=c, alpha=.15)
        Xp = Preprocessor("snv_sg1").fit_transform(X)
        ax[1].plot(wl, Xp.mean(0), color=c)
    ax[0].set(xlabel="Wavelength (nm)", ylabel="Absorbance", title="Mean spectra ± 1 SD")
    ax[1].set(xlabel="Wavelength (nm)", ylabel="SNV + 1st derivative", title="After pre-processing")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig1_spectra.png"); plt.close(fig)


def fig2(df):
    X, meta = sample_means(df)
    Xp = Preprocessor("snv_sg1").fit_transform(X)
    S = PCA(3).fit(Xp)
    Z = S.transform(Xp)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for k, c in C.items():
        camp, form = k.split("|")
        m = (meta.campaign == camp) & (meta.form == form)
        ax.scatter(Z[m.to_numpy(), 0], Z[m.to_numpy(), 1], s=18, color=c, label=LBL[k], alpha=.8)
    ax.set(xlabel=f"PC1 ({S.explained_variance_ratio_[0]*100:.0f}%)",
           ylabel=f"PC2 ({S.explained_variance_ratio_[1]*100:.0f}%)",
           title="Sample-mean spectra, all campaigns")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig2_pca.png"); plt.close(fig)


def fig3(df):
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3))
    sets = [("TK2023", "powder", ["Brightness", "Briskness", "Strength", "Quality"], "Tocklai 2023"),
            ("DH2023", "powder", ["L", "I", "Q"], "Dhunsree 2023"),
            ("TK2026", "granule", ["Brightness", "Briskness", "Strength", "Quality"], "Tocklai 2026")]
    for a, (camp, form, cols, title) in zip(ax, sets):
        _, meta = sample_means(df[(df.campaign == camp) & (df.form == form)])
        data = [pd.to_numeric(meta[c], errors="coerce").dropna() for c in cols]
        a.boxplot(data, labels=cols, widths=.6)
        for i, d in enumerate(data, 1):
            a.scatter(np.random.normal(i, .06, len(d)), d, s=6, alpha=.35, color="#555")
        a.set_title(title); a.set_ylabel("Score")
    fig.tight_layout(); fig.savefig(OUT / "fig3_reference.png"); plt.close(fig)


def fig4():
    leak = pd.read_csv("reports/leakage_demo.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    lbl = leak.target + "\n" + leak.model
    x = np.arange(len(leak))
    ax.bar(x - .2, leak.R2_scans_split_randomly, .4, label="Scans split at random (leaky)", color="#d62728")
    ax.bar(x + .2, leak.R2_samples_held_out, .4, label="All scans of a sample held out", color="#1f77b4")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylabel("Cross-validated R²"); ax.set_title("What replicate leakage buys you")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig4_leakage.png"); plt.close(fig)


def fig5():
    reg = pd.read_csv("reports/honest_regression.csv")
    reg["lab"] = reg.campaign + " " + reg.form + " · " + reg.target
    reg = reg.sort_values("R2")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(reg.lab, reg.R2, color=np.where(reg.R2 > 0, "#1f77b4", "#bbbbbb"))
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Cross-validated R² (samples held out)")
    ax.set_title("Honest PLSR performance, every target")
    ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(OUT / "fig5_regression.png"); plt.close(fig)


def fig6():
    cards = pd.DataFrame(json.load(open("models/model_cards.json")))
    cards["lab"] = cards.campaign + " " + cards.form + " · " + cards.target
    cards = cards.sort_values("auc_cv")
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.errorbar(cards.auc_cv, cards.lab, xerr=cards.auc_sd, fmt="o", color="#1f77b4", ms=4)
    ax.axvline(.5, color="k", lw=.8, ls="--")
    ax.set_xlabel("AUC, top third vs bottom third (5-fold CV, 5 repeats)")
    ax.set_title("Two-band screening is the only place any signal survives")
    ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(OUT / "fig6_auc.png"); plt.close(fig)


def fig7(tt):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.3))
    cols = ["Brightness", "Briskness", "Strength", "Colour", "Overall Quality"]
    rs, md = [], []
    for c in cols:
        a = pd.to_numeric(tt[0][c], errors="coerce"); b = pd.to_numeric(tt[1][c], errors="coerce")
        ok = a.notna() & b.notna()
        rs.append(a[ok].corr(b[ok])); md.append((a[ok] - b[ok]).abs().mean())
        if c == "Overall Quality":
            ax[0].scatter(a[ok] + np.random.normal(0, .08, ok.sum()), b[ok] + np.random.normal(0, .08, ok.sum()),
                          s=18, alpha=.7, color="#d62728")
            lim = [1, 10]; ax[0].plot(lim, lim, "k--", lw=.8)
            ax[0].set(xlabel="Taster 1 score", ylabel="Taster 2 score", title="Overall quality, same 57 teas")
    ax[1].bar(cols, rs, color="#1f77b4")
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("Correlation between tasters")
    ax[1].tick_params(axis="x", labelsize=7, rotation=20)
    ax[1].set_title("Agreement between two tasters")
    fig.tight_layout(); fig.savefig(OUT / "fig7_tasters.png"); plt.close(fig)


def fig8():
    pc = json.load(open("reports/positive_controls.json"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    names = ["Granule vs powder\n(Dhunsree)", "Manufacturing batch\n(Tocklai 2023)", "Leaf grade\n(Tocklai 2023)"]
    got = [pc["form_accuracy"], pc["batch_accuracy"], pc["grade_accuracy"]]
    chance = [0.5, pc["batch_chance"], pc["grade_chance"]]
    x = np.arange(3)
    ax[0].bar(x - .2, got, .4, label="Model", color="#1f77b4")
    ax[0].bar(x + .2, chance, .4, label="Chance", color="#cccccc")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7)
    ax[0].set_ylabel("Accuracy"); ax[0].set_title("Positive controls: the spectra do carry information")
    ax[0].legend(fontsize=7)
    ax[1].bar(["Repeat scans\nof one sample", "Differences between\nsamples"],
              [pc["within_scan_sd"], pc["between_sample_sd"]], color=["#cccccc", "#1f77b4"])
    ax[1].set_ylabel("Mean SD (pre-processed units)")
    ax[1].set_title(f"Sample differences are {pc['signal_to_repeatability_ratio']:.0f}× instrument noise")
    fig.tight_layout(); fig.savefig(OUT / "fig8_controls.png"); plt.close(fig)


if __name__ == "__main__":
    df = load_master(sys.argv[1] if len(sys.argv) > 1 else None)
    wl = wavelengths(df)
    fig1(df, wl); fig2(df); fig3(df); fig4(); fig5(); fig6(); fig8()
    tt_path = Path("reports/taster_tt1.csv")
    if tt_path.exists():
        fig7([pd.read_csv("reports/taster_tt1.csv"), pd.read_csv("reports/taster_tt2.csv")])
    print("figures written to", OUT)
