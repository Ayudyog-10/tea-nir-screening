"""METASPEQ tea NIR — data audit and screening preview (Streamlit)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tea_nir.data import load_master, sample_means, spectra, wavelengths, META_COLS
from tea_nir.preprocess import Preprocessor

st.set_page_config(page_title="METASPEQ Tea NIR", page_icon="🍃", layout="wide")


@st.cache_data
def get_data():
    return load_master(ROOT / "data" / "tea_master.parquet", drop_flagged=False)


@st.cache_resource
def get_models():
    return joblib.load(ROOT / "models" / "tea_screening_models.joblib")


@st.cache_data
def get_reports():
    r = {}
    for name in ["honest_regression", "leakage_demo"]:
        p = ROOT / "reports" / f"{name}.csv"
        if p.exists():
            r[name] = pd.read_csv(p)
    p = ROOT / "reports" / "positive_controls.json"
    if p.exists():
        r["positive_controls"] = json.load(open(p))
    return r


df = get_data()
models = get_models()
reports = get_reports()

st.title("🍃 Tea NIR — screening preview and data audit")
st.error(
    "**Research preview.** On this data, NIR does **not** predict tea-taster scores. "
    "The strongest screening model reaches AUC 0.72 on 43 samples; every quantitative "
    "calibration has R² at or below zero once all scans of a sample are held out together. "
    "Nothing here may be used to accept, reject, grade or price tea."
)

tab_screen, tab_data, tab_audit, tab_models = st.tabs(
    ["Screen a sample", "Dataset", "Audit findings", "Model cards"])

with tab_screen:
    key = st.selectbox("Model", list(models), index=list(models).index("TK2026|granule|TasterScore"))
    entry = models[key]; card = entry["card"]; model = entry["model"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", card["n_samples"])
    c2.metric("AUC (CV)", f"{card['auc_cv']:.2f} ± {card['auc_sd']:.2f}")
    c3.metric("Accuracy", f"{card['accuracy_cv']:.0%}")
    c4.metric("Permutation p", f"{card['permutation_p']:.3f}")
    st.caption(f"Upper band: reference above {card['cut_high']:g}. "
               f"Lower band: below {card['cut_low']:g}. "
               "Fourteen dataset/target combinations were screened, so an uncorrected "
               "p below 0.05 is not evidence of a usable calibration.")

    src = st.radio("Spectra", ["Pick a sample from the dataset", "Upload a CSV"], horizontal=True)
    X = None
    if src.startswith("Pick"):
        camp = st.selectbox("Campaign", sorted(df.campaign.unique()))
        sub = df[df.campaign == camp]
        sid = st.selectbox("Sample", sorted(sub.sample_id.unique()))
        rows = sub[sub.sample_id == sid]
        X = pd.DataFrame(spectra(rows).mean(0)[None, :], index=[sid])
        ref = rows.iloc[0]
        shown = {c: ref[c] for c in ["Brightness", "Briskness", "Strength", "Colour", "Quality",
                                     "L", "I", "Q", "TasterScore", "TenPoint"]
                 if c in rows.columns and pd.notna(ref[c])}
        st.write("Reference values on file:", shown)
        if ref.get("qc_flag", "ok") != "ok":
            st.warning("This sample is flagged: negative absorbance across the range (bad white reference).")
    else:
        up = st.file_uploader("CSV: one row per scan, first column sample_id, then 256 absorbance columns", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            idc = raw.columns[0]
            X = raw.groupby(idc, sort=False)[list(raw.columns[1:])].mean()

    if X is not None and st.button("Screen", type="primary"):
        for sid, r in zip(X.index, model.screen(X.to_numpy(float))):
            verdict = r["verdict"]
            msg = f"**{sid}** — P(upper band) = {r['probability_upper']:.2f} → **{verdict}**"
            (st.error if verdict == "OUT OF DOMAIN" else st.warning if verdict == "BORDERLINE" else st.info)(msg)
            st.caption(f"Hotelling T² {r['hotelling_t2']:.1f} (limit {r['t2_limit']:.1f}) · "
                       f"Q residual {r['q_residual']:.1f} (limit {r['q_limit']:.1f})")

with tab_data:
    st.subheader("What is in the consolidated file")
    summary = (df.groupby(["campaign_name", "form"])
                 .agg(samples=("sample_id", "nunique"), scans=("sample_id", "size")).reset_index())
    st.dataframe(summary, use_container_width=True, hide_index=True)
    camp = st.selectbox("Show spectra for", sorted(df.campaign.unique()), key="spec")
    sub = df[df.campaign == camp]
    Xs, meta = sample_means(sub)
    wl = wavelengths(df)
    mode = st.selectbox("Pre-processing", ["raw", "snv", "snv_sg1", "sg2"])
    Z = Xs if mode == "raw" else Preprocessor(mode).fit_transform(Xs)
    st.line_chart(pd.DataFrame(Z.T, index=np.round(wl, 1)))
    st.caption("Each line is one sample (replicate scans averaged).")

with tab_audit:
    st.subheader("Replicate leakage")
    if "leakage_demo" in reports:
        st.dataframe(reports["leakage_demo"].round(3), use_container_width=True, hide_index=True)
    st.caption("Splitting the 16 scans of a sample at random puts near-copies of the test spectra "
               "into training. That is what produces R² above 0.8 with a nearest-neighbour model.")
    st.subheader("Honest calibration performance")
    if "honest_regression" in reports:
        st.dataframe(reports["honest_regression"].round(3), use_container_width=True, hide_index=True)
    st.subheader("Positive controls")
    pc = reports.get("positive_controls", {})
    if pc:
        c1, c2, c3 = st.columns(3)
        c1.metric("Granule vs powder", f"{pc['form_accuracy']:.0%}", "chance 50%")
        c2.metric("Manufacturing batch", f"{pc['batch_accuracy']:.0%}", f"chance {pc['batch_chance']:.0%}")
        c3.metric("Sample vs scan noise", f"{pc['signal_to_repeatability_ratio']:.0f}×")
        st.caption("The instrument sees real, repeatable differences between samples. "
                   "What it cannot see is the taster's score.")

with tab_models:
    cards = pd.DataFrame([e["card"] for e in models.values()])
    st.dataframe(cards[["campaign", "form", "target", "n_samples", "auc_cv", "auc_sd",
                        "accuracy_cv", "permutation_p", "preprocessing", "n_components"]].round(3),
                 use_container_width=True, hide_index=True)
    st.caption("METASPEQ · Ayudyog Private Limited")
