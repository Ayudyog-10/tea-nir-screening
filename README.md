# tea-nir-screening

NIR screening of black tea against tea-taster scores — models, audit and Streamlit app.
METASPEQ · Ayudyog Private Limited.
https://tea-nir-screeninggit-6sfekaxtaq2wurzpmzrrzb.streamlit.app/

## Headline finding

**On the data collected so far, NIR does not predict tea-taster scores.**

| Check | Result |
|---|---|
| PLSR calibration, all scans of a sample held out together | R² between −0.47 and +0.15 on 18 target/dataset combinations |
| Same data, scans split at random (the usual mistake) | R² up to 0.81 with a nearest-neighbour model |
| Best two-band screening model (top vs bottom of the range) | AUC 0.72 on 43 samples, uncorrected p = 0.033 |
| Positive control — granule vs powder | 99 % accuracy |
| Positive control — manufacturing batch, 14 classes | 75 % accuracy against 9 % chance |
| Agreement between two tasters on the same 57 teas | r ≈ 0.6, exact agreement 43–70 % |

The instrument and the spectra are fine: they separate batches and physical forms
easily, and sample-to-sample differences are about seven times the scan-to-scan
noise. What is missing is a reference that NIR could plausibly track — the taster
scores are compressed into a few discrete steps, disagree between tasters, and
were collected on too few samples.

## Repository layout

```
src/tea_nir/       preprocessing, data loading, screening model with applicability domain
scripts/train.py       train every screening model, write model cards
scripts/benchmark.py   leakage demonstration, honest PLSR benchmark, positive controls
scripts/figures.py     the eight figures used in the white paper
scripts/predict.py     command-line screening for a CSV of scans
app/app.py             Streamlit app (screening, dataset browser, audit, model cards)
models/                tea_screening_models.joblib + model_cards.json
data/tea_master.parquet  consolidated spectra, 4112 scans, 254 samples, three campaigns
reports/               benchmark tables and figures
```

## Install and run

```bash
pip install -r requirements.txt
python scripts/benchmark.py        # reproduce the audit tables
python scripts/train.py            # retrain the screening models
streamlit run app/app.py           # the app
python scripts/predict.py data/example_scans.csv --model "TK2026|granule|TasterScore"
```

## The data

| Campaign | Samples | Scans | Form | Reference |
|---|---|---|---|---|
| Tocklai black tea, Aug 2023 | 100 | 1600 | powder | Brightness, Briskness, Strength, Quality |
| Dhunsree tea, Jun 2023 | 47 ×2 forms | 1536 | granule + powder | L, I, Q |
| Tocklai black tea, Jan 2026 | 60 | 976 | granule | five descriptors, taster valuation, 10-point total |

Spectra are absorbance, 892.43–1709.86 nm, 256 points, diffuse reflectance,
MetaspeQ NIR (Hamamatsu C14486GA head), 16 scans per sample.

`data/tea_master.parquet` is the single source of truth; the same content ships as
`Tea_NIR_Master_Raw_Spectra.xlsx` for Drive. 128 Tocklai-2023 scans (8 samples)
carry `qc_flag = NEGATIVE_ABSORBANCE_BAD_REFERENCE` and are excluded by default.

## How the models are validated

* Replicate scans are averaged per sample; a sample is never split across folds.
* Screening = top third vs bottom third of the reference range (or a median split
  where the reference has too many ties), 5-fold CV repeated five times.
* Each AUC is checked against 60 label permutations.
* Fourteen combinations were screened, so a nominal p below 0.05 is expected by
  chance and none survives correction.
* Every prediction carries Hotelling T² and Q-residual checks; spectra outside the
  training domain are reported as OUT OF DOMAIN rather than scored.

## Deploy on Streamlit Community Cloud

1. Push to a repository under the **Ayudyog-10** account (sign in as Ayudyog-10, or
   add your personal account as a collaborator first, otherwise the push fails 403).
2. share.streamlit.io → *Create app* → repo, branch `main`, main file `app/app.py`.
3. *Advanced settings* → **Python 3.12** (models were saved with scikit-learn 1.8.0).
4. Keep the app private and invite collaborators with *Share*.

Docker alternative: `docker build -t tea-nir . && docker run -p 8501:8501 tea-nir`

---
METASPEQ · Ayudyog Private Limited
