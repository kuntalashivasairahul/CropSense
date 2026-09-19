---
title: CropSense AI
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.64.0
app_file: app.py
pinned: false
license: mit
---

# CropSense AI

A composite crop-health dashboard that fuses three independently trained models
into a single score. Final-year B.Tech project.

| Tab | Model | Output |
|---|---|---|
| Disease Detection | MobileNetV2 transfer-learning CNN, 38 classes | `disease_score` |
| NDVI Crop Stress | RandomForestClassifier, 200 trees, 3 classes | `ndvi_score` |
| Pest Risk | LSTM (16 units) over a 4-week weather window | `pest_risk_score` |
| Fusion Dashboard | equal-weighted combination | `composite_score` |

```
composite_score = ⅓·disease_score + ⅓·ndvi_score + ⅓·(1 − pest_risk_score)
```

Buckets: **Healthy** ≥ 0.66 · **Moderate Risk** 0.33–0.66 · **Severe Risk** < 0.33

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_core.py      # sanity-checks all three models against known inputs
streamlit run app.py
```

## Notes on the models

These are served exactly as trained — the dashboard performs no retraining or
fine-tuning.

- **Disease CNN.** The saved model rescales internally (`x / 127.5 − 1`,
  MobileNetV2 `preprocess_input`, baked in as graph ops right after the input
  layer). Images are therefore fed as **raw 0–255 RGB** after a 224×224 resize.
  Adding manual normalisation here would double-scale the input and silently
  corrupt every prediction.
- **Disease CNN, domain shift.** Validated at 98.3% on lab-condition
  PlantVillage images but ~30.6% on real-world field photos. This is a known,
  documented limitation, surfaced in the UI.
- **Pest LSTM.** Trained on a limited historical dataset (~5 seasons, one
  region) and documented as a data-volume-limited baseline, included per the
  module's required architecture. Sequences are drawn from real test-set weeks
  and fed unscaled, as exported alongside the model.
- **Equal weighting** is a deliberate choice: with three models trained on three
  unrelated datasets there is no labelled ground truth linking all three signals
  to one outcome, so no statistically grounded weighting can be derived.

## Layout

```
app.py                     Streamlit UI (4 tabs)
core.py                    model loading, inference, score formulas
test_core.py               runnable sanity check
models/                    trained artefacts (git-lfs)
sample_images/             labelled PlantVillage samples for the demo
.streamlit/config.toml     theme
```
