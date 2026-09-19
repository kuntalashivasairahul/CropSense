
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

## Deployment

Target host is **Streamlit Community Cloud**: point it at this repo, branch
`main`, main file `app.py`, Python 3.12.

The model artefacts are git-lfs tracked here, but Community Cloud does not fetch
lfs objects — they would arrive as ~130-byte pointer stubs. `core.model_path()`
detects that and pulls the real artefact from
[rahulkuntala/cropsense-models](https://huggingface.co/rahulkuntala/cropsense-models)
instead. Locally, where lfs did run, the checked-out files are used directly and
nothing is downloaded.

Hugging Face Spaces is not the host: HF has retired the `streamlit` SDK for new
Spaces, and `docker`/`gradio` Spaces now require a PRO subscription. Model repos
remain free, which is why the artefacts live there.

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
- **Artefact hosting.** The three files in `models/` and the copies in the HF
  model repo are byte-identical (24,935,986 / 13,996,161 / 44,843 bytes).
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
