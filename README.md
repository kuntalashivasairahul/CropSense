
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
- **Explainable AI (Grad-CAM).** The disease module computes a gradient-weighted class
  activation map over the MobileNetV2 backbone (`out_relu`), visually grounding the
  diagnosis in foliar lesions rather than background artifacts.
- **Out-of-Distribution (OOD) Sanity Check.** A dual-heuristic pre-check combining HSV
  vegetation color coverage analysis (greens/yellows/browns $\ge 15\%$) and stock ImageNet
  MobileNetV2 classification (verifying plant-adjacent class representation in top-5). Flags
  non-leaf inputs with an explanatory caution box, demonstrating awareness of closed-set
  classifier limitations without hiding model predictions.
- **Demo Scenarios & Advisory Reports.** Includes one-click test scenarios (Healthy,
  Early Warning, Severe Outbreak, and Adversarial Non-Leaf Test), actionable agronomic
  intervention protocols, and downloadable assessment reports.
- **Multispectral (RGB+NIR) Canopy Analysis.** Module 2 accepts paired visible (RGB) and
  near-infrared (NIR) multispectral satellite tiles, computes per-pixel NDVI
  $((\text{NIR} - \text{Red}) / (\text{NIR} + \text{Red} + 10^{-8}))$, extracts 6 spatial
  summary statistics (`mean, std, min, max, p25, p75`), renders low-NDVI stress overlays
  and true NDVI heatmaps, and provides verified ground-truth sample parcels.
- **Modern Light Theme & Restrained Motion Choreography.** Built with an airy agricultural
  design language (fixed diagonal `#DDEBF3` to `#F0F4C3` gradient, `#0E1A12` ink, `#C6E94A` lime
  fills, self-hosted IBM Plex typography, and WCAG AA/AAA compliance). Pure CSS motion includes
  `linear()` spring easing, clip-path card reveals, `@property` count-up numerals, hand-built
  inline SVG composite gauge and pest timeline area charts, blurred stats ribbons, and full
  `prefers-reduced-motion` accessibility support.

## Layout

```
app.py                     Streamlit presentation layer (light theme, SVG visualisations)
core.py                    model loading, inference, score formulas, Grad-CAM, OOD checks
test_core.py               runnable verification suite for all models and heuristics
models/                    trained artefacts (git-lfs) & module2_samples/ (RGB+NIR pairs)
sample_images/             labelled PlantVillage samples and adversarial non-leaf test
static/                    self-hosted IBM Plex Sans and Mono fonts
.streamlit/config.toml     Streamlit light theme configuration
```

