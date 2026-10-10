# CropSense AI

Final-year crop-health prototype. Three independently trained models feed an equal-weight composite score. The inputs come from unrelated datasets, so the composite is a **simulated scenario**, with no validated fusion accuracy.

| Signal | Model and input | Current evaluation |
|---|---|---|
| Leaf disease | MobileNetV2, RGB image, 38 classes | 55.95% on 252 PlantDoc field photos; 96.83% on 8,145 PlantVillage validation images |
| Canopy stress | 200-tree Random Forest, six NDVI statistics from paired RGB/NIR images | Historical random-tile test: 66.75% on 400 images; grouped-field experiment in progress |
| Pest risk | Four-week, nine-feature LSTM, three classes | 52.63% on 19 observed-target weeks; persistence 42.11%; logistic regression 68.42% |

The original CNN checkpoint scored 30.95% on the same PlantDoc set when re-evaluated in the new notebook. Its PlantVillage result was 98.10% under the same image-loading code. The field test had been inspected in prior work, so it is a historical benchmark rather than a newly blind external test. The new pest evaluation uses a different target and split protocol from the older notebook, so its percentages cannot be compared directly with the old 36.8% result. See [model evaluation](docs/MODEL_EVALUATION.md).

Fusion is `⅓ × disease_score + ⅓ × ndvi_score + ⅓ × (1 − pest_risk_score)`. Equal weights reflect the absence of labelled three-signal outcomes; they were not learned from data.

## Windows 10 laptop

Use **64-bit Python 3.12**. Version 3.12.2 is suitable. Clone the prepared branch and double-click [setup_windows.cmd](setup_windows.cmd) while online. This creates `.venv`, installs pinned dependencies, checks model hashes, caches the ImageNet helper, and tests inference without network access. It can take a while because TensorFlow is large. Then double-click [start_windows.cmd](start_windows.cmd) and open `http://localhost:8501`.

The app runs CPU inference on native Windows. Recent TensorFlow does not support native Windows GPU execution. If a DLL is missing, install the official [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).

After setup, disconnect the laptop from Wi-Fi and run the full [demo checklist](docs/WINDOWS_DEMO.md). The launch script writes `reports/environment-windows.json`; share that report and any error text if the laptop behaves differently.

The model files are tracked with Git LFS, but [release_bundle.zip](models/release_bundle.zip) also contains the exact release files. A clone with only LFS pointers restores the models from this bundle. Every restored file is verified against [artifact_manifest.json](models/artifact_manifest.json). The demonstration needs no model download.

## Mac mini

Run `./setup_mac.sh` once with Python 3.12 installed, then `./start_mac.sh`. Both launch scripts use the same pinned runtime dependencies. [Continuous integration](.github/workflows/compatibility.yml) covers Windows, Apple Silicon, and Linux; the actual laptop remains the final Windows check.

## Training and evidence

The three new Kaggle notebooks are [Module 1 field adaptation](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit), [Module 2 field-separated Random Forest](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-2-field-splits-rf-and-metrics), and [Module 3 causal LSTM](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-3-causal-lstm-and-metrics). Local source copies, the original exported notebooks, and snapshots of the earlier Kaggle versions are in [notebooks](notebooks). The runs record splits, per-class precision/recall/F1, confusion matrices, predictions, training details, package versions, and checksummed model exports. Compact results are in [docs/evidence](docs/evidence).

The disease classifier has no unknown class and can be confidently wrong. NDVI labels come from anomaly masks, not agronomist diagnoses. Pest data are sparse and from one region. Grad-CAM highlights image regions influencing a prediction; it does not prove biological correctness. Hyperspectral analysis is outside this release scope.
