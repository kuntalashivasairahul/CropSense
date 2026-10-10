# CropSense AI — model evaluation and decisions

**Recorded 10 October 2026.** The model hashes are in [artifact_manifest.json](../models/artifact_manifest.json). Executed Kaggle notebooks and [compact evidence](evidence/) contain per-class precision, recall, F1, support, confusion counts, training history, package versions, split membership, and export checks. The composite score has no accuracy metric because the three datasets do not describe one labelled farm outcome.

Accuracy is correct predictions divided by all predictions. Macro F1 weights classes equally; weighted F1 accounts for class size. A confusion matrix has true classes in rows and predictions in columns. Each notebook stores probabilities and sample-level predictions so mistakes can be audited. Scores under different split or target rules must be kept separate.

## Module 1: MobileNetV2, 38 classes

The field candidate was selected on a held-out portion of PlantDoc **training** data. Its checkpoint and the original checkpoint were then compared on the same PlantDoc test images. Exact decoded-image duplicates between PlantDoc train and test were removed; near duplicates and overlaps with PlantVillage remain possible.

| Evaluation | Original checkpoint | Adapted candidate |
|---|---:|---:|
| PlantDoc test accuracy, n=252 | 30.95% | **55.95%** |
| PlantDoc macro F1 over 27 present classes | 0.297 | **0.552** |
| PlantDoc weighted F1 | 0.297 | **0.554** |
| PlantVillage validation accuracy, n=8,145 | **98.10%** | 96.83% |
| PlantVillage macro F1 | **0.977** | 0.956 |

The original status document quoted 98.29% lab accuracy and about 30.6% PlantDoc accuracy from earlier runs. This table evaluates both checkpoints through one notebook and one image loader. PlantDoc test had been inspected in prior work, and PlantVillage validation was used to select the original model; neither is a newly blind external test. There is a class tradeoff: on the small Tomato Late Blight subset, recall fell from 0.80 to 0.20, while Tomato healthy recall rose from 0 to 0.25. The full [class report](evidence/module-1-candidate/adapted_checkpoint__field_test_classification_report.csv) and [confusion counts](evidence/module-1-candidate/adapted_checkpoint__field_test_confusion_counts.csv) include all 38 output classes.

**Decision:** use the adapted checkpoint as the release candidate. It preserves class order, loads under the pinned Mac runtime, and still generates Grad-CAM. Laptop and cross-platform checks remain required. This is a closed-set classifier; confidence is not accuracy.

## Module 2: six-feature NDVI Random Forest

The original exported notebook's final run used 1,996 sampled images, with 400 in a stratified random test split. It reported **66.75% accuracy**, macro F1 about **0.537**, healthy recall **7/67**, and **9** true-severe images classified healthy. These are the actual saved [notebook outputs](evidence/module-2-original-outputs.txt); the continuity document's 398-image result was from a different run.

The new notebook processed all 8,345 RGB/NIR pairs and accepted 8,308 after water exclusion. It separated inferred field identifiers across train, validation, and test. This is a harder evaluation protocol, so its percentages are **not** directly comparable to the historical random-tile result. The six statistics, raw NDVI formula, and heuristic label rules are preserved. The inferred field identifier convention needs dataset-level confirmation, and labels are not agronomist-verified.

The first grouped run found a candidate with test macro F1 **0.492** versus **0.474** for its grouped baseline, but severe→healthy errors increased from **74/700** to **102/700**. It was rejected. The second run applies a strict validation rule: healthy recall cannot fall and severe→healthy rate cannot rise. Its final result is pending; the deployed model remains the prior 200-tree forest until a candidate passes both evaluation and integration checks.

## Module 3: four-week pest LSTM

The original Kaggle regression notebook used 2009 for early stopping, concatenated training seasons before building windows, and interpolated missing pest values using later observations. Its result is retained as history. The deployed classifier accepted **4×9** features, whereas the continuity document described 4×12; the exported original notebook did not explain that classifier artifact.

The new notebook uses 2005–2007 for fitting, 2008 for validation, and 2009 for final evaluation. Windows contain only consecutive weeks within one season. Inputs use past observations only; targets require both pests observed. Normalization and class thresholds come from training years, and weather normalization is embedded in the exported Keras graph.

| Model on 2009 observed-target weeks, n=19 | Accuracy | Macro F1 |
|---|---:|---:|
| Training-majority baseline | 36.84% | 0.180 |
| Previous-week persistence | 42.11% | 0.376 |
| **Selected LSTM, 16 units** | **52.63%** | **0.433** |
| Standardized logistic regression | **68.42%** | **0.676** |

The LSTM beats persistence under this protocol but trails a simpler model. Nineteen weeks from one region are too few for a strong generalization claim. The LSTM remains the release candidate because the architecture is required, while the stronger baseline is reported plainly. Its reloaded predictions match, and the companion `sample_sequences.json` contains 19 windows with the same feature order. The new 52.63% cannot be directly compared with the older 36.8% on interpolated targets.

## Release boundary

The repo currently contains the adapted CNN, revised LSTM with matching sample sequences, and the previous Random Forest while the second grouped run completes. All files are checksummed in a bundled release. The Mac offline inference suite passes. Native Windows installation and the full laptop demo remain unverified. Hyperspectral work is excluded from this deadline; the dashboard demonstrates RGB/NIR multispectral NDVI.
