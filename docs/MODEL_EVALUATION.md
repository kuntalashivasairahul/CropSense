# CropSense AI — model evaluation and decisions

**Recorded 10 October 2026, with later grouped-NDVI results added 11 October.** The model hashes are in [artifact_manifest.json](../models/artifact_manifest.json). The Kaggle audit code exported [saved evidence](evidence/) with per-class precision, recall, F1, support, confusion counts, training history, package versions, split membership, and export checks. The checked-in notebook copies do not embed cell outputs. The composite score has no accuracy metric because the three datasets do not describe one labelled farm outcome.

Accuracy is correct predictions divided by all predictions. Macro F1 weights classes equally; weighted F1 accounts for class size. A confusion matrix has true classes in rows and predictions in columns. Each notebook stores probabilities and sample-level predictions so mistakes can be audited. Scores under different split or target rules must be kept separate.

## Module 1: MobileNetV2, 38 classes

The field candidate was selected on a held-out portion of the Kaggle PlantDoc mirror's **training** data. Its checkpoint and the original checkpoint were then compared on the mirror's same 252 retained test examples. Exact decoded-image duplicates between these mirror partitions were removed; near duplicates and overlaps with PlantVillage remain possible. This split differs from the upstream dataset's test split.

| Evaluation | Original checkpoint | Adapted candidate |
|---|---:|---:|
| Kaggle PlantDoc mirror test accuracy, n=252 | 30.95% | **55.95%** |
| PlantDoc macro F1 over 27 present classes | 0.297 | **0.552** |
| PlantDoc weighted F1 | 0.297 | **0.554** |
| PlantVillage validation accuracy, n=8,145 | **98.10%** | 96.83% |
| PlantVillage macro F1 | **0.977** | 0.956 |

The original status document quoted 98.29% lab accuracy and about 30.6% PlantDoc accuracy from earlier runs. This table evaluates both checkpoints through one notebook and one image loader. PlantDoc test had been inspected in prior work, and PlantVillage validation was used to select the original model; neither is a newly blind external test. There is a class tradeoff: on the small Tomato Late Blight subset, recall fell from 0.80 to 0.20, while Tomato healthy recall rose from 0 to 0.25. The full [class report](evidence/module-1-candidate/adapted_checkpoint__field_test_classification_report.csv) and [confusion counts](evidence/module-1-candidate/adapted_checkpoint__field_test_confusion_counts.csv) include all 38 output classes.

An 11 October fresh retraining in Kaggle Module 1 **v3** produced a different candidate: **139/252 (55.16%)** on PlantDoc test, macro F1 **0.539**, and **96.77%** PlantVillage validation. Its SHA-256 differs from the released checkpoint. Module 1 **v5** directly re-evaluated the **locked release model** without retraining and reproduced the table above, including its artifact hash and a short faculty-facing summary. The [Kaggle walkthrough](KAGGLE_NOTEBOOK_PRESENTATION.md) identifies each version and what its outputs establish.

**Decision:** use the adapted checkpoint as the release candidate. It preserves class order, loads under the pinned Mac runtime, and still generates Grad-CAM. Laptop and cross-platform checks remain required. This is a closed-set classifier; confidence is not accuracy.

**Original exports recovered, 22 October:** the [v1 training archive and v5 verification archive](evidence/disease-recovery-2026-10-22/README.md) contain identical deployed CNN bytes and full split manifests. Both sets of saved prediction tables were checked against labels, probabilities and metrics. The 2,912 mapped field examples are retained **after** deduplication; the raw mirror count remains unknown. Exact RGB hash comparison found 185/236 upstream test images in mirror test, 2 in mirror validation, and 49 without an exact match. Thus the upstream test partition is not blind to this model's selection. The local notebook now logs explicit chart-folder exclusions, excluded file counts and accepted source membership before deduplication; no labels or weights were promoted.

## Image eligibility diagnostic, 22 October

A fixed [554-image diagnostic](evidence/ood-validation-2026-10-22/README.md) evaluated the existing color/ImageNet heuristic without tuning it. It flagged 38/236 source-labelled examples from the pinned upstream PlantDoc test set; those examples include slides and montages, so this is not an adjudicated farmer-photo false-rejection rate. All 270 selected non-plant object images were flagged, but 29/30 sunflower images passed; 24 accepted flowers also received forced crop-disease confidence at least 0.5. These results do not validate eligibility or change disease accuracy. The upstream 236-image test snapshot is different from the historical Kaggle mirror's 252-image evaluation; its recovered tree contains 2,578 image files in total, and the mirror derivation remains unresolved.

## Module 2: six-feature NDVI Random Forest

**Source recovery, 22 October 2026:** two historical records are now separated. The recovered public September notebook reports **267/400 = 66.75%**, macro F1 **0.537**, healthy recall **7/67**, and severe→healthy **9/185**. Its exported size, all six printed feature importances, parameters and training count match the local deployment's corresponding observations. The deployed bytes also match the 19 September hosted upload. This corrects the initial audit's implication that the 66.75% claim lacked a saved source. The notebook does not record an export hash or preserve original split/prediction rows, so it remains a **reported historical result with strong artifact consistency, not a fully replayed holdout verification**. See the [recovered notebook and checks](evidence/rf-provenance-2026-10-22/README.md).

The checked-in October rerun and [linked output](evidence/module-2-original-outputs.txt) separately show 1,993 accepted tiles and **260/399 = 65.16%**, macro F1 approximately **0.525**, healthy recall **7/69**, and **7** severe-to-healthy errors. Its serialized model size differs from deployment. Do not substitute that later rerun for the September result. Both archived outputs remain unchanged.

The new notebook processed all 8,345 RGB/NIR pairs and accepted 8,308 after water exclusion. It separated inferred field identifiers across train, validation, and test. This is a harder evaluation protocol, so its percentages are **not** directly comparable to the historical random-tile result. The six statistics, raw NDVI formula, and heuristic label rules are preserved. The dataset maintainers document a field-ID filename prefix for the challenge dataset; the exact 2017 miniscale archive lineage still needs confirmation. Labels are not agronomist-verified. A 22 October descriptive audit reproduced all 8,308 stored threshold labels: 1,491/1,505 healthy-labelled tiles contain selected anomaly pixels below the cutoffs. Thus healthy means below selected mask thresholds, not verified plant health; see the [label diagnostic](evidence/canopy-validation-2026-10-22/label_diagnostic.json) and [source-scope note](evidence/canopy-validation-2026-10-22/README.md). The grouped baseline below was **retrained on the new training split** with the same 200-tree configuration; it is not the exact deployed artifact from the old notebook.

The first grouped run found a candidate with test macro F1 **0.492** versus **0.474** for its grouped baseline, but severe→healthy errors increased from **74/700** to **102/700**. It was rejected. The second run tested seven RF configurations with a strict **validation** rule: healthy recall cannot fall and severe→healthy errors cannot rise. It selected the baseline configuration. The third run swept a healthy-class minimum-probability rule on validation and selected a candidate forest with threshold 0.45. Both were assessed on the same grouped test. The split comprised 5,175 training, 1,584 validation and 1,549 test images from disjoint inferred field IDs.

| Grouped test, 1,549 tiles | Retrained baseline configuration | Selected threshold candidate |
|---|---:|---:|
| Accuracy | **58.42%** | 57.84% |
| Macro F1 | 0.474 | **0.477** |
| Healthy recall | 23/280 (8.21%) | **29/280 (10.36%)** |
| Severe classified healthy | **74/700** | 81/700 |

On validation, the threshold candidate had macro F1 **0.491** versus **0.481** for baseline, healthy recall **56/318** versus **42/318**, and the same **26/613** severe→healthy errors. It therefore met the predeclared validation guardrails. The held-out test revealed seven additional severe→healthy errors, so the candidate was **rejected for deployment** despite its small F1 gain. [Per-class reports, both confusion matrices, sample-level predictions, and the threshold sweep](evidence/module-2-candidate/) are saved. The exact older deployment artifact remains in the release bundle; its historical random-tile score should not be substituted for this retrained grouped baseline result.

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

**Original exports recovered, 22 October:** the [public v3 archive](evidence/pest-export-recovery-2026-10-22/README.md) restores original source tables, 260 processed weeks, all 89 split windows, histories and predictions. Its serialized Keras file differs from deployment; the weights payload is identical and configurations match after serialization-ID canonicalization. Deployment reproduces the 12 validation and 19 test probability rows within 1e-6. These saved-run checks do not establish independent field or region accuracy.

## Release boundary

The repo contains the adapted CNN, revised LSTM with matching sample sequences, and the previous Random Forest because grouped candidate forests did not meet the held-out severe→healthy safety guardrail. All release files are checksummed in a bundle. The Mac mini and Windows Server 2025 and Linux GitHub runners pass the [offline inference checks](evidence/compatibility/); the actual Windows 11 laptop demo remains unverified. Those runner records describe the earlier snapshot, not the 22 October runtime corrections. Hyperspectral work is excluded from this deadline; the dashboard demonstrates RGB/NIR multispectral NDVI.
