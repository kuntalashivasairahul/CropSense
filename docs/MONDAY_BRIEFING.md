# CropSense AI — faculty briefing and demonstration

**Project:** three separately trained models joined by an illustrative dashboard score. The disease model accepts one RGB leaf photograph; the stress model accepts a paired RGB and near-infrared field tile; the pest model accepts four weeks of weather and prior pest observations. The datasets do not come from one farm, so the combined score is a simulated scenario, not a validated farm-level prediction. Hyperspectral analysis is outside this demonstration's agreed scope.

## Reproducible evidence

| Module | Executed Kaggle notebook | Evaluation and main result | What to show |
|---|---|---|---|
| Disease, MobileNetV2, 38 classes | [Module 1 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit) | PlantDoc test: 141/252 correct (55.95%), macro F1 0.552. Original checkpoint: 78/252 (30.95%), macro F1 0.297. PlantVillage validation: 96.83% for the new checkpoint. | Class report, confusion matrix, both domain results, and model selection on PlantDoc training only. |
| NDVI stress, Random Forest, 3 classes | [Module 2 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-2-field-splits-rf-and-metrics) | Retrained grouped-field baseline: 58.42% on 1,549 tiles, macro F1 0.474, healthy recall 23/280. Threshold candidate F1 0.477 but severe→healthy errors rose 74→81/700, so it was rejected. The deployed older artifact has a historical random-tile score of 66.75% on 400 tiles. | Six feature definitions, image and mask pairing, grouped split, per-class recall, severe-to-healthy errors. |
| Pest risk, LSTM, 3 classes | [Module 3 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-3-causal-lstm-and-metrics) | 2009 observed-target test: 10/19 correct (52.63%), macro F1 0.433. Persistence: 8/19; logistic regression: 13/19. | Year-based split, four-week windows, observed-only targets, all baselines and confusion matrices. |

For every module, the notebook records training configuration, dataset and split details, environment versions, model selection **before** the final test, accuracy, precision, recall, F1, support, confusion counts, and exported artifact checks. [Detailed evaluation](MODEL_EVALUATION.md) and [saved evidence](evidence/) are in the repository. Small test groups can change these percentages substantially, so state their sample counts aloud.

## Base-paper gap statements

These are topic-level comparisons based on the verified publication records; they do not claim to reproduce the papers' experiments.

- [Shrotriya et al., IEEE Access (2024)](https://doi.org/10.1109/ACCESS.2024.3411013) studies plant-leaf disease classification and comparison with conventional classification. Our work adds an explicit lab-to-field benchmark, a PlantDoc-training adaptation experiment, and a class-level tradeoff audit for the released 38-class model.
- [Le, Liou, and Pham, IEEE Access (2023)](https://doi.org/10.1109/ACCESS.2023.3283033) studies crop response to disease and water scarcity using a normalized-difference **latent heat** index. Our stress module uses a different index, RGB/NIR **NDVI**, then classifies six summary features with a Random Forest against Agriculture-Vision mask-derived labels. The index and target are different; we do not claim to reproduce their result.
- [Dong et al., IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2020)](https://doi.org/10.1109/JSTARS.2020.3013340) concerns dynamic pest and disease monitoring and early forecasting. Our pest module evaluates a four-week LSTM under a year-separated, observed-target protocol and reports persistence and logistic regression beside it. This is a retrospective study on one region, not a validated operational forecast.

## Suggested five-minute walkthrough

1. Start the app on the Windows laptop with `start_windows.cmd`. Confirm its environment check passes before presenting.
2. In Disease detection, load the Apple Black Rot reference image, classify it, then show the Grad-CAM view. Say that Grad-CAM shows influence on the prediction, not proof of a lesion. The ImageNet image check is a fallible heuristic and can flag a genuine leaf.
3. In Canopy stress, select one of the bundled RGB/NIR pairs. Show the per-pixel NDVI heatmap and six summary statistics. Explain that labels derive from anomaly masks, not agronomist assessments. The supplied healthy example may be misclassified; use it to discuss the weak healthy class rather than conceal it.
4. In Pest risk, select a four-week 2009 sequence and run the forecast. Explain that this is a retrospective held-out sequence, not live weather. Show the full class probabilities.
5. In Composite, show the equal-weight formula. No accuracy score exists for this fusion because there is no joined ground-truth dataset.

## Questions the panel may ask

**Why did the disease score fall from lab images to field images?** PlantVillage and PlantDoc differ in background, lighting, framing, and disease presentation. Adaptation improved the field benchmark but slightly reduced PlantVillage validation accuracy; a small Tomato Late Blight group got worse. The field test was inspected historically, so it is not a fresh blind replication.

**Why use an LSTM if logistic regression scores higher?** The LSTM is the required sequence-model architecture and beats persistence here, but the logistic baseline is stronger on 19 test weeks. We report that plainly; more seasons and external locations are needed before choosing a production pest model.

**Is the NDVI stress model a genuine field classifier?** It uses real RGB/NIR pixels and NDVI, but the targets are derived from masks and the field IDs are inferred from filenames. Evaluation is separated by inferred field to reduce tile leakage. It is a prototype, not an agronomic diagnosis.

**Will cloning on Windows work?** Python 3.12, exact dependency constraints, a checksummed model bundle, and platform checks are provided. CI checks Windows, macOS, and Linux; the presentation laptop still needs its own setup and offline rehearsal. Follow [the laptop checklist](WINDOWS_DEMO.md).
