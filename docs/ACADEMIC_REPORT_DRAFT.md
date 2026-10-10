# CropSense AI: Multispectral Crop Stress, Leaf Disease, and Pest Risk Analysis

**Team 2, AIML-E:** T. Smrithi (23EG107E61), K. Shiva Sai Rahul (23EG107E30), A. Aasritha (23EG107E05).
**Status:** evaluation draft, 11 October 2026. Check the laptop result before submission.

## Abstract

CropSense AI is a dataset-based software prototype that presents three distinct analyses in one dashboard: leaf disease classification from an RGB photograph, field stress classification from paired RGB and near-infrared imagery, and pest-risk classification from recent weather and scouting observations. A MobileNetV2 classifier was adapted using PlantDoc training photographs after a large laboratory-to-field performance gap was measured. Its field test accuracy rose from 30.95% to 55.95% on the same 252-image benchmark, with a small reduction in PlantVillage validation accuracy. A Random Forest consumes six NDVI summary features; a baseline retrained under a grouped-field protocol gives 58.42% accuracy and shows weak healthy-class recall. A four-week LSTM reaches 52.63% accuracy on 19 observed-target weeks, above persistence but below logistic regression. Each result is reported with class-level precision, recall, F1, confusion counts, split membership, training configuration, and exported model identity. The three datasets do not identify the same farm, so their equal-weight combination is an illustrative dashboard score without a validated accuracy figure.

## Problem and scope

Leaf symptoms, canopy reflectance, and weather-linked pest pressure describe different parts of crop condition. A single dataset covering all three signals and a farm-level outcome was unavailable for this prototype. We therefore train and evaluate each module independently and combine their outputs only for a simulated demonstration. This release uses RGB and near-infrared imagery for multispectral NDVI. Hyperspectral processing is excluded from the current agreed scope. The system is retrospective and dataset-based; it is not a live sensor network or an agronomic decision aid.

The related-work gap is stated conservatively in the [faculty briefing](MONDAY_BRIEFING.md): published work addresses leaf classification, spectral crop response, and pest forecasting, while this prototype emphasizes a transparent evaluation of each component and demonstrates their integration. No publication is claimed to have used our exact datasets, split, or score formula.

## Data and methods

**Disease module.** The closed-set classifier predicts one of 38 PlantVillage crop-disease categories from a 224×224 RGB image. Raw 0–255 pixel values enter the saved MobileNetV2 graph, which contains its own preprocessing. The original checkpoint was compared with a PlantDoc-adapted checkpoint using the same image loader. Adaptation used a held-out part of PlantDoc *training* images for checkpoint selection. The official PlantDoc test set and PlantVillage validation set were then evaluated with class-level reports. Exact decoded-image duplicates across PlantDoc train and test were removed; near duplicates and overlap with PlantVillage have not been ruled out. The PlantDoc test had been inspected historically, so it is not newly blind. Grad-CAM displays influential image regions but cannot prove a biological lesion.

**Canopy stress module.** Paired Agriculture-Vision RGB and single-band NIR field tiles are aligned by filename. At each pixel, NDVI is `(NIR − red)/(NIR + red + 1e−8)` using raw 8-bit band values. Mean, standard deviation, minimum, maximum, 25th percentile, and 75th percentile become the Random Forest features. Water-heavy tiles are excluded. Healthy, moderate, and severe training labels derive from the coverage of specified anomaly masks, not agronomist diagnoses. The new evaluation holds inferred field IDs apart across training, validation, and test to reduce adjacent-tile leakage; the field naming convention remains to be externally confirmed. The historical random-tile split and new grouped split cannot be directly compared as an improvement claim.

**Pest module.** The ICAR-CRIDA Akola cotton Jassid and Aphid weather/scouting records provide 2005–2009 weekly observations. A sequence contains four consecutive weeks within one growing season and nine features. Only past observations enter the input; a target is evaluated only where both pest values were observed. Training uses 2005–2007, validation uses 2008, and the 2009 season is reserved for final comparison. Weather scaling and class thresholds derive from training years; the scaler is embedded in the exported Keras model. A majority-class rule, previous-week persistence, and standardized logistic regression are reported beside the LSTM. The dataset has one region and only 19 usable test weeks.

**Fusion and dashboard.** The score is `(disease_score + ndvi_score + (1 − pest_risk_score))/3`. The equal weights are a presentation choice because no joined labels exist to learn them. Its output is a simulated scenario, not an experimentally validated health score. The Streamlit app also shows individual model probabilities, NDVI imagery and statistics, and the limitations of closed-set disease prediction.

## Results

| Evaluation | Accuracy | Macro F1 | Additional finding |
|---|---:|---:|---|
| Original disease checkpoint, PlantDoc test (252) | 30.95% | 0.297 over present classes | Same loader as new checkpoint |
| Adapted disease checkpoint, PlantDoc test (252) | **55.95%** | **0.552** over present classes | Tomato Late Blight recall fell on a 10-image subgroup |
| Adapted disease checkpoint, PlantVillage validation (8,145) | 96.83% | 0.956 | Original checkpoint: 98.10%, macro F1 0.977 |
| Retrained NDVI baseline, grouped-field test (1,549) | 58.42% | 0.474 | Healthy recall 23/280; severe→healthy 74/700 |
| Pest LSTM, 2009 observed-target weeks (19) | 52.63% | 0.433 | Persistence 42.11%; logistic regression 68.42% |

Confusion matrices, support, per-class precision/recall/F1, predictions, split manifests, model-selection records, learning curves, versions, and hashes are in the three executed [Kaggle audit notebooks](../README.md#training-and-evidence) and the repository [evidence](evidence/). The complete numerical definitions and caveats are in [Model Evaluation](MODEL_EVALUATION.md). The grouped baseline is a new fit of the previous configuration, not a direct evaluation of the exact older deployed forest.

## Discussion and limitations

The disease model gains 25.00 percentage points on the historically inspected PlantDoc benchmark but still performs well below its lab validation result. The change is not uniformly beneficial: the small Tomato Late Blight test subgroup loses recall. The disease softmax cannot output “unknown,” and an auxiliary ImageNet/color heuristic can produce false alarms even on genuine leaf images.

For NDVI stress, aggregate accuracy conceals a weak healthy class and a costly severe-to-healthy error type. One proposed forest increased macro F1 but raised severe-to-healthy errors from 74 to 102 on the grouped test. A validation-selected probability threshold reduced that tradeoff on validation, yet the held-out test still rose from 74 to 81 severe-to-healthy errors. Both replacements were rejected. This is a case where optimizing one headline metric would make the demonstration less trustworthy. The mask-derived labels also limit biological interpretation.

For pests, the year-separated LSTM is stronger than persistence under the new observed-target protocol but weaker than logistic regression. The older notebook used interpolated targets and a different experimental protocol; its reported score is not a directly comparable baseline. Nineteen test weeks from one location do not support a broad forecasting claim.

The composite score has no accuracy metric because its three inputs are evaluated on unrelated crops, farms, and datasets. The prototype should be assessed through its three module reports and its transparent integration behavior. Future work would require external field validation, more seasons and locations, agronomist stress labels, and a joined dataset before learning or evaluating the fusion weights.

## Reproducibility and demonstration

Notebook code and outputs identify every split, model configuration, export, and package version. The repository pins Python 3.12 dependencies, checksums the release models, and includes a bundle that restores them when a clone has only Git LFS pointers. Windows, Linux, and Mac checks compare inference outputs against reference predictions. The actual Windows 10 laptop must pass the [offline rehearsal checklist](WINDOWS_DEMO.md) before presentation.

## References

1. A. Shrotriya et al., “An Approach Toward Classifying Plant-Leaf Diseases and Comparisons With the Conventional Classification,” *IEEE Access*, vol. 12, pp. 117379–117398, 2024. [DOI: 10.1109/ACCESS.2024.3411013](https://doi.org/10.1109/ACCESS.2024.3411013).
2. M. S. Le, Y.-A. Liou, and M.-T. Pham, “Crop Response to Disease and Water Scarcity Quantified by Normalized Difference Latent Heat Index,” *IEEE Access*, vol. 11, pp. 55938–55946, 2023. [DOI: 10.1109/ACCESS.2023.3283033](https://doi.org/10.1109/ACCESS.2023.3283033).
3. Y. Dong et al., “Automatic System for Crop Pest and Disease Dynamic Monitoring and Early Forecasting,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 13, pp. 4410–4418, 2020. [DOI: 10.1109/JSTARS.2020.3013340](https://doi.org/10.1109/JSTARS.2020.3013340).
4. [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), [PlantDoc dataset](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset), and [ICAR data snapshot](https://www.kaggle.com/datasets/kuntalashivasairahul/icar-data); dataset-specific access and derivation are recorded in the notebooks.
