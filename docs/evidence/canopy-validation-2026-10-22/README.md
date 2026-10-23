# Canopy diagnostics, 22 October 2026

These diagnostics use the previously exported Agriculture-Vision feature table. No production weights were changed.

`results.json` and `rf_validation.py.txt` compare four fresh fits using the existing training and validation partitions only. Alternative models did not preserve the baseline severe-to-healthy guardrail. These are diagnostic CSV refits, not reproductions of the original in-memory training run.

`label_diagnostic.json` describes all existing splits without fitting or predicting. It verifies the saved labels against the recorded thresholds, checks disjoint recorded field IDs, and stores input hashes. All 8,308 labels match. Of 1,505 healthy-labelled tiles, 1,491 contain at least some selected anomaly pixels, below the cutoffs. Mean raw NDVI distributions overlap substantially across classes. This does not prove irreducible error or justify changing the labels after seeing model results. The labels represent coverage thresholds, not independent plant-health diagnoses.

To reproduce the label diagnostic, copy `canopy_label_diagnostic.py.txt` to a temporary `.py` file and run it with the project Python from a **disposable copy** of the project root, or supply that root as its first argument. It writes `label_diagnostic.json` in that copy and has assertions for source counts, label agreement and group separation. Preserve these frozen evidence files when rerunning.

## Field identifier source scope

The [Agriculture-Vision maintainer README](https://github.com/SHI-Labs/Agriculture-Vision#agriculture-vision-challenge-dataset), read through gstack browse on 22 October, documents filenames as `(field id)_(x1)-(y1)-(x2)-(y2)` and says the field ID identifies the source farmland. That supports the filename-prefix grouping convention. The surrounding section describes the **2019 challenge dataset**, whereas these CSVs came from the **2017 miniscale archive**. This documentation does not establish exact archive lineage or acquisition independence by itself. The split contains 369/123/123 disjoint recorded IDs, but that remaining source-scope qualification is retained.

The archive identity is recorded separately in `../module-2-candidate/data_provenance.json`. No full archive download, sensor calibration check, mask-boundary reprocessing or agronomist relabelling was performed in this diagnostic.
