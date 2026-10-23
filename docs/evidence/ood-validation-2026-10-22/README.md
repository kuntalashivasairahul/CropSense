# Image-check diagnostic, 22 October 2026

The existing color/ImageNet heuristic was evaluated on **554 declared inputs**, with no decoding/inference failures and no exact duplicate decoded evaluation images. This measures the heuristic on selected source examples, not general field eligibility or disease accuracy. Production model weights, keywords and decision thresholds were not changed using these outcomes.

| Source | Inputs | Flagged | Accepted | Interpretation |
|---|---:|---:|---:|---|
| Upstream PlantDoc test examples | 236 | 38 | 198 | 16.1% flagged against source leaf labels; not an independently adjudicated farmer-photo false-rejection rate. |
| Nine Caltech non-plant object categories | 270 | 270 | 0 | All selected faces, leopards, motorbikes, airplanes, cars, pianos, boats, scorpions and stop signs were flagged. |
| Caltech sunflower category | 30 | 1 | 29 | 96.7% accepted despite being outside the leaf task; 24 accepted images also had forced disease confidence at least 0.5. |
| Bundled aerial crop scenes | 18 | 16 | 2 | Neither accepted scene reached disease confidence 0.5, so the separate low-confidence rule still requests review. |

Across all 300 Caltech examples, 29/300 (9.7%) passed the heuristic. Pooling a category-balanced sample into one number hides the flower failure. For example, one sunflower was called Strawberry healthy at 95.47% confidence with healthy-class probability mass 0.9909. A plant-related ImageNet class is not proof of a supported leaf. The application upload guidance and exported limitations now state this directly. The direct color-check entry point also gained the shared pixel cap before allocation; that boundary repair does not change these within-limit heuristic decisions.

For PlantDoc, the color screen flagged six examples and the ImageNet screen flagged 34; their union was 38. Source class labels are only a proxy for eligibility. Visual review of one fixed example per test folder showed isolated leaves, multi-leaf scenes, text overlays, slides and montages. No examples were removed after review or scoring. The raw results preserve all per-category counts, input hashes, resizing, ImageNet guesses and forced CNN results.

## Sources and provenance

- [PlantDoc classification repository](https://github.com/pratikkayal/PlantDoc-Dataset), commit `5467f6012d78d1c446145d5f582da6096f852ae8`: all 236 test image blobs were fetched through gstack browse and matched their Git blob hashes; SHA-256 values are retained. The 27 folders and all decoded dimensions are in `plantdoc_preflight.json`.
- [Caltech 101, version 1.0, DOI 10.22002/D1.20086](https://data.caltech.edu/records/mzrjq-6wc02), by Fei-Fei Li, Marco Andreeto, Marc'Aurelio Ranzato and Pietro Perona: the 137,414,764-byte ZIP matched published MD5 `3138e1922a9193bfa496528edbbc45d0`. `caltech_retrieval.json` records its new SHA-256. Selection was fixed at 30 images per declared category, ordered by SHA-256 of category/filename. The source record identifies CC BY 4.0; the PlantDoc README also identifies CC BY 4.0. Original dataset attributions remain required.
- All 18 bundled canopy RGB tiles were included separately; these are existing demonstration examples, not fresh field observations.

The pinned upstream PlantDoc tree contains **2,342 training and 236 test images, 2,578 total**. Its README abstract quotes 2,598 data points, and the consumed Kaggle mirror was documented as 2,912 images. Those are different snapshots. This recovery does not establish the mirror's exact derivation, replace its historical 252-image evaluation, or recover the original training-run split records.

## Reproduction and limitations

`protocol.json` was written before scoring. `selected_inputs_before_scoring.json` freezes the actual cohort before model calls. `source_review_before_scoring.json` records the source-label caveat without changing selection. `results.json` contains the complete rows and summaries. The core source hash in those results is the tested image-check implementation before the subsequent export-warning text edit; the image-check decisions were unchanged by that wording edit.

To reproduce, use a disposable project copy. Fetch the pinned PlantDoc images with `download_ood_leaves.py.txt` copied to a `.py` file, using gstack's browse executable. Place the checksum-verified Caltech ZIP at `reports/audit-fixes-2026-10-22/ood-data/caltech-101.zip`, then run `evaluate_ood.py.txt` as Python from the project root. The scripts write fixed evidence paths in that copy. Image files and the large dataset ZIP remain outside the versioned evidence package; hashes and source paths permit retrieval. An evaluation copy is resized, preserving aspect ratio, to at most 1024×1024 before the normal core pipeline. Eighty-nine PlantDoc examples needed that transform.

This diagnostic is now inspected development evidence. No threshold or keyword tuning on it is presented as independently validated improvement. Near duplicates, ImageNet pretraining overlap, unknown crops, camera conditions, sensor variation and farmer-population coverage remain unverified. A calibrated leaf/crop eligibility model would need an explicit target definition, separate fitting/validation data and independent validation. The OOD-quality finding remains open.
