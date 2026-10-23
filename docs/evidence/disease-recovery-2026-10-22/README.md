# Recovered CNN training and verification exports

Recovered from the public Kaggle output downloads on 22 October 2026. These are saved-run records, not a newly blind benchmark or a new training run.

| Saved version | Run | Archive SHA-256 | What it establishes |
|---|---|---|---|
| [Module 1 v1](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit?scriptVersionId=357023819) | `20261010T105018Z` | `e8b0c172cae0028d3d1633bf9db3d29107f098a34fc2b316b52834aae10cd2dc` | Original adaptation histories, checkpoint selection, splits and predictions |
| [Module 1 v5](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit?scriptVersionId=357329688) | `20261011T112353Z` | `2f11802463d7af273de84f71d8baecefe7bf1c61080d5768a277880bbb14af44` | Verification of the locked release, without retraining |

The v1 and v5 candidate model files both match the deployed CNN exactly: `5c263a58b9d4613625af07b344500bd45011336c680b30b2095c6b051d0d8deb`. Their full split manifests are byte-identical. The recorded selection history contains two head-training epochs and eight fine-tuning epochs; the maximum recorded validation macro F1 is 0.559977 at fine-tuning epoch 8. This documents the saved selection rule; it does not establish reproducibility across GPU runs.

All five prediction tables in each archive were checked against their split memberships, class order, probability bounds/sums, argmax classes, confusion matrices, accuracy and macro F1. Four prediction CSVs are byte-identical across versions. The original checkpoint's field-validation probabilities differ by at most 0.000001431; its sample IDs, true labels and predicted classes are identical. See [training-to-verification.json](training-to-verification.json). The release's saved field result remains **141/252 = 55.95%**, macro F1 **0.552** over the 27 present classes, with PlantVillage validation **7,887/8,145 = 96.83%**.

## Split and source limitations

The saved manifest has 46,160 PlantVillage training, 8,145 PlantVillage validation, 2,128 field training, 532 field validation and 252 field test records. **2,912 is the retained mapped field count after deduplication, not a proven raw mirror total.** Six exact train/test duplicate exclusions are recorded separately; within-split deduplication was not fully logged. One directory-level unmapped entry also does not establish how many individual source files were excluded. All retained field decoded hashes are unique across the three field partitions; PlantVillage hashes were not populated.

The Kaggle mirror's test partition differs from the pinned [upstream repository](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8). Comparing the 236 upstream test images with the retained Kaggle manifest found **185 exact decoded RGB matches in field test, 2 in field validation and 49 without an exact match; none matched field training**. The [complete comparison](upstream-test-hash-comparison.json) uses the notebook's `SHA256(str(RGB.size) + RGB bytes)` rule. A missing exact match does not exclude resizing, re-encoding or other near duplicates. Because two upstream test images entered model-selection validation, that upstream partition cannot be presented as a blind test of this release. The historical 55.95% result belongs to the mirror's 252-example partition.

## Source-label review and local notebook fix

Two upstream images in `Tomato two spotted spider mites leaf` were visually reviewed and verified against the pinned Git blob hashes. Both are multi-condition comparison charts with panels labelled Bacterial Spot, Bacterial Speck, Early Blight and Mite Damage. Assigning each entire chart a single mite label is not justified. Their identities and decisions are in [source-label-review.json](source-label-review.json); neither decoded hash appears in the retained field manifest.

The current local audit notebook keeps that folder excluded with an explicit label-review reason, records excluded source-file counts, and exports an accepted source inventory before deduplication. Its 27-class mapping and retained-data policy are unchanged. A regression exercises known exclusions and unknown folders. This fixes audit transparency; other source labels, montages and crop eligibility still need review. Historical notebooks and production weights were preserved.

## Files and replay boundary

`source/` preserves v5 JSON/CSV/Markdown; `original-training/source/` separately preserves v1 records. Large CSVs are losslessly gzip-compressed, with decompressed bytes checked against the originals. Each `archive_members.json` records every original member's size and SHA-256, including model/figure files that are not duplicated here. Each `validation.json` and inert `verification_script.py.txt` records the checks. Replaying a script requires its unchanged ZIP under `reports/kaggle-recovery-2026-10-22/` in a disposable project copy; the scripts refuse to overwrite frozen validation evidence. No cloud execution or model replacement occurred.
