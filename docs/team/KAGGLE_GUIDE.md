# Kaggle, datasets, and experiment evidence

Use the [notebook index](../../notebooks/README.md) to open all sources locally. Runtime inference uses the released files in `models/`; it does not require Kaggle credentials, training datasets, or notebook execution.

## Which notebooks to study

| Audit source and recorded hosted notebook | Purpose | Recorded completed version |
|---|---|---|
| [Module 1 source](../../notebooks/cropsense-module-1-audit/cropsense-module-1-audit.ipynb), [Kaggle](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit) | Inventory/mapping, checkpoint verification, lab/field evaluation; default audit path verifies the locked release | v5, `20261011T112353Z`; v3 preserves a separate fresh training run |
| [Module 2 source](../../notebooks/cropsense-module-2-audit/cropsense-module-2-audit.ipynb), [Kaggle](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-2-field-splits-rf-and-metrics) | RGB/NIR/mask extraction, grouped-field splits, RF comparisons, validation-only threshold selection | v5, `20261011T112352Z` |
| [Module 3 source](../../notebooks/cropsense-module-3-audit/cropsense-module-3-audit.ipynb), [Kaggle](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-3-causal-lstm-and-metrics) | Causal seasonal windows, observed targets, LSTM versus simple baselines, saved contracts | v3, `20261011T112402Z` |

These are historical saved-run identifiers from checked-in evidence, not a claim that the hosted pages were rechecked on the handoff date. All three audit metadata files have **`is_private: true`**. Teammates may need the owner's Kaggle sharing access to view those hosted runs. Local sources and compact evidence are already included. A GitHub branch cannot grant Kaggle access; account credentials must not be shared.

The source copies have no embedded execution outputs. To study recorded results offline, use the evidence below. For hosted output walkthroughs and screenshots, read [Kaggle notebook presentation](../KAGGLE_NOTEBOOK_PRESENTATION.md). Its dates and saved version references remain historical; current local source includes later audit corrections.

## Data and access

| Module | Sources / requirements | What is in this branch |
|---|---|---|
| 1 | [PlantVillage Kaggle dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and [PlantDoc mirror](https://www.kaggle.com/datasets/nirmalsankalana/plantdoc-dataset); attach both in Kaggle. Metadata enables GPU and Internet. | Source/mapping/split records, compact predictions/metrics, recovered training/verification records, and a few app images; full datasets excluded |
| 2 | Agriculture-Vision 2017 mini archive, downloaded by the notebook from the public S3 URL below. Internet is enabled; metadata has no attached dataset and GPU is off. | 18 paired RGB/NIR examples plus manifest; grouped split/features/predictions and source hashes in evidence; full archive excluded |
| 3 | ICAR-CRIDA Akola cotton Jassid/Aphid endpoints in the notebook. Internet is enabled, GPU is off. Failed acquisition stops the audit rather than inventing observations. | Preserved raw observations, processed data, split/prediction records, and 19 app sequences |

Agriculture-Vision archive source:

`https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_paper_2020/Dataset/data2017_miniscale.tar.gz`

The recorded v5 archive SHA-256 is `2b4bf0b2357ba982fe6d81da55841874afdd721a867bb8fddede9e04e4914027`. The recorded size is about **1.75 GiB**, with extra disk space needed for extraction. Keep dataset downloads outside the Git checkout. Current URL availability, account permissions, and source terms must be checked before a new download. Local app setup does not require this archive.

## Evidence map and interpretation

| Directory | What it establishes / limits |
|---|---|
| `docs/evidence/module-1-candidate/` | Preserved adapted checkpoint evaluation, per-class results, confusion counts, predictions, and split records |
| `docs/evidence/disease-recovery-2026-10-22/` | Recovered original training and locked-model verification records, hashes, and split-overlap analysis |
| `docs/evidence/module-2-candidate/` | Grouped-field RF baseline/candidate results and the rejected selection; these are not deployed-forest metrics |
| `docs/evidence/rf-provenance-2026-10-22/` | September forest origin trace; original hash-at-run and split/prediction rows remain missing |
| `docs/evidence/canopy-validation-2026-10-22/` | Mask-derived label and retrained-baseline diagnostics |
| `docs/evidence/module-3-candidate/` | LSTM, persistence, logistic, and majority comparisons and recorded contracts |
| `docs/evidence/pest-export-recovery-2026-10-22/` | Original saved pest-run exports and release-output agreement; equivalent weights/configuration need not imply byte-identical Keras containers |
| `docs/evidence/pest-recovery-2026-10-22/` | Separately identified reconstruction and regional diagnostics |
| `docs/evidence/ood-validation-2026-10-22/` | Image-screen diagnostic, including accepted non-leaf examples and flagged source-labelled leaves |
| `docs/evidence/kaggle-browser/` | Historical screenshots of completed hosted output cells |
| `models/artifact_manifest.json` | Identity of the actual release artifacts used by the app |

The [runtime contract](../RUNTIME_CONTRACT.md), [model evaluation](../MODEL_EVALUATION.md), and [artifact handoff](../KAGGLE_ARTIFACT_HANDOFF.md) explain unresolved records. In particular:

- The CNN release is pinned by hash; a new training run is a new candidate even if it follows similar code.
- The retained RF's reported September **267/400 (66.75%)** has artifact consistency but incomplete original run identity. The October export **260/399 (65.16%)** and grouped baseline **58.42%** are separate records.
- The released pest model is **4×9**, with **10/19** recorded test successes and **0/5** medium recall. Logistic regression's **13/19** should remain visible in comparisons.
- There is no joint fusion accuracy, and previously inspected data cannot become a new blind holdout by rerunning it.

## If the team later reruns training

1. Read the audit source and its metadata before executing. Original notebooks are historical; the app's pinned dependencies are an inference environment, not a guarantee of a complete Kaggle training environment.
2. Fork/import into the teammate's own Kaggle account. Change metadata notebook IDs to that account for any upload, attach the required datasets, and review Internet/GPU settings. Do not publish to the original owner's IDs or copy credentials into Git.
3. Use Kaggle paths/settings from the source. These notebooks use `/kaggle/...` locations and are not unmodified PC training scripts. The Module 1 default locked-checkpoint verification is different from a new fine-tuning experiment.
4. Save source/version, data hashes and exclusions, split/sample IDs, parameters/seeds, environment, selection rules, predictions/probabilities, class metrics/confusion counts, and export hashes together. Inspect completed outputs rather than relying on a successful job status.
5. Keep new output as a named candidate. Preserve existing release files and historical results. Review model quality and input/feature/class compatibility before proposing any model replacement, then update manifest, recovery bundle, references, tests, and documentation as one reviewed change.
