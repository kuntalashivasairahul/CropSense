# Kaggle files needed to finish the model audit

The original public Random Forest notebook and its September saved output have now been recovered directly. The most useful remaining files are the **original `final_df_scaled.csv`, split/prediction files, or working-output archive**, if you retained them. There is no need to retrain or calculate hashes before sending files.

## Which Random Forest run

The deployed `module2_rf.pkl` is 13,996,161 bytes and matches the artifact uploaded on **19 September 2026**. The recovered [public version 1](https://www.kaggle.com/code/kuntalashivasairahul/module-2?scriptVersionId=351052482) records September execution, 267/400 correct, matching file size, parameters and six reported feature importances. Its public page currently shows zero output files, so an owner-retained archive may still be needed. The checked-in `notebooks/originals/module-2.ipynb` records a **9 October rerun**, and October grouped-field audits fitted separate candidates. Preserve these distinctions. The [origin trace](evidence/rf-provenance-2026-10-22/README.md) records the evidence and the missing hash/split link.

Download the saved version's existing output files as an archive. Preserve original filenames and notebook outputs. A link to that saved version is useful even if the archive is incomplete. Put downloaded files in a separate handoff folder, such as `reports/kaggle-handoff/`, or share their local path; do not overwrite the project's model files.

## What to include if it was saved

| Item | Why it is needed |
|---|---|
| Exported `.pkl` or `.keras` model | Compare the actual trained artifact with deployment. A notebook metric alone does not identify a model file. |
| Input dataset links and version numbers, or downloaded archive identity | Identify the exact source snapshot. |
| Train/validation/test filenames or indices, including ordering | Recover the actual partition rather than generating a different split. This matters for the old unsorted 2,000-image RF sample. |
| Class names/order, feature order, normalization and label thresholds | Verify that training and application inputs mean the same thing. |
| True labels, predictions and class probabilities for evaluation samples | Recompute accuracy, per-class precision/recall/F1 and confusion counts for the same artifact and split. |
| Saved metrics, seed, parameters and package versions | Interpret and reproduce the run. |
| CNN/LSTM training history, selected epoch and checkpoint-selection rule | Establish which checkpoint was exported and how it was selected. This epoch history is not required for the forest. |

**CNN and LSTM exports are recovered; no need to resend them.** The [CNN v1 training and v5 verification exports](evidence/disease-recovery-2026-10-22/README.md) include the full split manifest, histories, selection and sample-level predictions. Both exported CNNs match deployment exactly. The [pest v3 archive](evidence/pest-export-recovery-2026-10-22/README.md) includes original observations, all 89 windows and selected-model/baseline results. Its Keras container differs from deployment, but weights and effective configuration match, and deployment reproduces all 31 validation/test probability rows within 1e-6. The earlier public-data reconstruction remains separately labelled.

A targeted local search found only `module2_samples.zip`, which contains the 18 RGB/NIR demo pairs and their manifest, not the RF training CSV or original split. The September public output page lists no files and a direct request for `final_df_scaled.csv` returned 404. If you still have that CSV or the original working-output ZIP, it is the most useful next handoff; do not rerun training to replace the historical record.

If some files were never saved, say which ones are missing. The notebook and model are still useful, but missing historical split/prediction records limit what can be verified. A new run can be evaluated as a new experiment; it must not be substituted silently for the original model's historical result.

Independent labelled field data and actual Windows laptop results are separate remaining requirements. Recovering a notebook archive does not by itself repair model quality, sensor calibration or cross-region generalization.
