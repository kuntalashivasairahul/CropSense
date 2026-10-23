# Deployed Random Forest origin trace

On 22 October, the local deployment was compared with pinned public Hugging Face metadata and its Git history. No remote model was downloaded and no weights were changed.

The 13,996,161-byte `module2_rf.pkl`, SHA-256 `2fc55e901be9f29f209f7bd30c1e33a4a793c999b3f87c8e901faaaade8eafe2`, matches:

- The LFS identity at the [first hosted RF upload](https://huggingface.co/rahulkuntala/cropsense-models/commit/8d5a322df80ad76d0000a9ae183cc00d3f1803da), recorded **19 September 2026, 16:00:13 UTC**.
- The current hosted RF identity at repository revision `4b10be42c2b8ce1c9f97588330ab5f067834ff80`.
- The LFS pointer first added to this Git repository in `f483a82a94e00eea0b9f64875475db7d9fa41773`. The available Git history contains no later change to this RF path.

The checked-in `notebooks/originals/module-2.ipynb` records execution on **9 October 2026**, after that artifact was uploaded. Its saved output reports 260/399 correct and a differently sized serialized forest. A later rerun is not proof of the earlier artifact's holdout score. Different serialized sizes alone do not prove different prediction behavior, but neither run identity nor functional equivalence has been established here.

## September notebook output recovered

The public [original Module 2 notebook, version 1](https://www.kaggle.com/code/kuntalashivasairahul/module-2?scriptVersionId=351052482) was also recovered directly. Its cell execution metadata records 19 September, and its final saved output reports **267/400 = 66.75%**, macro F1 **0.537** from confusion counts, healthy recall **7/67**, and severe→healthy **9/185**. It reports the same 13,996,161-byte export. All six printed feature importances match the local forest at the reported six-decimal precision; the 200 trees, balanced class weights, seed 42 and stored training count 1,596 also agree. These observations strongly support association with deployment.

This corrects the initial audit's implication that 66.75% lacked a saved source. That audit had inspected the later October rerun and missed this September output. However, the recovered notebook does not record an export SHA-256 or retain original split/prediction rows, so exact run-to-artifact identity and the historical holdout cannot be fully replayed from these observations alone. The public page currently reports zero output files. The remaining useful handoff is the original `final_df_scaled.csv`, split/prediction files or working-output archive, if retained.

The public notebook's hardcoded trailing annotation mentions a baseline of 6 and SMOTE count of 20; its actual confusion matrices contain 9 and 23. The archive is preserved unchanged, and analysis uses the matrix counts. October grouped-field candidates remain separate experiments.

`public_september_run.json` records the recovered counts, consistency checks and notebook hash; `public_module2_v1.ipynb.txt` is the unchanged downloaded source with saved outputs, and `public_page_summary.txt` records the public page observation. `trace.json` includes notebook and model hashes, recorded notebook execution times, the local Git trace and the unresolved conclusion. `hosted_current.json`, `hosted_commits.json` and `hosted_rf_first_upload.json` preserve the public API observations. Their endpoints are respectively the model metadata API, commit list and pinned revision metadata for `rahulkuntala/cropsense-models`; all were read using gstack browse. The [Kaggle handoff checklist](../../KAGGLE_ARTIFACT_HANDOFF.md) lists the original files needed next.
