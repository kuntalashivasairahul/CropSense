# Showing the executed Kaggle audit notebooks

**Checked 11 October 2026.** Sign in to the `kuntalashivasairahul` Kaggle account, open each link, select the completed version below, and scroll to **Presentation evidence from this saved run** near the end. The code computes those tables from the run's exported CSV/JSON files. The preceding cells show data preparation, per-class reports, confusion plots and export checks; Module 1 v3 and Modules 2/3 also show model selection, while Module 1 v4 reloads the already selected release checkpoint. The **Output** tab offers `REPORT.md`, `results_summary.csv`, confusion PNG/CSV files, split manifests, model contracts and predictions. Kaggle's account URL and version history identify the owner; the printed owner label alone is not proof.

| Notebook | Saved version and run ID | What to show |
|---|---|---|
| [Module 1 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-1-audit) | **v4**, `20261011T102617Z` | Direct verification of the locked release artifact, SHA-256 `5c263a58…d0d8deb`. 38 PlantVillage classes, 54,305 images; 46,160 training and 8,145 validation. Largest Orange Huanglongbing 5,507, smallest Potato healthy 152. Original checkpoint: **78/252, 30.95%** PlantDoc test, **98.10%** PlantVillage validation. Released adapted checkpoint: **141/252, 55.95%**, macro F1 **0.552**, and **96.83%** validation. Show the per-class report and 38-class confusion counts. |
| [Module 2 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-2-field-splits-rf-and-metrics) | **v4**, `20261011T085857Z` | 8,308 accepted tiles, grouped by inferred field into 5,175 training, 1,584 validation and 1,549 test. Retrained baseline: **58.42%**, macro F1 **0.474**, healthy recall **23/280**, severe→healthy **74/700**. Threshold candidate: **57.84%**, macro F1 **0.477**, severe→healthy **81/700**; rejected. The older **66.75%** is from a different random-tile protocol in the [original Module 2 notebook](../notebooks/originals/module-2.ipynb). |
| [Module 3 audit](https://www.kaggle.com/code/kuntalashivasairahul/cropsense-module-3-causal-lstm-and-metrics) | **v2**, `20261011T085851Z` | Train 2005–2007, validate 2008, test 2009; 19 observed-target test weeks. LSTM **10/19, 52.63%**, macro F1 **0.433**; persistence **8/19, 42.11%**; logistic regression **13/19, 68.42%**. Show class report and confusion counts beside the stronger simple baseline. |

Kaggle CLI reported **Complete** for all three versions. Their logs reached the final evidence cells and Kaggle generated executed `__notebook__.ipynb` and HTML renders. The latest pulled source of each notebook matches the repository source. Checked-in notebook files are source copies and have **no embedded output cells**. The private Kaggle pages could not be visually inspected through the unauthenticated browser here, so open the saved versions in your signed-in Chrome account before presenting and check that the rendered outputs are visible. If Kaggle opens the editor, switch to the completed saved version.

## Module 1 release score versus fresh rerun

The **locked demo model** came from the earlier Module 1 training audit and was independently re-evaluated in **v4** without retraining: `141/252 = 55.95%` PlantDoc test accuracy, macro F1 `0.552`, and `96.83%` PlantVillage validation. Its SHA-256 in the v4 output exactly matches the [release manifest](../models/artifact_manifest.json); the earlier exported [metrics](evidence/module-1-candidate/adapted_checkpoint__field_test_metrics.json) and [confusion counts](evidence/module-1-candidate/adapted_checkpoint__field_test_confusion_counts.csv) are also preserved. To show fine-tuning and validation-based selection, choose **v3** in the same notebook's version history. That fresh run trained a **different candidate** and scored `139/252 = 55.16%`, macro F1 `0.539`, and `96.77%` validation, with SHA-256 `82c0a884fed6b5463306600003a0d5fc86d4a0edcf2dca3a6af3278221ea4a7b`. It did **not** replace the demo model. State the score and hash for the version being shown. Both PlantDoc tests were historically inspected and are not new blind holdouts.

The new Module 3 run also exported another candidate file even though its test percentages matched the earlier run. The Windows demo still uses the checked [release bundle](../models/artifact_manifest.json). The composite dashboard has no labelled joint dataset and thus no accuracy or confusion matrix.

## Agriculture-Vision mini archive for local use

The Module 2 audit downloads `data2017_miniscale.tar.gz` from the [public source archive](https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_paper_2020/Dataset/data2017_miniscale.tar.gz). A HEAD request on 11 October returned **1,879,589,036 bytes** (about 1.75 GiB). The v4 run recorded SHA-256 `2b4bf0b2357ba982fe6d81da55841874afdd721a867bb8fddede9e04e4914027`. To download outside the Git repository on the Windows laptop, run in PowerShell:

```powershell
$archive = Join-Path $env:USERPROFILE 'Downloads\data2017_miniscale.tar.gz'
curl.exe -L --retry 3 -C - -o $archive 'https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_paper_2020/Dataset/data2017_miniscale.tar.gz'
(Get-FileHash $archive -Algorithm SHA256).Hash.ToLower()
```

Compare the printed hash with the value above before extracting. Keep several additional gigabytes free for extraction. This is the **mini 2017 archive**, not all Agriculture-Vision data; it is sufficient for the documented Module 2 audit. The Kaggle notebook uses `/kaggle/temp` paths, so a local training rerun needs those paths changed. The Windows **app demonstration does not need the archive**: its paired sample images and verified model are already in the release bundle.
