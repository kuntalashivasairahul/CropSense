# Windows 11 demonstration checklist

Use the laptop before presentation day. The Mac mini and a clean Windows Server 2025 GitHub runner have passed the offline inference suite; native **Windows 11 laptop** readiness is complete only after this laptop passes. The [runner reports](evidence/compatibility/) record the checked versions and model hashes for the 10 October snapshot. The 22 October audit fixes require a fresh run of this checklist on the branch that contains them; the current local Mac checks do not establish Windows readiness.

1. Confirm **64-bit Python 3.12**. Version 3.12.2 is acceptable.
2. For the **22 October audit fixes**, use the provided `CropSenseAI-audit-fixes-20261022-r4.zip` and its `.sha256` file. Compare the archive with `Get-FileHash .\CropSenseAI-audit-fixes-20261022-r4.zip -Algorithm SHA256`, then extract it into a new folder. The ZIP includes source, model weights, recovery bundle and a file-hash manifest; it excludes `.git` and the Mac `.venv`. The local fix branch has not been pushed, so cloning `codex/monday-demo-ready` alone retrieves the earlier snapshot. Keep any existing working copy separate.
3. Double-click `setup_windows.cmd` while online. It creates a fresh `.venv`, installs pinned dependencies, checks model hashes, and caches the ImageNet helper. Save any error text.
4. From the extracted project folder, run `.venv\Scripts\python.exe -m unittest discover -s tests -v` and `.venv\Scripts\python.exe scripts\verify_environment.py --offline --full`. Save both command outputs and check `reports/environment-windows.json` says `"status": "passed"`. The current Mac result is 27 passing unittest cases; a ZIP replay on the Mac does not replace this Windows run.
5. Disconnect Wi-Fi, then double-click `start_windows.cmd`. It repeats the offline check and opens the local Streamlit app.
6. Run a PlantVillage leaf, inspect the class and Grad-CAM, then show a field-photo limitation example if useful.
7. Load all three paired RGB/NIR sample buttons. Confirm a real heatmap and statistics. Try mismatched dimensions and check the clear error. A colour photograph is not a valid NIR band.
8. Run a pest sequence, change the selection, and confirm the old result clears.
9. Check the simulated mean index and component warning tier appear after all three modules run. A severe component must not be cancelled by the mean. Flagged/low-confidence leaf inputs must show Needs review. Download the report and confirm source IDs, model hashes and warnings are included.
10. If you want to show the Agriculture-Vision source data, download and hash-check the mini archive using the [Kaggle walkthrough](KAGGLE_NOTEBOOK_PRESENTATION.md#agriculture-vision-mini-archive-for-local-use). Extract it to a folder outside the Git repo and show a matching RGB/NIR pair. The app uses its bundled pairs, so this archive is not needed to run it.
11. Keep a local copy of the repo and this checklist on the laptop for Monday.

Windows TensorFlow uses CPU inference. If a DLL error appears, install the official [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) and rerun setup. If a model hash fails, send the report; do not hand-copy an unverified model.
