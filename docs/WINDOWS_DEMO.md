# Windows 10 demonstration checklist

Use the laptop before presentation day. The Mac check has passed; native Windows is complete only after this laptop passes.

1. Confirm **64-bit Python 3.12**. Version 3.12.2 is acceptable.
2. In PowerShell or Command Prompt, run `git clone -b codex/monday-demo-ready https://github.com/kuntalashivasairahul/CropSense.git` and open the new `CropSense` folder. If already cloned, switch to that branch and pull its latest commit. Do not copy the Mac `.venv` folder.
3. Double-click `setup_windows.cmd` while online. It creates a fresh `.venv`, installs pinned dependencies, checks model hashes, and caches the ImageNet helper. Save any error text.
4. Check `reports/environment-windows.json` says `"status": "passed"`.
5. Disconnect Wi-Fi, then double-click `start_windows.cmd`. It repeats the offline check and opens the local Streamlit app.
6. Run a PlantVillage leaf, inspect the class and Grad-CAM, then show a field-photo limitation example if useful.
7. Load all three paired RGB/NIR sample buttons. Confirm a real heatmap and statistics. Try mismatched dimensions and check the clear error. A colour photograph is not a valid NIR band.
8. Run a pest sequence, change the selection, and confirm the old result clears.
9. Check the composite score appears after all three modules have run.
10. Keep a local copy of the repo and this checklist on the laptop for Monday.

Windows TensorFlow uses CPU inference. If a DLL error appears, install the official [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) and rerun setup. If a model hash fails, send the report; do not hand-copy an unverified model.
