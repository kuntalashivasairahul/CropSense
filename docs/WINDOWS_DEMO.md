# Windows laptop: final demo on 21 November 2026

Use **`deploy/streamlit-demo`**, also used by the [public app](https://cropsense-ai-demo.streamlit.app/). `team/project-handoff` remains the team's reference. `main` is the current stable baseline. The old October archives and Monday branch are historical; use the final-demo ZIP or branch below.

Prepare the laptop **while online on 20 November**, then rehearse without Wi-Fi. Windows/macOS/Linux CI passed on 20 November for the same application, weights, dependency pins, and scripts. That does not establish readiness on the physical college laptop. See [verification and limits](FINAL_DEMO_2026-11-21.md).

## Obtain a fresh copy

Option A: copy `CropSenseAI-final-demo-2026-11-21.zip` and its `.sha256` file to the laptop. In PowerShell, run:

```powershell
Get-FileHash .\CropSenseAI-final-demo-2026-11-21.zip -Algorithm SHA256
Get-Content .\CropSenseAI-final-demo-2026-11-21.zip.sha256
```

The hashes must match. Extract to a new short path such as `C:\CropSenseAI-demo`. Work in the extracted folder containing `app.py` and `setup_windows.cmd`, not inside the ZIP preview. The package contains source, samples, notebooks, evidence, and the checksummed model recovery bundle. It excludes Git history, environments, credentials, and unfinished local paper/report edits.

Option B: if Git is installed, clone this branch into a new folder. Disabling LFS download filters avoids an unnecessary network dependency; setup restores the three exact model files from the included ZIP:

```powershell
git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false clone --single-branch --branch deploy/streamlit-demo https://github.com/kuntalashivasairahul/CropSense.git CropSenseAI-demo
cd CropSenseAI-demo
git branch --show-current
```

Expected branch: `deploy/streamlit-demo`. Keep existing working copies separate.

## Set up once while online

1. Install **64-bit Python 3.12** with the Windows Python launcher. Check `py -3.12 --version`. Do not substitute another Python minor version or a Windows ARM interpreter.
2. Double-click `setup_windows.cmd`. Leave its window open until it says setup is complete. It creates a machine-local `.venv`, installs pinned dependencies, restores/checks model hashes, caches the public ImageNet helper and class labels, then verifies inference with network connections blocked. TensorFlow installation can take time. No Kaggle account or full dataset download is required.
3. In PowerShell in that project folder, run:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_environment.py --offline --full
```

Require **28 passing tests**, a verifier `PASS`, and `reports\environment-windows.json` containing `"status": "passed"`, `"offline": true`, and `"full": true`. Save a copy of that full report before launch; the launcher's shorter check overwrites the same report file.

## Rehearse with Wi-Fi off

For manual uploads, open the project's [demo input folder](../demo_inputs/README.md): five leaf examples and three matching RGB/NIR pairs are grouped by module. Pest uses the built-in numeric sequence selector; Composite needs no separate image. Choose **Manual / Custom inputs** before uploading.

1. Disconnect Wi-Fi. Double-click `start_windows.cmd`, keep the terminal open, and open `http://localhost:8501`. A successful launch needs the setup cache under the same Windows user account.
2. Select **Scenario 3 · Severe Combined Risk**. Visit all four sections. Expected fixture results: Potato Early Blight about **99.46%** softmax confidence, canopy severe stress **81.50%** class confidence / **0.100** assigned score, pest medium **43.80%** class confidence / **0.570** expected pressure, composite mean **0.177** / **Severe Risk**. These are reproducible fixture outputs, not benchmark accuracy.
3. Switch the leaf view to **Grad-CAM saliency**. Explain influence rather than verified lesion localization. Load the non-leaf scenario and confirm the warning, then return to a valid leaf.
4. In Canopy stress, run **Healthy**, **Moderate**, and **Severe** paired samples and inspect the real heatmap and six statistics. The known healthy-fixture failure remains visible; do not promise that every button's label equals the model prediction. A colour photo is not a NIR band.
5. In Pest risk, change the selected week: the old result and completed composite must clear. Run the forecast again. This is a saved historical replay, with prior pest pressure among nine features, not live weather forecasting.
6. In Composite, download and open the text report. Check inputs/source week, three model hashes, warnings, and simulation language. A severe component must remain visible even when the mean is high.
7. Close/reopen the app once while still offline. Keep the verified folder and ZIP on the laptop; do not depend on college Wi-Fi or the free cloud app waking immediately.

## If setup or launch fails

- If `py -3.12` is unavailable, install the x64 Python 3.12 interpreter and its launcher, then rerun setup.
- Native Windows uses CPU inference. For TensorFlow DLL-load errors, install Microsoft's official [Visual C++ x64 Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist), then retry.
- If `.venv` belongs to another machine/OS, rename it for recovery and rerun setup; do not copy a Mac environment.
- If a model hash or offline helper check fails, rerun setup online. Keep the error output and environment report; do not replace model files or loosen dependency pins.
- If port 8501 is occupied, close the previous demo terminal and retry. Keep the current public URL as a secondary option, not the primary offline plan.
