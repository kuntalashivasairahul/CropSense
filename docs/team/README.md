# Team mini-project handoff

This is the shared learning branch: **`team/project-handoff`** in [kuntalashivasairahul/CropSense](https://github.com/kuntalashivasairahul/CropSense/tree/team/project-handoff).

Prepared on **20 November 2026**, starting from published application commit `103256f508a4dafa95e4950adce38b90674934a5` on `deploy/streamlit-demo`. The handoff adds documentation and agent instructions; application code, weights, samples, notebooks, and existing tests are inherited from that commit. This branch does not become the deployment branch automatically.

## Start here

Give your PC's Codex agent the complete [onboarding prompt](CODEX_PROMPT.md). It tells the agent which branch to clone and how to teach the project from the actual files. Git is sufficient for reading the material; Python and the large ML dependencies are needed only to run the app. No Kaggle account is needed to understand the local notebook sources and saved evidence.

To clone manually into a new folder:

```sh
git clone --single-branch --branch team/project-handoff https://github.com/kuntalashivasairahul/CropSense.git CropSense-team
cd CropSense-team
git branch --show-current
```

Expected branch: `team/project-handoff`. Preserve an existing working folder rather than resetting it. If Git LFS downloads fail, clone into a different new folder using:

```sh
git -c filter.lfs.process= -c filter.lfs.smudge= -c filter.lfs.required=false clone --single-branch --branch team/project-handoff https://github.com/kuntalashivasairahul/CropSense.git CropSense-team-no-lfs
```

That clone retains model pointers. The normal setup restores their exact bytes from the included ZIP, without a model download. You can also restore them with standard-library Python from the repository root:

```sh
python -c "from scripts.release_assets import ensure_local_models; ensure_local_models()"
```

## Reading order

| Order | Material | What you should understand afterward |
|---|---|---|
| 1 | [Project guide](PROJECT_GUIDE.md) | The problem, three modules, data flow, code responsibilities, and demo limits |
| 2 | [Runtime contract](../RUNTIME_CONTRACT.md), then `core.py` and `app.py` | Exact inputs, scores, warnings, state handling, and report behavior |
| 3 | [Notebook index](../../notebooks/README.md) and [Kaggle guide](KAGGLE_GUIDE.md) | Training versus inference, datasets, original versus audit notebooks, and release versus candidates |
| 4 | [Model evaluation](../MODEL_EVALUATION.md) and linked evidence | What each metric measures, which artifact it belongs to, and unresolved provenance |
| 5 | [Verification record](VERIFICATION.md) and local setup below | What was actually checked and how to reproduce checks on your PC |

Read the older paper, slides, and academic report after the current runtime guide. They are presentation/reference material and may describe older behavior. Local unfinished paper/report revisions from the author's working directory were excluded from this handoff.

## What the branch includes

| Location | Included material |
|---|---|
| `app.py`, `core.py` | Full Streamlit application and shared inference/report logic |
| `models/` | Three released models, artifact manifest, checksummed recovery ZIP, 18 paired canopy samples, 19 pest sequences |
| `sample_images/` | Five leaf/field/non-leaf demonstration fixtures |
| `notebooks/` | All nine notebooks: three audit sources, three original exports with outputs, three older Kaggle snapshots; six Kaggle metadata files |
| `docs/evidence/` | Metrics, confusion counts/plots, split and provenance records, saved source records, recovery checks, and screenshots |
| `tests/`, `test_core.py` | Regression contracts, reference predictions, and real-model smoke checks |
| `scripts/`, setup/start scripts | Environment setup, exact model recovery, offline verification, and release packaging |
| `requirements.txt`, `requirements/constraints.txt` | Pinned runtime dependencies |
| `.github/workflows/compatibility.yml` | Windows/macOS/Linux compatibility workflow |
| `static/`, `.streamlit/` | Local fonts and Streamlit configuration |
| `research_paper/`, existing report/deck files in `docs/` | Committed historical academic material; check claims against current code |

Full training datasets and the author's Python environment are not included. Dataset sources and access instructions are in the Kaggle guide. Setup installs dependencies and caches a public ImageNet helper online; the prepared app can then run offline. The helper is separate from the three bundled project models.

## Run on your own PC

Use **64-bit Python 3.12**. Keep the repository path reasonably short and create a local environment on this machine; copying the author's `.venv` will not work across systems.

| System | Setup, once while online | Launch |
|---|---|---|
| Windows | Double-click `setup_windows.cmd` | Double-click `start_windows.cmd` |
| Apple Silicon Mac | `./setup_mac.sh` | `./start_mac.sh` |
| Linux | `python3.12 scripts/setup_environment.py` | `.venv/bin/python -m streamlit run app.py` after the checks below |

The Mac setup explicitly targets Apple Silicon. Intel macOS is not covered by that script. Native Windows uses CPU inference. Open `http://localhost:8501` when Streamlit starts.

After setup, run these from the repository root in PowerShell on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\verify_environment.py --offline --full
```

On macOS/Linux:

```sh
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_environment.py --offline --full
```

Expect all tests to pass and `reports/environment-<system>.json` to report `"status": "passed"`. Generated reports stay local. If the helper tries to download during an offline check, rerun setup online with the same OS account and cache location. Record errors rather than removing version pins or substituting model files.

For a presentation, follow [Windows demo checks](../WINDOWS_DEMO.md) using **this branch**; its older archive/branch references are historical. Exercise all three modules, change inputs to check results clear, inspect the simulated composite, and download its report. Passing on the author's Mac or CI does not establish readiness on your PC.

## Working together

Keep this branch as a shared reference. For a later assigned change, make a personal branch from it, document the change and checks, and submit a reviewable pull request. Agree team roles among yourselves; repository ownership and saved notebook ownership do not establish who did each piece of work.
