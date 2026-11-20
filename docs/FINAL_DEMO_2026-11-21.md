# Final demo preparation: 21 November 2026

Verified on **20 November 2026**. This is a packaging, runtime, deployment-comparison, and branch-maintenance checkpoint; no model retraining or new accuracy evaluation was performed.

## Branch roles

| Branch | Purpose |
|---|---|
| `main` | Current stable project baseline, advanced without rewriting history |
| `team/project-handoff` | Preserve the team handoff at `f2bc6214a11b4ab5edfb997dd8fbd7a0385469a4` |
| `deploy/streamlit-demo` | Final Windows demo and existing Streamlit deployment source |

Reusing the deployment branch avoids breaking the configured public app. The obsolete remote `codex/monday-demo-ready` and `feat/ui-repaint-multispectral` tips are ancestors of the handoff; removing those branch names does not remove their commits from the retained history. The old open Monday-demo PR is superseded by that retained history. Before cleanup, a complete Git bundle and a branch-tip inventory were saved locally under ignored `reports/final-demo-20261121/branch-backup/`. A hash inventory protects the author's 241 unfinished local paper/report files; these are excluded from this release. The separate local paper-review branch has unique commits and is retained locally.

## Application understanding

`app.py` presents three independent inputs and stores results per Streamlit session. `core.py` validates inputs and model probabilities, performs inference, preserves component warnings, and generates a traceable report. The equal-weight mean combines unrelated source datasets; it has no measured joint accuracy. The [team project guide](team/PROJECT_GUIDE.md) and [runtime contract](RUNTIME_CONTRACT.md) give the exact data flow.

CNN: raw 0–255 RGB, 224×224, 38 classes, embedded scaling. Canopy: paired registered RGB/grayscale NIR, six NDVI statistics, released Random Forest. Pest: four weeks × nine features, including scaled prior pest pressure, released LSTM. Setup and launch scripts already support x64 Windows with Python 3.12; no new launcher or dependency was needed.

## Fresh checks

| Check | Observed result |
|---|---|
| Working runtime vs deployment branch | No uncommitted application/model/dependency/script/test changes |
| Handoff vs deployed Git source | Only eight documentation/agent-entry files differ; application, model bytes, fixtures, notebooks, dependency pins, launchers, and tests are unchanged |
| Dependency consistency | Existing Python 3.12 environment: `pip check` passed |
| Regression suite | **28 tests passed**, including invalid inputs, Grad-CAM forward agreement, source invalidation, loader/inference failures and retry, offline model recovery, and Streamlit state contracts |
| Full verifier | `scripts/verify_environment.py --offline --full` passed with socket connections blocked; pinned packages, all release model hashes, reference outputs, auxiliary helper cache, and real-model smoke checks passed |
| Fresh cross-platform installation | [20 November CI run 44691622952](https://github.com/kuntalashivasairahul/CropSense/actions/runs/44691622952), at `0d3639f8b7b8c8a5b8105f87084185f1bfda2772`: Windows, macOS, Linux passed. Executable files are unchanged in this final-demo preparation. |
| Local browser | All four sections, severe scenario, Grad-CAM control, canopy imagery, pest replay, composite warnings, actual report download; changed pest week cleared the previous prediction/composite |
| Current hosted app | All four sections and severe scenario ran at `https://cropsense-ai-demo.streamlit.app/`; report download returned HTTP 200 |
| Hosted/local comparison | Downloaded reports were identical after removing only the generation timestamp; all three model hashes, source IDs, probabilities, component warnings, mean **0.177**, and **Severe Risk** matched |

The public app reports Streamlit **1.64.0**, matching the local pin. Public status does not expose a deployed Git SHA. Source equality, matching report/model identities and observed behavior establish runtime agreement for these checks; an owner-dashboard/build-log inspection of the exact deployed SHA was not performed. No later runtime change was found after October deployment; today's handoff adds documentation only.

The browser's shared console captured cloud-wrapper anonymous-user-detail 404s and iframe feature warnings. Inference and downloaded reports still worked; the initial local-only console check was clean. These wrapper diagnostics are not evidence of a failed model call. Screenshots, downloaded reports, command logs, environment report, and backup remain local under ignored `reports/final-demo-20261121/`.

The older `https://cropsense-btech.streamlit.app/` showed the inactivity sleep screen. Its historical configuration points at `main`; it was not woken or accepted as the presentation app. Use the current URL above.

## Remaining boundaries

- The physical Windows laptop has not been accessed or tested here. Complete the [Windows checklist](WINDOWS_DEMO.md) before calling that laptop ready, especially offline launch, caches, and DLLs.
- Released canopy healthy-fixture recognition remains **0/6**; pest performance remains **10/19**, with **0/5 medium recall**. The severe demonstration's recorded High pest week predicts Medium; this is a known model error, not a launcher failure.
- No new broad model benchmark, training, sensor calibration, cloud load/concurrency test, long-duration soak, or physical-device browser matrix was run. October performance figures retain their original scope/date.
- Source archives need online Python/dependency/helper setup once on the target laptop; copying the archive alone does not install Windows dependencies.
