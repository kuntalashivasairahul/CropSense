# Public demo deployment

Target: free Streamlit Community Cloud, public access, initially 1–3 simultaneous viewers. This is an academic prototype; deployment does not validate field accuracy or the simulated composite index.

## Configuration

| Setting | Value |
|---|---|
| GitHub repository | `kuntalashivasairahul/CropSense` |
| Deployment branch | `deploy/streamlit-demo` |
| Entrypoint | `app.py` |
| Python | 64-bit Python 3.12 |
| Dependencies | Root `requirements.txt`, including `requirements/constraints.txt` |
| Visibility | Public |
| Secrets | None |
| Live address | https://cropsense-ai-demo.streamlit.app/ |

The public app was deployed and exercised on 23 October 2026. The older **https://cropsense-btech.streamlit.app/** still points to `main` and is preserved; it was 24 commits behind the initial prepared release. Use the new address for the audited runtime.

The [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) describes the repository, branch, entrypoint and Python settings. The owner signs in and grants GitHub access in the browser; passwords, OAuth credentials and verification codes must not enter the repository or logs. No Kaggle credentials, GPU, paid plan or inference API key is required.

## Release gates

1. Commit the reviewed source, tests and required evidence on the deployment branch. Exclude environments, secrets and generated `reports/` files. Keep model weights and their manifest hashes unchanged.
2. Run the existing **Demo compatibility** workflow on that exact commit. Require the Linux job to pass installation, `pip check`, checksummed model preparation, full offline inference verification and all regression tests. The workflow also exercises Windows and macOS.
3. In Community Cloud, create an app using the configuration above. Keep prototype warnings visible. A successful build or health response alone is not acceptance.
4. Test anonymously: disease inference and Grad-CAM; paired RGB/NIR and manual NDVI; pest sequence selection; composite warnings and the actual downloaded report. Exercise invalid/oversized inputs and a visible model-failure/retry path.
5. Test three separate Streamlit sessions concurrently with different scenarios. Confirm each retains its own selected input, probabilities and report. Repeat ten analyses and record cold-start time, warm inference time, observed memory and any failures. Check desktop/mobile fonts, charts and controls.

Model files are LFS-tracked. The existing checksummed `models/release_bundle.zip` restores missing/pointer files locally, without relying on LFS downloads or an unpinned remote model. The stock ImageNet helper and its class-index JSON are cached on first use; a fresh host filesystem requires outbound HTTPS for those public assets. Setup checks exercise both before offline verification. Do not replace these checks with a silent fallback model.

Community Cloud's initial build forcibly replaced PyArrow 25.0.1 with 24.0.0 and upgraded `certifi` to 2026.10.19. Linux constraints now explicitly match those hosted versions; verified macOS/Windows pins remain unchanged. The [Arrow issue](https://github.com/apache/arrow/issues/50471) explains the original thread-initialization fault; the cloud enforces its own broad version cap even though the pinned desktop patch is newer. Relative font URLs support the hosted app's iframe prefix. Server-only `CROPSENSE_MEMORY` log records contain event names, Linux RSS and available cgroup usage/limit values, without images, input values or session identifiers; inaccessible OS metrics are skipped.

## Operation and recovery

Community Cloud rebuilds from the selected GitHub branch. Prepare updates on a candidate branch, pass compatibility checks, then update the deployment branch to the tested commit. Never push credentials, local environments or raw user uploads. The application uses session state and downloadable reports; it has no application database or upload-history store.

Free apps can sleep after inactivity, and [resource limits can change](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resources-and-limits). Visit the app before a presentation, wait for wake-up, and exercise all three inference modules once. Retain the verified local/Windows package as a presentation fallback.

Use **Manage app → logs** for installation, startup and inference failures. Save the deployed commit, final app URL, CI run URL and measured acceptance results. A resource-limit or dependency failure blocks release; do not remove modules or enable billing to bypass the gate.

The known-good runtime is `7356f0f947ee051ac38639870e70856c24afe3b9`. To roll back, revert subsequent faulty runtime commits on `deploy/streamlit-demo` with normal commits and push them. Do not force-push or rewrite release history. Confirm the branch/commit in the build logs; if an update remains stale, use **Manage app → Reboot app** and wait for startup. Recheck model hashes and anonymous inference/report downloads after the rebuild. A reboot clears in-memory sessions and caches; users should download reports before maintenance.

## Current record — 23 October 2026

- Audit runtime/evidence checkpoint: `05b4fc4`.
- Tested and deployed runtime: [`7356f0f947ee051ac38639870e70856c24afe3b9`](https://github.com/kuntalashivasairahul/CropSense/commit/7356f0f947ee051ac38639870e70856c24afe3b9). Later documentation-only commits do not change that runtime or CI configuration.
- The [final compatibility run](https://github.com/kuntalashivasairahul/CropSense/actions/runs/39879798310) passed Linux, Windows and macOS: installation, dependency consistency, checksum restoration, full offline inference and **28 regression tests per platform**. All three uploaded environment reports show `offline: true`, `full: true`, `status: passed` and unchanged model hashes. Local regression checks also passed. The initial runner-global `pipx` conflict was resolved by using an isolated app virtual environment.
- Local fresh-helper-cache preflight: first complete native analysis 12.40 seconds; ten warm analyses averaged 0.147 seconds (maximum 0.154); three parallel native analyses each completed in about 0.334 seconds with matching outputs. Peak native process RSS was 919.7 MiB. ImageNet weights and class-index JSON were cached successfully. These are Mac measurements, not hosted latency, cloud available memory or browser-session isolation results.
- Cloud fixes retain all modules and weights: Linux dependency markers match the host, relative font paths work inside its iframe, the hidden toolbar no longer intercepts navigation, and report downloads avoid a rerun that invalidated their media URLs. Initial model restoration and ImageNet helper/class-index downloads completed; subsequent requests used caches. No secrets or paid hosting were configured.
- Existing source/model contracts are unchanged by deployment preparation. Historical model-quality and sensor-calibration limitations remain open.

### Hosted acceptance and measurements

| Check | Observed result |
|---|---|
| Public access | Three separate fresh browser profiles opened the app without sign-in. |
| Four sections | Leaf inference and Grad-CAM, paired RGB/NIR NDVI and imagery, pest replay/chart, simulated composite and warnings rendered. |
| Concurrent scenarios | Three profiles started distinct scenarios together; each completed in **7.47–7.59 seconds** and retained its own leaf, canopy input and pest source week. All three actual report downloads returned HTTP 200 with matching inputs and distinct report hashes on `7356f0f`. |
| Repeated inference | Ten warm leaf analyses completed: mean **2.47 seconds**, range **2.34–2.86 seconds**. Browser click-to-result timing includes UI/network overhead. This series used `ba43b72`; the final change only prevents report-download reruns. |
| Cold workflow | First full scenario after reboot completed in **17.67 seconds**. This measures scenario selection to results, not total cloud provisioning or a guaranteed wake-up time. |
| Upload validation | A 1,049,600-pixel PNG, a non-image `.png` and an 11 MiB upload showed visible errors. Rejected inference cleared the old prediction. A subsequent valid leaf prediction recovered in **2.51 seconds**. |
| Desktop/mobile | Fonts, navigation, result imagery and charts checked at desktop size and 390×844. Mobile navigation scrolls horizontally. This was Chromium viewport testing, not a physical-phone/browser matrix. |
| Memory | Observed container limit **3 GiB**. Peak logged process RSS **1,325.7 MiB**; peak aggregate cgroup usage **3,071.6 MiB**, including cache and other container usage. No OOM or failed inference was observed in these runs, but aggregate usage approached the limit. |

These results support a small presentation demo, not a sustained capacity guarantee. Wake and test the app before presenting; keep the local package available. More simultaneous viewers, long-duration stability, inactivity wake-up timing and a live model-file failure were not tested. A separate local AppTest injected a model-loader failure: the error was visible, no old prediction remained, and retry recovered. Inference-failure regression checks also passed. Anonymous host-wrapper requests for user details/analytics returned 404s; they did not prevent inference or the final downloads. A fresh navigation check had no console errors. Earlier font and report errors were corrected and retested.

The compact public [validation record](evidence/deployment-2026-10-23/validation.json) identifies the runtime, checks and measurements. Raw build logs, CI reports, timings, screenshots and downloaded test reports are retained locally under ignored `reports/deployment-2026-10-23/`. Browser authentication state and raw session identifiers are excluded from published evidence.
