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
| Preferred address | `cropsense-ai-demo.streamlit.app`, if available; otherwise use the assigned address |

The new public app was created at **https://cropsense-ai-demo.streamlit.app/** on 23 October. The older **https://cropsense-btech.streamlit.app/** still points to `main` and is preserved. Hosted acceptance is tracked below; creation alone does not establish that every check passed.

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

To roll back after a successful deployment, revert the faulty update on `deploy/streamlit-demo` with a normal commit and push it. Do not force-push or rewrite the release history. Recheck the prior model hashes and run the anonymous smoke checks after the rebuild. Before the first successful deployment, no rollback target is established.

## Current record — 23 October 2026

- Audit runtime/evidence checkpoint: `05b4fc4`.
- Local verification: 27 regression tests, dependency consistency and full offline inference checks passed.
- Deployment branch has been published. The [initial compatibility run](https://github.com/kuntalashivasairahul/CropSense/actions/runs/39875532734) passed Linux and macOS, while Windows stopped at `pip check` because the runner's preinstalled `pipx` requires a different `filelock` version. CI now uses an isolated app virtual environment and preserves dependency pins. The [final candidate run](https://github.com/kuntalashivasairahul/CropSense/actions/runs/39876107038), on `5b622c6`, passed Linux, Windows and macOS: installation, dependency checks, model preparation, full offline inference and all 27 regression tests. Subsequent documentation-only commits do not change that tested runtime or CI configuration.
- Local fresh-helper-cache preflight: first complete native analysis 12.40 seconds; ten warm analyses averaged 0.147 seconds (maximum 0.154); three parallel native analyses each completed in about 0.334 seconds with matching outputs. Peak native process RSS was 919.7 MiB. ImageNet weights and class-index JSON were cached successfully. These are Mac measurements, not hosted latency, cloud available memory or browser-session isolation results.
- Account sign-in succeeded and exposed the older `cropsense-btech` app on `main`, 24 commits behind the initial prepared release. It was preserved. The new app was created from release commit `1bf86ce`; initial cloud installation and a complete scenario succeeded. Cloud-specific dependency/font fixes and server memory logging now require fresh CI and hosted acceptance. Build logs and execution records are saved locally under the ignored deployment report folder.
- Existing source/model contracts are unchanged by deployment preparation. Historical model-quality and sensor-calibration limitations remain open.

Machine-readable execution records and raw local/CI logs belong under ignored `reports/deployment-2026-10-23/`; update this record with verified public CI/deployment links when available. The machine-readable release record identifies the current commit and separates native preflight evidence from hosted acceptance.
