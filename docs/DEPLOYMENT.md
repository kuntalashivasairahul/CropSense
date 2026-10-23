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

The [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) describes the repository, branch, entrypoint and Python settings. The owner signs in and grants GitHub access in the browser; passwords, OAuth credentials and verification codes must not enter the repository or logs. No Kaggle credentials, GPU, paid plan or inference API key is required.

## Release gates

1. Commit the reviewed source, tests and required evidence on the deployment branch. Exclude environments, secrets and generated `reports/` files. Keep model weights and their manifest hashes unchanged.
2. Run the existing **Demo compatibility** workflow on that exact commit. Require the Linux job to pass installation, `pip check`, checksummed model preparation, full offline inference verification and all regression tests. The workflow also exercises Windows and macOS.
3. In Community Cloud, create an app using the configuration above. Keep prototype warnings visible. A successful build or health response alone is not acceptance.
4. Test anonymously: disease inference and Grad-CAM; paired RGB/NIR and manual NDVI; pest sequence selection; composite warnings and the actual downloaded report. Exercise invalid/oversized inputs and a visible model-failure/retry path.
5. Test three separate Streamlit sessions concurrently with different scenarios. Confirm each retains its own selected input, probabilities and report. Repeat ten analyses and record cold-start time, warm inference time, observed memory and any failures. Check desktop/mobile fonts, charts and controls.

Model files are LFS-tracked. The existing checksummed `models/release_bundle.zip` restores missing/pointer files locally, without relying on LFS downloads or an unpinned remote model. The stock ImageNet helper and its class-index JSON are cached on first use; a fresh host filesystem requires outbound HTTPS for those public assets. Setup checks exercise both before offline verification. Do not replace these checks with a silent fallback model.

## Operation and recovery

Community Cloud rebuilds from the selected GitHub branch. Prepare updates on a candidate branch, pass compatibility checks, then update the deployment branch to the tested commit. Never push credentials, local environments or raw user uploads. The application uses session state and downloadable reports; it has no application database or upload-history store.

Free apps can sleep after inactivity, and [resource limits can change](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resources-and-limits). Visit the app before a presentation, wait for wake-up, and exercise all three inference modules once. Retain the verified local/Windows package as a presentation fallback.

Use **Manage app → logs** for installation, startup and inference failures. Save the deployed commit, final app URL, CI run URL and measured acceptance results. A resource-limit or dependency failure blocks release; do not remove modules or enable billing to bypass the gate.

To roll back after a successful deployment, revert the faulty update on `deploy/streamlit-demo` with a normal commit and push it. Do not force-push or rewrite the release history. Recheck the prior model hashes and run the anonymous smoke checks after the rebuild. Before the first successful deployment, no rollback target is established.

## Current record — 23 October 2026

- Audit runtime/evidence checkpoint: `05b4fc4`.
- Local verification: 27 regression tests, dependency consistency and full offline inference checks passed.
- Deployment branch has been published; GitHub compatibility CI is running.
- Live URL, cloud build logs, hosted timings/memory and three-session acceptance: pending account sign-in and actual deployment. Do not present the preferred address as a live app.
- Existing source/model contracts are unchanged by deployment preparation. Historical model-quality and sensor-calibration limitations remain open.

Machine-readable execution records and raw local/CI logs belong under ignored `reports/deployment-2026-10-23/`; update this record with verified public CI/deployment links when available.
