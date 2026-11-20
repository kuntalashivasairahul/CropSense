# CropSenseAI agent entry point

This checkout includes the team mini-project handoff and final demo. Read these before explaining or changing the project:

1. `docs/team/README.md` — scope, reading order, setup, and included material.
2. `docs/team/PROJECT_GUIDE.md` — architecture, code symbols, inputs, and limitations.
3. `docs/RUNTIME_CONTRACT.md` — authoritative interpretation of the current runtime.
4. `notebooks/README.md` and `docs/team/KAGGLE_GUIDE.md` — notebook versions, data, and evidence.
5. `docs/team/VERIFICATION.md` — what was checked and what remains unverified.

Use current source and `models/artifact_manifest.json` for runtime facts. Older paper/deck exports, notebook outputs, and candidate metrics describe their own snapshots. Distinguish inspected code, reproduced behavior, historical measurements, and assumptions. Do not invent team contributions or describe the simulated composite as measured fusion accuracy.

Key contracts:

- `app.py` is the Streamlit interface; `core.py` owns inference, validation, assessment, reports, and model loading.
- CNN input is raw 0–255 RGB, resized to 224×224; preprocessing is already in the saved graph.
- Canopy inputs are aligned RGB and grayscale NIR; use the six unscaled features in `core.NDVI_FEATURES` order.
- The released pest model takes four weeks × nine features, including prior scaled pest pressure. It is not weather-only inference.
- Use `core.composite_assessment` for the displayed warning tier and report. The mean must not hide a severe component or uncertain leaf.
- Preserve validation, source IDs, model hashes, input-change invalidation, and error handling when editing.
- Model files are Git LFS objects. `scripts/release_assets.py` restores exact files from the checked local ZIP; keep manifest and bundle consistent. Never replace a release with a candidate merely because it loads.

For changes, inspect callers and the relevant tests first. Use the existing setup scripts and pinned Python 3.12 environment. Run `python -m unittest discover -s tests -v` and `python scripts/verify_environment.py --offline --full` with that environment when runtime code changes. Update relevant documentation. Keep credentials, `.venv`, generated reports, and raw dataset downloads out of Git. Do not push later changes unless the teammate asks.
