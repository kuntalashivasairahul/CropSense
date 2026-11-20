# Handoff verification

Prepared **20 November 2026** for `team/project-handoff`, based on application commit `103256f508a4dafa95e4950adce38b90674934a5`. These are engineering/package checks, not new model evaluation or field-validation results.

## Completed before publication

| Check | Result |
|---|---|
| Source boundary | Application, models, notebooks, fixtures, dependency pins, and tests inherited unchanged from the published application commit; only handoff documentation and agent instructions added |
| Runtime dependencies | `python -m pip check`: no broken requirements; full verifier also checked the installed applicable dependency pins |
| Regression suite | `python -m unittest discover -s tests -v`: **28 passed**, including Streamlit AppTest cases |
| Full runtime | Existing `scripts/prepare_demo.py` completed, including `scripts/verify_environment.py --offline --full`: **passed**, with socket connections blocked during offline verification |
| Release assets | SHA-256 and sizes of all three models and sample-sequence JSON matched `artifact_manifest.json`; recovery ZIP SHA-256, ZIP CRC, and bundled manifest asset hashes matched |
| Notebook package | All **nine** notebooks parsed as notebook-format 4; all **six** metadata `code_file` targets existed; all **three** original export hashes matched `docs/evidence/notebook_sources.json` |
| Fresh clone and recovery | Cloned `team/project-handoff` from local Git transport with all LFS download filters disabled. The clone started with all three model pointers; `ensure_local_models()` restored their exact hashes from the local ZIP. All nine notebooks and handoff files were present. Full offline inference then **passed** from that clone using the existing pinned environment/cache. |
| Handoff navigation | All **62** relative link targets in the eight handoff entry/documentation files existed; Markdown code fences were balanced |
| Outgoing content | Checked packaged filenames for local environments/credentials, scanned text for high-confidence private-key/token patterns, and checked GitHub file-size limits; no findings from those checks |

Checks ran on an **Apple Silicon Mac, Python 3.12**, using the author's existing pinned environment with source loaded from the isolated handoff worktree. The first sandboxed test attempt could not reuse the Keras helper cache and attempted a blocked download. Running the existing setup verification and regression suite with normal cache access passed. No application or test changes were made to obtain the pass.

The recorded diagnostics still show **0/6** bundled healthy canopy labels matched and **10/19** pest predictions correct. A successful contract check does not repair those quality limits.

## Reproduce on a teammate PC

Use the setup and OS-specific commands in [the handoff guide](README.md#run-on-your-own-pc). The meaningful final checks are:

```text
python -m pip check
python -m unittest discover -s tests -v
python scripts/verify_environment.py --offline --full
```

Here `python` means the interpreter in that machine's newly created `.venv`. Keep the resulting local environment report and error output if any check fails.

## Boundaries

- A fresh dependency installation on a teammate's physical PC has not been performed here. Windows/macOS/Linux CI is useful evidence for its own runner environments, not that PC.
- No new Kaggle training, full-dataset download, independent labelled field test, or sensor calibration was performed. Hosted Kaggle pages and the public Streamlit demo were not rechecked for this handoff; local evidence keeps its original dates.
- Credential-pattern scanning is a focused check, not a comprehensive security audit. Existing published history and historical evidence were retained.
- Unfinished local paper/report edits were excluded; current application behavior is documented separately from committed older paper/deck material.
