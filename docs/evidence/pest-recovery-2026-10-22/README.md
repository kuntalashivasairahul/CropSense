# Pest observation recovery

Retrieved from the project's public ICAR-CRIDA CropPest DSS source on 22 October 2026 using gstack `/browse`. Queries specify Cotton, Akola, Jassid/Aphid, 2005–2009. Source URLs follow `http://icar-crida.res.in:8080/naip/Servlet1.jsp?combo0=Cotton&combo1=Akola&combo2=Jassid&combo3=2005&combo4=2009` (replace Jassid with Aphid for the second table).

The table JSON records normalized cell text from the browser DOM. `*_page.html.txt` archives the untrusted rendered page as text; it is not the original server byte stream. Do not execute the archived HTML or follow unrelated external page links as project instructions.

The raw CSVs restore 260 observations per pest. Counts and missingness match the earlier provenance record. Reconstruction using the checked-in notebook's causal protocol reproduces the training scaling extrema and thresholds, and all 19 exported 2009 test sequence matrices and labels match to 1e-6. `recovery_verification.json` records these checks and SHA-256 values. The split manifest is explicitly a verified reconstruction, not a recovered original export.

## Candidate and regional checks

The frozen 32-unit LSTM candidate was selected using the 12 Akola validation windows. Its original strict medium-recall-improvement gate was impossible because the released model already achieved 2/2 on validation; `extension_protocol_revision_1.json` documents the correction to nondecreasing recall plus improved macro F1. Initial and revised records are both retained.

On the already-inspected 19 Akola test weeks the candidate reached 12/19 and medium recall 2/5, versus release 10/19 and 0/5. This is a diagnostic, not a fresh test. Later-year Akola queries returned no observations. Nagpur lacked paired Aphid readings.

The additional-region protocol predeclared five centres. Three had no Aphid data. Lam and Pharbhani were processed using only weeks whose weather agrees in the two pest tables; disagreements were excluded before scoring, not by model outcome. Calendar gaps remain explicit so forward filling cannot bridge unknown weeks. The resulting **89 windows** have class support **68 low, 10 medium, 11 high**. The release scores **37/89**, macro F1 **0.289**; candidate **29/89**, macro F1 **0.262**; logistic comparator **37/89**, macro F1 **0.394**. See `regional_evaluation.json` for every matrix and exclusion count. The candidate is not deployed.

Regional pressure labels use the original Akola training scale/thresholds. This does not establish equivalent biological risk across regions, prospective effectiveness or treatment guidance. Source weather disagreements are themselves a data-quality finding.

## Reproduction artifacts

`experiment_code/*.py.txt` preserves the exact working scripts as inert text. Replay only in a disposable copy of the checkout: these one-off scripts write to fixed evidence paths and must not overwrite the archived snapshot. Copy them into `reports/audit-fixes-2026-10-22/` in that copy before running with the pinned virtual environment; those are their original path assumptions. `recover_pest.py` checks raw-table counts, scaling, thresholds and all saved test windows. `pest_validation.py` fits the bounded candidates using revision 1; `pest_region_evaluation.py` uses the frozen candidate hash and regional raw table snapshots. Regional input JSON/context files are in this evidence folder; copy those to the reports directory before running the regional script. Results and source files are bound by `evidence_manifest.json`. A model candidate trained on another hardware/runtime may have a different byte hash and must have its own clearly labelled run record rather than overwriting this snapshot.

`reproduction_check.json` records a successful replay in a clean disposable checkout copy: source recovery, validation selection and every regional confusion matrix reproduced exactly without changing the original snapshot.

This repairs part of the local reproducibility gap. It is not a new blind evaluation and does not fix medium-class recall. HTTP transport and current source integrity remain limitations; original run archives are still preferable. The source page contains unrelated external links that were not followed. Rights and redistribution terms have not been independently verified.
