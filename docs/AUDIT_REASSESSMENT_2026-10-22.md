# CropSenseAI audit reassessment

**Current assessed score: 74/100 (7.4/10), up from 57/100 before the fixes.** This is a judgment of the academic prototype's engineering and evidence, not predictive accuracy. The software repairs improve how results are computed, interpreted and reported. No production model was replaced, and the remaining model-quality issues are not counted as solved.

The assessment originated on `codex/audit-fixes-20261022`, based on `5e0c64b`. On 23 October the reviewed runtime/evidence changes were committed as `05b4fc4` and published on `deploy/streamlit-demo`; actual Windows laptop verification remains open. The [finding-by-finding status](AUDIT_FIX_STATUS_2026-10-22.md), [runtime contract](RUNTIME_CONTRACT.md) and [deployment record](DEPLOYMENT.md) describe the implementation and its limits.

| Domain | Before | Current | Basis for current assessment |
|---|---:|---:|---|
| Runtime correctness and result interpretation | 11/20 | 18/20 | Healthy probability mass, worst-component warning tier, Grad-CAM forward parity, state consistency and visible failures are repaired. Thresholds and illustrative scores remain uncalibrated. |
| Predictive model quality | 11/25 | 11/25 | Production weights are unchanged. Field disease errors, weak healthy-canopy recognition and pest medium-class failure remain. Regional pest evidence further limits generalization claims. |
| Evaluation design and provenance | 8/15 | 11/15 | September RF source separated from the October rerun; original CNN/pest exports recovered and checked; failed candidates and regional exclusions retained. Mirror/upstream split and chart-label issues are now explicit. Original RF split/hash records and fresh independent validation remain incomplete. |
| Tests and verification | 9/15 | 13/15 | 27 passing cases cover semantic errors, invalid inputs, failures, state, exports and training-source exclusions; real-model offline checks and browser evidence pass. A 554-image diagnostic characterizes image-check failures; independent eligibility and actual laptop validation remain missing. |
| Packaging and deployment | 8/10 | 8/10 | Checksummed local restoration works and release weights are preserved. Current changes have no fresh Windows laptop or remote CI result. |
| UI and usability | 7/10 | 9/10 | Scenario selection, error visibility, rounding and simulation/label explanations are corrected. No farmer usability study was conducted. |
| Security and input robustness | 3/5 | 4/5 | Pixel caps, finite/shape/probability checks and allowlisted verified artifacts improve boundaries. This remains a narrow application review, without a dependency-vulnerability or penetration assessment. |
| **Total** | **57/100** | **74/100** | **Improved academic demonstration; unresolved scientific limitations prevent a stronger assessment.** |

## What was repaired

Eleven of the original 17 findings have implemented software repairs: H1–H3 and M7–M14. Disease uncertainty no longer becomes apparent health; a favorable mean cannot hide a severe component; Grad-CAM follows the classifier's forward preprocessing. Invalid inputs and model failures are handled visibly. Scenario selections match inference inputs, numeric counters round correctly, and exported reports retain limitations, provenance and model hashes. Unsupported treatment prescriptions were removed. See the status document for acceptance evidence rather than treating passing tests as model-quality proof.

**Correction to the initial audit's H4 evidence:** the original public September notebook was recovered and does report 267/400 (66.75%). Its file size, six printed feature importances, parameters and training count match the local artifact's corresponding observations. The initial audit had inspected the later October rerun and missed this source. The reported September result is now distinguished from October's 260/399 and the grouped retrained baseline. An export hash and original split/prediction rows are still missing, so the historical holdout is not fully replayed or cryptographically tied to the run. [Recovered evidence](evidence/rf-provenance-2026-10-22/README.md) is retained alongside unchanged historical model/notebook/document binaries.

## Model metrics that still matter

| Evaluation | Result | Meaning and boundary |
|---|---|---|
| Released CNN, saved PlantVillage validation | 96.83% on 8,145 | Lab-image result; not field accuracy. |
| Released CNN, saved Kaggle PlantDoc mirror test | 141/252, 55.95%; macro F1 0.552 over 27 present classes | Historically inspected mirror split differs from upstream; late-blight recall 2/10 and healthy-cherry recall 0/10 remain concerns. |
| Released RF, 18 bundled paired images | Healthy 0/6, moderate 6/6, severe 6/6 | Small fixture diagnostic; exact historical release holdout accuracy remains unknown. |
| Separately retrained grouped RF baseline | 58.42% on 1,549; healthy recall 23/280; severe→healthy 74/700 | A different model, not the deployed forest. |
| Released LSTM, saved Akola 2009 test | 10/19, 52.63%; macro F1 0.433; medium recall 0/5 | Retrospective pressure classification. Saved logistic baseline: 13/19, 68.42%. |
| Regional pest diagnostic, 89 compatible windows | Release 37/89, 41.57%; macro F1 0.289 | Lam and Pharbhani records use training-defined pressure thresholds, not independent agronomist outbreak labels. |
| Fused index | No measured accuracy | The three input datasets are unrelated and have no joined farm outcome labels. |

The latest canopy alternatives raised severe-to-healthy mistakes despite some gains in healthy recall. The validation-selected pest LSTM improved the already-inspected Akola diagnostic to 12/19, but dropped to 29/89 on the regional diagnostic and was rejected. Neither experiment justified changing release weights. [Canopy evidence](evidence/canopy-validation-2026-10-22/README.md) and [pest protocols, reconstruction and matrices](evidence/pest-recovery-2026-10-22/README.md) preserve the unsuccessful results.

A further label audit reproduced all 8,308 canopy threshold labels. Of 1,505 healthy-labelled tiles, 1,491 contain selected anomaly pixels below the cutoffs. The UI, follow-up text and export now explain that healthy means the below-threshold training class, not verified plant health. This descriptive result is not a new model benchmark.

## Verification and work still required

Latest local checks: **27/27 unittest cases pass**, full offline environment/inference verification passes, and the diff whitespace check passes. Local browser checks verify the canopy explanation and revised disease benchmark scope with no console errors after clean reloads. Earlier checks in this same fix session verified confidence carry boundaries, scenario/pest state and downloaded report contents. Model weights, recovery bundle and original demo sequences remain unchanged. The software regression tests do not establish semantic accuracy on new crops or regions.

A fixed 554-image image-check diagnostic now flags 38/236 source-labelled PlantDoc examples, flags all 270 selected non-plant object images, but accepts 29/30 sunflower images. This exposes the difference between plant-like content and a supported leaf. The [full results](evidence/ood-validation-2026-10-22/README.md) retain the source-label, montage and sampling caveats. No eligibility threshold was tuned on these outcomes, and the score remains 74/100.

Original [CNN training/verification exports](evidence/disease-recovery-2026-10-22/README.md) and [pest exports](evidence/pest-export-recovery-2026-10-22/README.md) are now recovered. CNN model and split bytes match across versions; the pest weights/configuration and deployed validation/test outputs match despite different serialized containers. The mirror's 2,912 examples are post-deduplication. Two upstream test images appear in mirror validation, and ambiguous charts require curation. These findings improve traceability while leaving the 74/100 assessment unchanged.

Remaining requirements are better supported canopy targets/features and independent data, pest validation that survives region shifts, independently validated leaf/crop eligibility, the exact original forest CSV/split/hash records, sensor calibration/registration evidence, and testing these changes on the actual Windows laptop. The original RF metric cannot be recovered by substituting a newly trained model or split. The current paper/deck binaries still describe an older snapshot and require review before reuse.

The project is stronger as a transparent academic demonstration. The full “fix the issues” goal remains incomplete because these model/data and hardware requirements remain open.
