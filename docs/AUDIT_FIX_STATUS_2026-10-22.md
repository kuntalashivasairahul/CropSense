# Audit remediation status

Current [audit reassessment](AUDIT_REASSESSMENT_2026-10-22.md): **74/100**, compared with 57/100 before repairs. This is an engineering/evidence judgment, not accuracy.

Work branch: `codex/audit-fixes-20261022`. Baseline: `5e0c64b`. Original model weights and recovery bundle are preserved. A research-only LSTM candidate exists separately; it is not deployed. This is a progress record, not a claim that every scientific limitation is solved.

| Finding | Current disposition | Verification / remaining work |
|---|---|---|
| H1 Uncertainty becomes health | Implemented | Healthy-class probability sum and review flag; regression covers an ambiguous 38-class distribution. No calibration claim. |
| H2 Mean hides critical signals | Implemented | Shared assessment retains worst component and review warnings; UI/export use it. Extreme-component tests cover each module. |
| H3 Grad-CAM forward mismatch | Implemented | Explanation head probabilities match the classifier to 1e-5 on the real fixture. |
| H4 RF metric provenance | Historical source recovered; exact split/hash link remains open | The [September notebook](evidence/rf-provenance-2026-10-22/README.md) reports 267/400 = 66.75%, with size, six importances, parameters and training count consistent with deployment. This corrects the initial audit's missed source. The October export is a separate rerun. Original split/prediction rows and hash-at-run are still missing. [Files needed](KAGGLE_ARTIFACT_HANDOFF.md). |
| H5 RF healthy recognition | Open model-quality issue | Real-fixture diagnostic replaces synthetic input-search smoke examples; 0/6 healthy matches remain visible. |
| H6 LSTM medium recall / stronger baseline | Candidate researched and rejected; quality remains open | Validation-selected candidate improves inspected Akola test from 10/19 to 12/19 and medium recall 0/5 to 2/5, but falls from release 41.6% to 32.6% on 89 independent-region diagnostic windows; not promoted. See recovered-source evidence and protocols. |
| M7 Hidden failures | Implemented | User-visible module errors, helper loading inside handled inference, retry clears error; regression exercises failure and retry. |
| M8 Scenario sequence mismatch | Implemented | Scenario updates the actual selector; selection persists across module navigation; regression compares visible sequence to prediction input. |
| M9 Impossible manual statistics | Implemented | Shared finite/bounds/order/variance validation runs before inference; invalid vectors cannot call the model. |
| M10 Nonfinite core values | Implemented | Pest shape/finiteness, probability-vector and composite scalar checks. |
| M11 Image resource limit | Implemented | Pre-allocation pixel cap, including direct color/image-check entry points; regression rejects before conversion. No memory-exhaustion test attempted. |
| M12 Unsupported treatment advice | Implemented | Neutral confirmatory follow-up replaces product names, generic anti-fungal fallback and trap dosage. |
| M13 Export warning/provenance loss | Implemented | UTC, model hashes, input source IDs/hashes, simulation and component/model warnings. |
| M14 Decimal carry | Implemented and browser-verified | Whole-value rounding renders the healthy-tomato confidence as 100.0%, with CSS counters 100 and 0; gauge carry boundary is covered by a regression. |
| M15 OOD validation | Broader diagnostic completed; eligibility quality remains open | Frozen 554-image check: 38/236 source-labelled PlantDoc examples flagged; 270/270 non-plant objects flagged, but 29/30 sunflower images accepted. No threshold tuning or eligibility calibration claimed. [Complete evidence](evidence/ood-validation-2026-10-22/README.md). |
| M16 Missing exports / fresh evaluation | CNN/pest exports recovered; RF and independent-validation gaps remain | Original CNN v1/v5 exports restore full splits, training/selection and predictions; both CNN files match deployment. Original pest v3 exports restore all 89 windows; effective model parity and all 31 validation/test probability rows are verified. RF original CSV/splits and independent compatible validation remain missing. |
| M17 Acquisition/calibration contract | Partly mitigated | Blank pairs rejected and supported 8-bit/alignment assumptions documented; real sensor calibration/registration cannot be established from uploaded encoding. |

The corrected [runtime contract](RUNTIME_CONTRACT.md) defines score semantics and input constraints. Archived notebook exports, prior evaluation matrices, model weights and manuscript/deck binaries remain historical evidence; they have not been silently regenerated with different results.

Verification: 27 unittest cases, `test_core.py`, the full offline environment/inference check and `pip check` pass. Browser checks confirm corrected confidence counters, scenario/pest selection, simulation warnings and the actual downloaded report contents. Adjacent regressions also cover returning from a scenario to manual inputs and stale leaf-source invalidation. No production model weight was changed.

The [recovered CNN archives](evidence/disease-recovery-2026-10-22/README.md) expose mirror/upstream split differences and ambiguous comparison-chart labels. The current notebook explicitly logs the excluded chart directory and source-file counts, and saves accepted source rows before deduplication. A new regression exercises excluded and unknown folders. This repairs audit transparency without changing the mapping, release model or historical results. The [original pest export checks](evidence/pest-export-recovery-2026-10-22/README.md) are separate from the earlier reconstruction.

A diagnostic comparison used only the existing 5,175 training and 1,584 validation tiles, with six NDVI features. No final-test predictions were used to select candidates and no weights were deployed. This is a fresh fit from serialized CSVs, not a claim to reproduce the original training run exactly:

| New diagnostic fit | Validation macro F1 | Healthy recall | Severe → healthy, out of 613 |
|---|---:|---:|---:|
| RF baseline | 0.483 | 45/318 | 24 |
| Extra Trees | 0.477 | 72/318 | 49 |
| Histogram gradient boosting | 0.484 | 98/318 | 75 |
| Logistic regression | 0.419 | 45/318 | 60 |

None of the three alternatives preserved the baseline severe-to-healthy guardrail. Better healthy recall alone is not a safe reason to replace the release. This leaves the need for label/acquisition review and additional data unresolved. A descriptive label audit reproduced all 8,308 stored threshold labels; 1,491/1,505 healthy-labelled tiles have selected anomaly pixels below the cutoffs. UI, follow-up text and exports now explain this target meaning. The [diagnostic and source-scope note](evidence/canopy-validation-2026-10-22/README.md) distinguish documented challenge-dataset naming from the remaining exact miniscale-archive lineage question.

Pest work recovered the public Akola tables and verified all 19 saved sequences and labels. The released model has 2/2 medium validation cases correct, so an initially declared strict-increase recall gate was impossible; revision 1 transparently requires preserving recall while improving macro F1. Both protocol versions and the initial result are retained. The selected 32-unit LSTM improved the already-inspected 2009 diagnostic to 12/19, macro F1 0.606, and medium recall 2/5, without lowering low/high recall there.

That improvement did not generalize to the newly inspected regional check. Five additional source centres were predeclared; only Lam and Pharbhani had paired usable records. Weeks with inconsistent weather between pest tables were excluded before model scoring, with calendar gaps preserved. The remaining 89 windows gave release accuracy 37/89 (41.6%), macro F1 0.289; candidate 29/89 (32.6%), macro F1 0.262; logistic 37/89 (41.6%), macro F1 0.394. These use training-defined pressure thresholds across regions, not agronomist outbreak labels. The candidate was rejected and production weights remain unchanged. [Protocols, raw records, reconstruction and complete matrices](evidence/pest-recovery-2026-10-22/README.md) preserve the evidence. Full completion requires resolving the open model/data items as well as verifying the software changes.
