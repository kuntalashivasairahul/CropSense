# Manuscript review notes

Prepared 11 October 2026. The manuscript is a first complete draft for the authors and guide to review, not a submitted paper or a similarity certificate.

## Runtime correction after this draft

On 22 October 2026 the dashboard inference/reporting contract was corrected: Grad-CAM now follows classifier preprocessing, the disease signal sums healthy-class probabilities, and component warning tiers no longer bucket the mean. The manuscript and binary PDF/DOCX below describe the earlier snapshot and have not been silently rewritten. Review the new [runtime contract](../docs/RUNTIME_CONTRACT.md) and [fix status](../docs/AUDIT_FIX_STATUS_2026-10-22.md) before reusing or submitting those drafts. A new [pest source reconstruction](../docs/evidence/pest-recovery-2026-10-22/README.md) restores observations and matches all 19 saved test sequences; it is labelled as a reconstruction, not the original run export. The linked October RF rerun shows 260/399 and a model size different from deployment. A subsequently recovered [September notebook](../docs/evidence/rf-provenance-2026-10-22/README.md) does report 267/400 (66.75%), with matching size, parameters, six printed importances and training count. This corrects the initial audit's missed source. Treat it as a reported historical result with strong artifact consistency; the original export hash and split/prediction rows remain missing, so exact holdout replay is still unverified.

## Deliverables

- `CropSense_AI_Paper.md`: editable text source, with the complete bibliography.
- `CropSense_AI_Paper_Draft.docx`: editable A4 document with a two-column main body.
- `CropSense_AI_Paper_Draft.pdf`: PDF exported from that Word document using Pages; inspected page by page.
- `references.json`: 26 references with source URLs, intended claim support, and verification scope.
- `figures/`: five deterministic plots in PNG and vector PDF formats, built from the repository's saved CSV reports.
- `build_figures.py` and `build_paper.cjs`: repeatable figure and Word-generation scripts. They contain small assertions for the source counts, key results, manuscript sections, figures, tables, abstract length and citation coverage.

The authors' names use the same title-case styling: Kuntala Shiva Sai Rahul, Smrithi Thimmaraju, Allva Asritha, and Dr. Ch. Padmini. Their supplied official email addresses are included. The guide designation in the acknowledgement follows the uploaded PPT; the common affiliation follows the submission guidelines.

Final export checks: **8 pages, 5 figures, 4 tables, 26 cited references, and a 229-word abstract**. All four email addresses are present in the PDF. The DOCX passes the document skill's XML/schema validation. Extracted PDF text remains within page bounds. Visual inspection checked all pages, including the table column widths, equation layout, figure captions and consistent author styling. The figures use source-derived data; none is a generative-AI illustration.

## Evidence checks and deliberate boundaries

Headline metrics were checked against all three result-summary CSVs. The RF and LSTM confusion counts were checked for totals and the key error cases. PlantVillage counts and mapped PlantDoc counts were checked against `class_distribution.csv`. RF split counts were checked against its local split manifest. Methods were checked against the audit notebooks, configuration files, contracts, checkpoint-selection history, and current score functions in `core.py`.

The paper identifies the following boundaries explicitly:

1. The PlantDoc benchmark and the 2009 pest season had been inspected in earlier work. The paper does not describe either as a newly blind external test.
2. The grouped RF is retrained, whereas the dashboard release retains the older forest. The grouped test score is not attributed to the released artifact.
3. RF development repeatedly examined the same test partition; a validation-only selection step does not make that full development history independent of the test.
4. The LSTM is weaker than logistic regression and has zero medium-class recall on the 19-week test.
5. The raw-band NDVI pipeline uses heuristic mask labels and inferred field identifiers. Neither physical reflectance calibration nor agronomist severity labels have been established.
6. The three datasets do not form a joined farm-level test. No composite accuracy or agronomic benefit is claimed.

No models were retrained or changed for this manuscript. No application behavior was changed. The `core.py` SHA-256 before manuscript work was `5c2b3f00a8681f58d3a1f4381fff6b988d07f11f3646201612d7ecdf11bc375d`.

## Reference verification scope

All 26 entries were checked against original arXiv/journal/database pages or publisher-deposited Crossref metadata through the gstack browser. The arXiv abstracts, the two JMLR abstracts, the full Mohanty article, the Roberts and original LSTM abstracts returned in Crossref metadata, and the official ICAR Jassid source table were read. DOI-only entries were checked for title, authors, year, venue and available identifiers; the draft limits their use to broad topic or method attribution where full text was not read. This is not a claim that all 26 full texts were reviewed.

The publisher page for the Liakos review returned an access-denied response; its Crossref abstract and bibliographic record were available. For leakage, the 2023 published Patterns record is cited and its corresponding 2022 arXiv abstract was read. The incorrect Barbedo year and abbreviated title from the old PPT were replaced with the verified 2016 record and full title.

Before submission, the team should read the original three supplied base papers and the DOI-only sources where a more detailed literature comparison is desired. Do not add reported performance values or algorithm details from those papers without checking their full text.

## Academic-humanizer audit and edit report

Applied [academic-humanizer v0.3.3](https://github.com/AIScientists-Dev/academic-humanizer/blob/main/SKILL.md) as an editing workflow. The approved friend's paper informed structure and readability; its prose was not copied.

The initial prose was already drafted with direct sentences, explicit denominators, and calibrated claims. The audit found no need for a broad rewrite. It identified three small language changes:

| Location | Audit finding | Edit |
|---|---|---|
| III.C | Abstract phrasing: “The contribution assessed here is…” | Use the direct actor: “We evaluate…” |
| VII.A | Vague approval: “a useful software boundary” | State its limited role: keeping inputs separate helps interpretation. |
| VI.D | Indirect phrasing: “The result motivates examining…” | State that the errors warrant examining the pressure representation and class boundaries. |

Checks confirmed that this language pass preserved numeric tokens, equation text, citation sequence, paragraph count, and section structure. Bibliography renumbering was performed separately, before this pass, to follow order of first citation; source identities were preserved. No findings, anecdotes or intentional errors were invented to affect an AI detector. No similarity or AI-detection service received the manuscript.

## Items requiring author or guide review

- **Dataset provenance correction, 22 October:** recovered original CNN records establish that 2,912 is the retained mapped field count **after deduplication**, not the raw mirror total. The mirror's 252-example test differs from upstream's 236-image test: 185 exact upstream matches enter mirror test, two enter mirror validation, and 49 have no exact match. The upstream partition is therefore not blind to model selection. The historical manuscript wording needs review before submission; do not imply dataset equivalence or independent evaluation. See [recovered records and label review](../docs/evidence/disease-recovery-2026-10-22/README.md).
- **Local exports recovered, 22 October:** original CNN v1 training/v5 verification and pest v3 records now restore full split/window manifests, histories and sample predictions. CNN files match deployment exactly. Pest containers differ, with identical weights/effective configuration and deployed output parity on 31 validation/test rows. See [pest archive checks](../docs/evidence/pest-export-recovery-2026-10-22/README.md). Original RF CSV/splits/hash-at-run remain missing. Historical paper/deck binaries were not silently regenerated.
- **Field identifiers:** on 22 October, the maintainer README confirmed the field-ID prefix convention for the challenge dataset. This supports the grouping rule but does not by itself establish the exact 2017 miniscale archive lineage; retain that remaining qualification. See the [source-scope note](../docs/evidence/canopy-validation-2026-10-22/README.md).
- **Figures and assistance:** the user confirmed coordinator permission for AI-assisted writing and ordinary experiment plots. The figures are data plots, with no image-generation model used. Follow any required institutional disclosure wording.
- **Submission requirements:** obtain the official similarity report, author/guide approval, and any exact instructions on whether reference pages count toward the eight-page minimum. The draft includes substantive text rather than blank pages, and the export meets the stated total-page minimum.

No actual farmer study, field outcome validation, probability calibration experiment, statistical-significance analysis, or new runtime benchmark was performed for the paper. These are reported as limitations rather than filled with invented results.

## Rebuilding

From the project root, install document-generation dependencies in a temporary tools directory rather than changing the application's dependencies:

```sh
uv run --with matplotlib python research_paper/build_figures.py
npm install --prefix /tmp/cropsense-paper-tools docx
NODE_PATH=/tmp/cropsense-paper-tools/node_modules node research_paper/build_paper.cjs
```

Export the generated DOCX to PDF using Pages or Word and inspect the pagination again. A later renderer or manuscript change can alter page breaks. The PDF is not automatically rebuilt by the Word-generation script.
