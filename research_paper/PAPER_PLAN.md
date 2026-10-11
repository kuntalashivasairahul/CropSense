# CropSense AI research paper blueprint

Prepared 11 October 2026. This is a planning document, not a submission manuscript. It is based on the four supplied files, current project documentation, and exported results CSVs. Experiments were not rerun for this review.

Confirmed student author order (11 October): **Kuntala Shiva Sai Rahul**, **Smrithi Thimmaraju**, **Allva Asritha**. Project guide: **Dr. Ch. Padmini**, following the supplied Review-2 presentation, to follow the students in the proposed author list. Official emails: Kuntala Shiva Sai Rahul — 23eg107e30@anurag.edu.in; Smrithi Thimmaraju — 23eg107e61@anurag.edu.in; Allva Asritha — 23eg107e05@anurag.edu.in; Dr. Ch. Padmini — padmini.ai@anurag.edu.in.

## 1. Recommended paper and contribution

**Working title:** CropSense AI: An Integrated Prototype for Leaf Disease Classification, Multispectral Vegetation Stress Assessment, and Pest Risk Prediction

Write an empirical software-prototype paper. The contribution is the integration and transparent evaluation of three established methods, with explicit limits on what their combined output means. Do not present this as a new learning algorithm or a validated farm advisory service.

The central question is: **What do three independently evaluated crop-analysis modules reveal about the reliability and limits of a combined dashboard?**

Three supporting questions organise the results:

1. How does PlantDoc adaptation change field-photo disease classification and laboratory validation performance?
2. What does field-grouped evaluation reveal about NDVI stress classification, particularly healthy recall and severe-to-healthy mistakes?
3. Does a four-week pest LSTM outperform persistence and a simpler statistical classifier under year-separated evaluation?

The answer can be mixed. The CNN improves on the inspected field benchmark; the NDVI model has important class-level weaknesses; the LSTM beats persistence but trails logistic regression. Reporting these findings is more defensible than claiming all three modules outperform alternatives.

## 2. Submission requirements and decisions

Source: [Guidelines for Mini Project Research Paper Submission.docx](<Guidelines for Mini Project Research Paper Submission.docx>).

| Requirement | Plan |
|---|---|
| Deadline | Submit on or before **17 October 2026**; reserve 16 October for final checks and the required report. |
| Length | Minimum **8 pages**. Aim for 8–9 pages in the supplied IEEE template. Confirm whether references count toward the minimum; budget substantive content rather than padding. |
| Figures | **4–5** original, numbered, captioned, and discussed in the text. Students prepare the final visual assets. |
| Tables | **3–4** numbered, captioned, and discussed in the text; use four substantive tables below. |
| Authors | All student authors and the guide, correct official emails, affiliation **School of Engineering, Anurag University**. |
| References | At least **25** credible and relevant sources, verified individually and cited where used. |
| Similarity | Overall similarity below **15%**, AI-content result **0%**, with a valid report. No editing workflow can guarantee these scores. |
| Figure restriction | The guidelines explicitly exclude figures generated using ChatGPT, Claude, or other generative AI tools. Do not create manuscript figures with an image-generation tool. Clarify whether existing experiment plots are acceptable before selecting final assets. |

On 11 October, the user confirmed that the coordinator permits AI-assisted writing/editing and ordinary experiment plots, provided the submission passes the required check. Proceed with drafting and evidence-based language editing under that confirmation. Keep final figures consistent with the original-figure requirement; no generative-AI illustrations are needed. The required similarity report remains a final submission check, and its scores cannot be guaranteed in advance.

## 3. What to borrow from the approved example

The supplied [organic farming paper](<Organic farming research paper .pdf>) has eight pages, an author/guide table, an abstract and keywords, eight main sections, five figures, four tables, an acknowledgement, and 29 references. Its main sequence is Introduction → Related Work → System Design → Methodology → Data and Implementation → Evaluation Plan → Discussion → Conclusion and Future Work.

Use that sequence and its readable paragraph style as structural guidance. Use the supplied [IEEE template](<IEEE Format.docx>) for final typography and author layout. The template specifies an abstract of at least 200 words and an A4 layout with a two-column main body. Its placeholder author layout should be adapted for the actual four authors.

Our Section VI should be **Experimental Evaluation and Results**, because actual experimental evidence exists. Do not copy the example's prose, proposed features, references unrelated to CropSense, or blank results tables. Its future-tense experiments are appropriate to its design-paper scope; ours should distinguish completed work from future work.

## 4. Proposed manuscript structure

Page allocations are approximate and must be checked after rendering, not treated as a verified page count.

| Part | Approx. pages | Content |
|---|---:|---|
| Title, authors, abstract, keywords | 0.5 | Confirmed student order followed by the guide; official emails supplied. A 200–250 word abstract: problem, three modules, evaluation protocols, key findings, fusion limitation. Write the abstract last. |
| I. Introduction | 0.75 | Concrete problem; why these three signals are useful; dataset-only scope; research questions; specific contributions and paper organisation. Avoid unsupported national statistics and market projections. |
| II. Related Work | 1.0 | Disease detection and domain shift; multispectral vegetation indices; pest/weather prediction; integrated systems and evaluation limitations. End with a bounded comparison to relevant studies. |
| III. System Design | 0.75 | Three independent input/model paths, dashboard, output mapping and illustrative fusion. Clearly show that no shared farm links the datasets. |
| IV. Methodology | 1.5 | MobileNetV2 adaptation, preprocessing and Grad-CAM; raw-band NDVI, six statistics and mask-derived labels; causal four-week pest sequences; baselines and fusion equation. |
| V. Data and Implementation | 0.75 | Dataset provenance, label mapping, split sizes, training configurations, missing-data treatment, software environment, model identity and reproducibility. |
| VI. Experimental Evaluation and Results | 1.25 | Protocols and metrics first, then comparisons, class-level errors and candidate decisions. Separate historical deployment results from retrained evaluation results. |
| VII. Discussion and Limitations | 0.75 | Domain shift; weak healthy recall; repeated test inspection; small temporal test set; stronger logistic baseline; limits of unlinked fusion and raw-band imagery. |
| VIII. Conclusion and Future Work | 0.25 | Answer the research questions. Limit future work to external validation, better labels, more locations/seasons and joined data. |
| Acknowledgement and references | 1.0 | Accurate guide acknowledgement and at least 25 verified citations. |

This totals about 8.5 pages. Do not add generic deployment, ethics or future-feature subsections merely to reach eight pages.

## 5. Planned figures and tables

These are figure specifications, not final assets. Keep editable originals and source data. Under the user's coordinator-permission confirmation, ordinary plots from the experiment data can be prepared for the paper. Check their data, captions and provenance before inclusion.

| Figure | Purpose | Source / preparation |
|---|---|---|
| Fig. 1. System architecture | Three separate data streams through preprocessing, models and dashboard; mark composite output as illustrative. | Students redraw from the implementation and Review-2 architecture slide. Do not imply a joined farm dataset. |
| Fig. 2. Evaluation protocols | Panels for disease adaptation, inferred-field NDVI split and chronological pest split. | Students draw from split manifests and notebook configuration. Mark prior PlantDoc test inspection. |
| Fig. 3. Laboratory-to-field disease comparison | Original/adapted accuracy on PlantVillage validation and PlantDoc test, with sample counts and split labels. | Module 1 results CSV; students prepare a plot with an honest axis. Do not use a truncated axis to magnify changes. |
| Fig. 4. NDVI stress errors | Grouped-baseline confusion matrix, highlighting severe predicted healthy and weak healthy recall. | Module 2 confusion counts, not the older deployed RF's random-split matrix. |
| Fig. 5. Pest baseline comparison | Majority, persistence, selected LSTM and logistic regression on the same 19 test weeks. | Module 3 results CSV; show both accuracy and macro F1, with the limited sample size. |

| Table | Contents |
|---|---|
| Table I. Related-work comparison | A focused subset of relevant studies: task, data, method, evaluation setting and relationship to CropSense. No unsupported claims that a method is absent from all prior work. |
| Table II. Data and evaluation design | Dataset, input, target construction, train/validation/test membership, exclusions and principal validity limitation for each module. |
| Table III. Model configuration and inputs | Input contracts, feature order/dimensions, preprocessing, selected configuration and validation selection criterion. Read exact values from notebooks/configs before writing. |
| Table IV. Results and model decisions | Panels A/B/C for CNN, RF and pest comparisons; accuracy, macro F1, sample counts and decision-relevant error counts. Include the stronger logistic baseline. |

## 6. Claim-to-evidence ledger

The following headline numbers were checked against the exported CSVs during this review. Link manuscript statements to the underlying reports, not just a previous prose draft.

| Claim eligible for the paper | Direct source | Required qualifier |
|---|---|---|
| PlantDoc original CNN 30.95%; adapted CNN 55.95%, n=252; gain 25.00 percentage points. | `docs/evidence/module-1-candidate/results_summary.csv` | Same-loader comparison; test inspected historically, not a newly blind external test. No significance claim without an appropriate analysis. |
| Field macro F1 increases from 0.297 to 0.552. | Same CSV, `macro_f1_present_classes` | Averaged over the 27 present classes. Across all 38 output classes, values are about 0.211 and 0.392; do not mix the definitions. |
| PlantVillage validation accuracy changes from 98.10% to 96.83%, n=8,145. | Same CSV | Validation results, not a separate held-out PlantVillage test. |
| Grouped RF baseline accuracy 58.42%, macro F1 0.474, n=1,549. | `docs/evidence/module-2-candidate/results_summary.csv` | A retrained baseline under inferred-field grouping, not the exact released older forest. Field-identifier convention still needs confirmation. |
| Healthy recall 23/280; severe predicted healthy 74/700. | Same CSV plus `rf_baseline__test_classification_report.csv` and confusion counts | Mask-derived stress categories, not agronomist diagnoses. |
| Threshold candidate macro F1 0.477, but severe-to-healthy errors rise to 81/700. | Same CSV and `docs/evidence/module-2-candidate/decision.json` | Not deployed. Validation selection and subsequent held-out inspection must be described separately. |
| LSTM 52.63%, macro F1 0.433; persistence 42.11%; logistic regression 68.42%, macro F1 0.676. | `docs/evidence/module-3-candidate/results_summary.csv` | Same 19 observed-target weeks in 2009; one region; chronological dependence limits broad conclusions. |
| Integrated dashboard exists and was functionally tested. | `docs/QA_REPORT_2026-10-11.md` | Functional QA is not a farmer usability study or evidence of agronomic benefit. |
| The combined score has no validated accuracy. | `docs/ACADEMIC_REPORT_DRAFT.md`, fusion implementation to verify during drafting | Inputs are from unrelated datasets; equal weights are illustrative. No joint confusion matrix or composite benchmark. |

Additional method details must be checked against the relevant notebook/config before entering the manuscript: PlantDoc mapping and exclusions; adaptation settings; NDVI mask thresholds and water exclusion; inferred-field extraction; all nine pest feature names; missing-value handling; pest class thresholds and target horizon; seeds and package versions. No code or models need changing for this paper stage.

## 7. Corrections to carry forward from Review-2

| Earlier PPT wording | Paper treatment |
|---|---|
| Multispectral / hyperspectral imaging | Describe implemented RGB/NIR multispectral processing. Hyperspectral work is outside the evaluated prototype. |
| Soil-condition monitoring | No direct soil measurements or soil model are evaluated. Describe canopy vegetation stress. |
| No existing IEEE / Elsevier / Springer paper fuses these signals | Remove the universal novelty claim. Establish a bounded gap through verified related work. |
| Tomato feasibility figures 79.4% / 18.1% | Keep separate from the current 38-output-class audit. Omit unless their original protocol and saved results are recovered. |
| Pest interpolation strategy | Explain the current causal protocol and observed targets. Do not describe interpolation using future observations as part of the revised evaluation. |
| Weighted combined health score | State the actual equal-weight illustrative formula and its absence of joint validation. |
| Recommendations, early warning, benefits to farmers | Describe implemented outputs accurately. Improved yields, earlier detection and usefulness to farmers require studies not currently available. |

Use the current evidence documents to distinguish release checkpoints, fresh retraining runs and rejected candidates. The released CNN's 55.95% and a fresh run's 55.16% are different artifacts. Also disclose that successive RF experiments examined the same test partition; do not portray the whole development history as a single untouched final test.

## 8. Reference-building plan

Build a verified bibliography before writing Related Work. The supplied PPT and old draft provide search leads, not a verified final bibliography. Do not copy the friend's reference list to satisfy the count.

Allocate at least 25 relevant sources approximately as follows:

- Disease datasets, field-image evaluation and domain shift: 6.
- MobileNetV2, transfer learning and visual explanation methods: 4.
- Agriculture-Vision, NDVI and vegetation-stress interpretation: 5.
- Pest/weather studies, the ICAR data source and temporal modelling: 5.
- Random Forest/logistic baselines and evaluation methodology: 3.
- Integrated agricultural systems and broader research context: 2.

Include the three coordinator-facing base papers after checking their originals. The latent-heat-index paper motivates crop stress; CropSense does not reproduce its NDLI method. The national monitoring paper motivates pest analysis; its targets and metrics do not form a direct comparison to our cotton classifier.

For each entry record full title, authors, year, venue, DOI or official URL, the claim supported, and the section where it is cited. Read the supporting passage. Prefer publisher pages, original papers and official dataset documentation. Correct any inconsistent metadata in the old slides rather than carrying it forward. Add relevant recent studies where they inform the comparison; retain foundational sources for methods actually used.

## 9. Academic-humanizer workflow

Reviewed source: [academic-humanizer SKILL.md, v0.3.3](https://github.com/AIScientists-Dev/academic-humanizer/blob/main/SKILL.md), repository commit shown as `94b88b2` at review time.

**Recommendation: evidence-first drafting followed by a separate section-by-section editing pass, with the skill's clarity principles used from the start.** Its intended process is audit → rewrite → change report. It is an editing aid, not a source of findings, a similarity certificate or a detector-evasion tool.

With AI-assisted writing/editing confirmed as permitted by the user:

1. Assemble evidence pointers and author notes for one section.
2. Draft in direct academic prose, using exact method names and calibrated claims.
3. Audit locations with generic wording, overclaims or missing evidence before editing them.
4. Edit without changing section content, numbers, equations or citations. Preserve useful uncertainty and technical language.
5. Compare before/after factual statements and record a short change log. Correct factual errors as explicit source-checked corrections, separately from stylistic editing.
6. Have every author read, revise and approve the section, including its limitations. Follow applicable disclosure rules.

Match the example's readability and structure, without imitating its wording or treating a friend's style as the team's own established voice. Do not introduce intentional mistakes, informal anecdotes or invented experience to make prose seem human.

## 10. Work sequence through submission

| Date | Deliverable |
|---|---|
| 11 October | This blueprint; requirement extraction; headline evidence check; author and policy questions. |
| 12 October | Confirm author details and assistance/figure rules; complete source verification and 25-entry bibliography; freeze the claims ledger. |
| 13 October | Team-approved Methods, Data and Results sections; exact configuration and split checks. Team prepares original figure assets. |
| 14 October | Introduction, Related Work and Discussion; verify every literature comparison and practical claim. |
| 15 October | Abstract and conclusion; permitted language editing; assemble in the supplied IEEE template. |
| 16 October | Render and inspect every page; check minimum length, figures/tables, citations, author emails and cross-references; obtain required similarity report and guide review. |
| 17 October | Submit the reviewed manuscript and required report by the institutional deadline. |

Do not schedule new model development merely to improve the paper's headline scores. New experiments are needed only if a claimed result cannot be supported or the guide requires them.

## 11. Details still needed

1. All four official emails are confirmed. Guide designation follows the supplied PPT: Assistant Professor; affiliation follows the submission guidelines.
2. Whether references count toward eight pages, whether there is a maximum length, and which required similarity-report service/submission channel applies. These do not block drafting or evidence preparation.

## Completion boundary

Drafting update, 11 October: the complete first manuscript is now available as `CropSense_AI_Paper.md`, `CropSense_AI_Paper_Draft.docx` and `CropSense_AI_Paper_Draft.pdf`. The final export has eight pages, five source-derived plots, four tables and 26 references. The final figure selection uses empirical plots rather than the proposed conceptual diagrams. See `REVIEW_NOTES.md` for verification scope and remaining provenance/submission checks; the planning-stage status below is retained as the original review record.

Completed: all four uploaded files reviewed; example structure and submission rules extracted; current headline result CSVs checked; humanizer workflow reviewed; paper blueprint prepared.

Confirmed since the initial review: full student names and order; coordinator permission for AI-assisted writing/editing and ordinary experiment plots, as reported by the user.

Pending: verified 25-source literature review, exact method/configuration checks, final original figures, full manuscript, rendered page review and official similarity report. No claim of guaranteed approval, publication readiness or a particular detector score is made.
