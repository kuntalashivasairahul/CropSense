# Runtime interpretation and input contract

Updated 22 October 2026 for the audit-fix branch. Model weights and benchmark results are unchanged. This document describes the corrected runtime; older paper/deck exports describe a historical snapshot.

## Leaf inference and explanation

The saved CNN consumes raw 0–255 RGB at 224×224. Its graph contains MobileNetV2 scaling. Grad-CAM must replay that same scaling before the backbone and reproduce classifier probabilities; `test_gradcam_forward_probabilities_match_classifier` checks this numerically. The overlay shows influence, not verified lesions.

`disease_score` now sums the output probabilities of **all classes whose label contains healthy**. It is an uncalibrated healthy-class probability mass, not a severity estimate. Previously, one minus a disease-class confidence incorrectly turned ambiguity into apparent health.

`needs_review` is true if the largest class probability is below 0.5 or an image heuristic flags the photo. The 0.5 cutoff is a conservative demonstration rule, not an experimentally calibrated operating point. The image heuristic can flag real leaves and miss classification errors. An unknown class or validated eligibility detector has not been trained. The [554-image diagnostic](evidence/ood-validation-2026-10-22/README.md) flagged 38/236 source-labelled PlantDoc examples and accepted 29/30 sunflower examples; passing the screen does not establish supported leaf/crop eligibility. The PlantDoc examples include slides and montages, so their source-label flag rate is not a calibrated farmer-photo error rate.

## Canopy target meaning

The canopy labels are thresholded anomaly-mask coverage, not independent plant-health diagnoses. A severe class means at least one selected severe mask exceeds 5%; otherwise a moderate class means at least one selected moderate mask exceeds 2%; otherwise the training label is `healthy`. Coverage below those limits may still contain anomalies. A descriptive check reproduced all 8,308 stored labels and found anomaly pixels in 1,491/1,505 healthy-labelled tiles. These counts describe the existing dataset, not new model accuracy. The [label diagnostic](evidence/canopy-validation-2026-10-22/label_diagnostic.json) includes source hashes and its reproducible script.

## Mean index and component warnings

The numeric index remains the equal-weight mean of leaf healthy probability, illustrative canopy score, and inverted expected pest pressure. Its inputs come from unrelated datasets; no fused accuracy or farmer outcome has been measured.

The **warning tier is separate from the mean**. It uses the worst component: below 0.33 is Severe Risk, 0.33–below 0.66 is Moderate Risk, otherwise Healthy. A flagged/uncertain leaf produces Needs review and retains component warnings. Consequently a severe component is not cancelled by favorable remaining components. These labels describe prototype model signals, not a field diagnosis or agronomic severity scale.

Use `core.composite_assessment` for display and exports. `core.composite_score` is arithmetic only; `core.bucket` classifies an individual scalar and must not be used to replace component warnings.

## Inputs and validation

- Images: JPG/PNG, at most 10 MB in the UI and **1,048,576 pixels** per image. Dimension checks occur before conversion/array allocation. Larger inputs must be resized before upload.
- Canopy: spatially registered RGB plus 8-bit grayscale `L` NIR of equal size. A color photo is not NIR. Equal size does not prove registration, genuine NIR acquisition, or reflectance calibration. An all-zero red/NIR pair is rejected.
- NDVI: `(nir - red) / (nir + red + 1e-8)` and the existing six features are unchanged. Manual values must be finite, within physical feature bounds, ordered min ≤ p25 ≤ p75 ≤ max, with mean inside min/max and feasible standard deviation.
- Pest: a finite array matching the model's four-week × nine-feature input. The ninth feature is training-scaled prior pest pressure. Weather normalization is embedded in the saved model. This is historical cotton/Akola replay, not weather-only or live ingestion.
- Model outputs: shape, finiteness, nonnegative probability bounds and probability sum are checked. Vectors within floating-point sum tolerance are normalized; derived probability scores stay within 0–1 despite roundoff. Composite scalars reject NaN, infinity and values outside 0–1.

## Artifacts and reports

`core.model_path` now resolves only names in the release manifest and verifies/restores them from the checksummed local bundle. It no longer accepts arbitrary local model bytes or falls back to an unpinned remote revision. The release remains recoverable without network access.

Reports include a UTC timestamp, model hashes, available source IDs/content hashes, component warnings, model limitations and explicit simulation language. Acquisition times are marked unrecorded; report generation time is not substituted for collection time. A supplied report score/tier must match the component assessment.

Follow-up text requests confirmation and field review. The software no longer prescribes pesticide products, generic anti-fungal treatment, trap density or irrigation from an unverified prediction.

## Evidence that remains unresolved

The deployed forest recognizes 0/6 bundled healthy pairs. A recovered September notebook reports 267/400 (66.75%) with matching size, six printed feature importances, parameters and training count, but lacks an export hash and original split/prediction rows. A separate October rerun reports 260/399 (65.16%) and a different model size. These records are now distinguished in the [origin trace](evidence/rf-provenance-2026-10-22/README.md); neither source recovery nor the separately retrained grouped baseline's 58.42% is a new validation of the release.

The released LSTM reproduces 10/19 correct with 0/5 medium recall; saved logistic regression scores 13/19. Those already-inspected weeks cannot become a new blind test. More appropriate data and preserved run records are needed before claiming either quality issue fixed.
