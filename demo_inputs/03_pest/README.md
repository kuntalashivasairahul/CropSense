# Pest-risk demo: no image input

Open **Pest risk** in the Streamlit app. Select an existing four-week test sequence and click **Forecast pest outbreak**. The actual inputs are four weeks × nine numeric features, including training-scaled prior pest pressure. The app reads these from `models/sample_sequences.json`, already included in the clone.

- `2009-W40 (recorded: Low)` is a saved low-labelled target week.
- `2009-W45 (recorded: High)` is used in the severe demonstration. The released model predicts Medium, with expected pressure about 0.570. Explain the mismatch rather than presenting the recorded label as the model's result.
- Change the selected week and show that the old prediction/composite clears; rerun before opening Composite.

This is historical cotton/Akola replay, not current weather ingestion. No pest photograph, screenshot, CSV upload, API key, or new dataset download is needed. Recorded model performance remains 10/19 correct, with 0/5 medium recall; the saved logistic baseline scored 13/19.
