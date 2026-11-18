# Contributing

Yardline Intelligence is a research repository. Contributions should preserve scientific traceability and avoid overstating model performance.

## Before changing code

1. Read the project methodology and data-governance documentation.
2. Keep train, selection and test periods chronological.
3. Verify that every feature is available before the prediction cutoff.
4. Preserve missing values as missing; do not silently replace them with zero.
5. Compare every candidate model with the current baseline.

## Experiments

Every experiment should record its ID and timestamp, source dataset versions and checksums, target definition and scoring rules, train/selection/test seasons, feature set and model parameters, random seeds, metrics and sample counts, plus known limitations and failed hypotheses.

Do not overwrite previous experiment outputs. Create a new version or experiment ID.

## Code quality

- Keep ingestion, feature engineering, modeling and evaluation separated.
- Add tests for temporal leakage, missing-value behavior and player identity joins.
- Run `pytest` before opening a pull request.
- Never commit credentials, private league data or raw datasets.

## Reporting

A model should only be described as better when committed out-of-sample evidence supports it. Report negative results as carefully as positive ones.
