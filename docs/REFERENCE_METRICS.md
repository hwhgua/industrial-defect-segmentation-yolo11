# Reference metrics

`REFERENCE_METRICS.csv` contains aggregate split counts and reference model metrics for the documented seed-42 experiment. Values are transcribed from `RESULTS.md`; the CSV does not contain per-image identifiers, source images, masks, predictions, or model weights.

The aggregate false-alarm comparison is visualized in [`assets/false-alarm-rate.svg`](../assets/false-alarm-rate.svg).

The `official_ultralytics_map` rows use confidence 0.001, prediction NMS IoU 0.7, and `max_det=300`. The `frozen_operating_point` rows use confidence 0.0975, prediction NMS IoU 0.3, and a match IoU threshold of 0.5, except for explicitly noted aggregate counts.

The official test split was inspected during earlier exploratory work. Treat these as reference experiment results, not an unbiased estimate of performance on unseen deployment data. The frozen test normal-image false-alarm rate (2.24%) exceeded the approximately 1.20% validation-derived target.

Dataset images and masks, per-image evaluation tables, and trained checkpoints are intentionally excluded. See `DATASET.md` for dataset source terms and `LIMITATIONS.md` for claim boundaries.
