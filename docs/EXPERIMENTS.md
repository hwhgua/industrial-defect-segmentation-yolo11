# Experiment history and decisions

## Canonical R4 line

R4 retains full Mosaic augmentation and disables HSV color jitter. The decision was motivated by the fixed appearance of the inspected industrial surface: aggressive color perturbation was treated as a domain assumption to test, not as an automatic improvement.

The public configuration fixes all training parameters in `configs/r4_seed42.yaml`, including:

- YOLO11n-seg, image size 640, 50 epochs, batch 11;
- AdamW with initial learning rate 0.002;
- seed 42 with deterministic mode enabled;
- Mosaic 1.0, closed for the final 10 epochs;
- HSV hue/saturation/value all set to zero;
- translation 0.1, scale 0.5, and horizontal flip 0.5;
- standard mask ratio 4.

## Augmentation sequence

R0–R4 were used to separate changes rather than bundle them into one opaque training run. Most R0–R3 comparisons used seed 42 and should therefore be read as controlled observations, not broad causal claims.

The final HSV-on control and R4 were then repeated as paired seed-42/seed-43 runs. Under a per-seed normal-alarm budget of at most 5/417 images, R4 reached box recall 0.706 for both seeds and reduced false negatives by seven and six relative to its paired control. Precision, false-positive count, and the selected confidence varied by seed; the defensible claim is therefore limited to a repeated recall direction under this protocol.

## Threshold selection

mAP and operating-point selection answer different questions:

- mAP measures confidence ranking across a sweep;
- the frozen operating point evaluates one threshold under a normal-image alarm budget.

Seed-42 R4 selected `confidence=0.0975` with NMS IoU `0.3` on validation. The four-decimal confidence is retained throughout this repository because rounding it to `0.10` changes reproducibility.

## Architecture experiments

A1–A4 explored P2/detail-aware and score-calibration variants through a local `ultralytics-p2-gated` fork. None produced evidence strong enough to replace R4. Several runs were single-seed or incomplete across seeds, and the fork was external to the original project directory.

Those experiments are intentionally excluded from the public executable path. They are best discussed in an interview as negative experiments that led to a stopping decision—not as reproducible model improvements.
