# Reproducibility checklist

## Reference environment

| Component | Reference version |
|---|---|
| Python | 3.10.19 |
| Ultralytics | 8.3.221 |
| NumPy | 2.2.6 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |

PyTorch and CUDA should be installed using the selector appropriate for the target platform. `requirements.txt` pins Ultralytics and NumPy but does not force a CUDA-specific PyTorch wheel.

## Verification levels

1. `pytest -q` runs conversion, deterministic splitting, verification, and matching tests on generated synthetic images. It needs no network, private data, GPU, or model weights.
2. `ksdd2-train --dry-run` validates and prints the exact R4 plan without importing Ultralytics.
3. Dataset preparation should reproduce the expected counts and reference manifest checksum.
4. A one-epoch CPU/GPU smoke run confirms that the installed PyTorch/Ultralytics stack can start and finish training.
5. The full 50-epoch run and frozen test evaluation reproduce the reference experiment, subject to platform-level numeric variation.

## Integrity checks before publishing

Run from the repository root:

```bash
python -m compileall src tests
pytest -q
ksdd2-train --config configs/r4_seed42.yaml --dry-run
```

Also confirm that Git is not tracking datasets, checkpoints, cache files, local `.ultralytics` state, generated runs, absolute machine paths, or UUIDs.

## Results provenance

The values under `metadata/reference_results.json` and `docs/RESULTS.md` were copied from the completed local experiment. They are evidence records rather than outputs generated during CI. Generated metrics should be compared with those values, not silently overwrite them.
