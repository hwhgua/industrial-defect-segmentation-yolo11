# Limitations and claim boundaries

## Test-set history

The original KolektorSDD2 test split was used as validation during early exploratory baseline, confidence, and crop work. A formal train/validation protocol was later created, R4 and its operating point were frozen, and the official test split was evaluated once under that frozen protocol.

Because the images had been inspected earlier, this repository does not call the official test result fully unseen or unbiased. A new external holdout is required for a research-grade generalization claim.

## Validation selection pressure

Validation contains only 49 positive images and was used across multiple augmentation, checkpoint, confidence, NMS, and architecture decisions. Validation overfitting is plausible even though split leakage checks pass.

## Deployment readiness

At the frozen operating point, the model localized 75 of 110 positive test images and generated false alarms on 20 of 894 normal images. The 2.24% normal-image false-alarm rate exceeded the original approximately 1.20% target. False negatives also remain unacceptable for a high-reliability inspection line without an additional decision layer or human review.

Accordingly, this project demonstrates an experimental CV workflow. It does not claim production deployment, safety qualification, or a finished AOI product.

## Reproducibility limits

- GPU kernels and library versions can cause small run-to-run differences even with deterministic mode enabled.
- The trained checkpoint is not redistributed; reference metrics require retraining or access to the local artifact.
- Dataset images are not redistributed and must be obtained under their original terms.
- The reference environment used CUDA 12.8; PyTorch installation is platform-specific and intentionally not forced by `requirements.txt`.

## Dataset scope

KolektorSDD2 represents one constrained surface and acquisition setup. Results should not be generalized to different materials, cameras, lighting, defect types, or factory conditions without new data and validation.
