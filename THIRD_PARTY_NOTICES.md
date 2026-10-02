# Third-party notices

## Ultralytics

This project imports the open-source Ultralytics package and uses the YOLO11 segmentation model interface. The reference version is `ultralytics==8.3.221`; see `pyproject.toml` for the declared runtime dependencies.

- Project: [Ultralytics](https://github.com/ultralytics/ultralytics)
- License for the open-source software distribution: GNU Affero General Public License v3.0 (AGPL-3.0)
- License guidance: [Ultralytics licensing](https://docs.ultralytics.com/help/contributing/)

The dependency's software license does not by itself establish the redistribution terms for every separately distributed model asset. This repository includes no Ultralytics pretrained weights, trained checkpoints, or Ultralytics source fork. Obtain and use any model asset only after reviewing the applicable terms for that specific asset and use case. Ultralytics names and trademarks remain with their respective owner.

## KolektorSDD2 / ViCoS

The experiment uses KolektorSDD2 from the ViCoS Laboratory project:

- Source: [mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021)
- License stated by the source repository for its code and dataset: CC BY-NC-SA 4.0
- Dataset paper and citation details: see the official source repository and its linked publication.

KolektorSDD2 images, masks, archives, converted labels, and dataset-derived previews are not included here. Users must obtain the data from the source and follow the license terms, including attribution, share-alike, and non-commercial conditions where applicable. Check the source terms for the intended use; attribution or citation alone does not authorize a use restricted by the license. This repository does not grant rights to the dataset or its derivatives.

## Results and model artifacts

The numerical tables in `docs/RESULTS.md` and `metadata/reference_results.json` are an aggregate record of the local experiment. They are not model weights and do not include image content. No pretrained weights, trained checkpoint, raw or converted dataset, prediction overlay, contact sheet, or other dataset-derived visual example is distributed by this repository. Locally generated files under `datasets/`, `runs/`, and `reports/` are excluded from Git.

If a checkpoint, prediction image, or other derived artifact is proposed for distribution later, first identify its source and applicable terms and document the basis for redistribution. Do not assume that the code license covers it.

## Separation of licenses

The `LICENSE` file applies to original project code only. Third-party software dependencies, datasets, model assets, names, and dataset-derived content remain under their own terms. The presence of a dependency in `pyproject.toml`, a citation, or a reference metric does not transfer ownership or grant redistribution permission for those materials.
