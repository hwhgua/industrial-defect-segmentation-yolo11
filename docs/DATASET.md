# Dataset preparation

## Source

KolektorSDD2 is obtained from the [official ViCoS mixed-segmentation repository](https://github.com/vicoslab/mixed-segdec-net-comind2021). This project does not mirror the archive, source images, masks, or the original helper code.

Review the source repository's CC BY-NC-SA 4.0 terms before downloading or using the data. Cite the dataset and associated paper as requested by its maintainers.

## Expected source layout

```text
KolektorSDD2/
├── train/
│   ├── <image>.png
│   └── <image>_GT.png
└── test/
    ├── <image>.png
    └── <image>_GT.png
```

Each image must have a same-stem `_GT.png` binary mask. A blank mask represents a normal image.

## Formal protocol

The preparation command splits the original training set into train and validation subsets using defect-aware stratification with seed 42. The original test directory remains the test subset. It then converts every binary mask into one or more normalized YOLO polygons and writes empty label files for normal images.

```bash
ksdd2-prepare \
  --source data/raw/KolektorSDD2 \
  --output datasets/kolektorsdd2_yolo_seg_formal \
  --val-ratio 0.2 \
  --seed 42
```

The tool builds into a staging directory and moves the completed dataset into place. `--overwrite` only replaces directories that contain its own generation marker; it refuses to delete an arbitrary directory.

The generated `data.yaml` intentionally omits the optional `path` field. In Ultralytics 8.3.221, a literal `path: .` can resolve against the process working directory rather than the YAML directory. Omitting it makes Ultralytics use the YAML file's parent as the dataset root.

## Verification

```bash
ksdd2-verify \
  --dataset datasets/kolektorsdd2_yolo_seg_formal \
  --output reports/metrics/dataset_verification
```

Checks cover image/label pairing, Pillow readability, class IDs, polygon coordinate shape, `[0,1]` normalization, and cross-split stem overlap.

Preparation fails explicitly if an image and mask have different dimensions, if a foreground mask is removed entirely by the polygon area threshold, or if a material enclosed hole cannot be represented faithfully by YOLO polygons. A failed `--overwrite` build leaves the previous generated dataset in place until the replacement has completed successfully.
