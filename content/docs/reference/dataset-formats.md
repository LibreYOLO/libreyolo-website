---
title: Dataset formats
seo_title: "LibreYOLO dataset formats for every task"
description: "The dataset-file contract per canonical task: YAML keys, folder layouts, label rows, mask and map conventions, and the loader that reads each one."
lead: "This page mirrors the dataset-file contract in the library's own docs/dataset_schema.md. It covers the YAML keys and on-disk layout each canonical task expects."
keywords:
  - libreyolo dataset format
  - yolo label format
  - data.yaml
  - segmentation mask dataset
  - coco panoptic format
  - depth dataset
  - pose kpt_shape
last_verified: "1.5.0"
verification: "Mirrors docs/dataset_schema.md in the libreyolo repository at v1.5.0, with loader names cross-checked against libreyolo/data/."
snippets:
  usage:
    - label: Parse one detection label row
      language: python
      code: |
        from libreyolo.data import parse_yolo_label_line

        # class_id cx cy w h, normalized to [0, 1]
        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480, num_classes=80)

        # (class_id, x1, y1, x2, y2, area) in pixels
        print(row)
---

## Common YAML

Applies to `detect`, `segment`, `pose` and `obb`.

| Key | Required | Meaning |
|---|---|---|
| `path` | | Dataset root |
| `train` | For training | Training images |
| `val` | For validation | Validation images |
| `test` | | Test images |
| `names` | Yes | Class list, or an integer-keyed mapping |
| `nc` | | Class count; must match `names` when present |
| `download` | | Download instructions; Python scripts need explicit opt-in |
| `annotations` | | Split to native COCO JSON file, for detect, segment and obb |

`train`, `val` and `test` may be image directories, image-list `.txt` files,
or lists of those. Label paths follow one substitution:

```text
images/.../image.jpg -> labels/.../image.txt
```

For a native COCO JSON dataset, `annotations` maps a split to its JSON file
and the split path gives the image root:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

When `names` is present, native COCO JSON category names must match the YAML
class names, and those names define the model label IDs. Without `names`, COCO
category IDs are sorted and mapped densely to `0..N-1`.

A dataset YAML does not carry a `task` key. Explicit model and task selection
wins.

Rules common to every text label file:

- one `.txt` label file per image;
- a missing or empty label file means no objects;
- `class_id` is an integer in `0..nc-1`;
- coordinates are finite normalized floats in `[0, 1]`;
- coordinates are relative to the original image width and height;
- rows carry no confidence and no track ID.

<code-tabs name="usage" />

## detect

Exactly five fields per row:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` is a normalized axis-aligned box, and `w` and `h` must be
positive.

## segment

A polygon row:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` is at least 3, the coordinate count after `class_id` must be even, and the
polygon must be non-degenerate. A five-field detection row is also accepted and
represents a rectangular segment.

## pose

YAML adds `kpt_shape`, which is required and is `[K, 2]` or `[K, 3]`, and the
optional `flip_idx`, an integer permutation of `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

The field count is exactly `5 + K * D`, where `D` is the second `kpt_shape`
value. Keypoint coordinates are normalized. Visibility `v`, when present, is
`0`, `1` or `2`.

## obb

Exactly nine fields:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

The four points are normalized image coordinates in `[0, 1]` and form a
non-degenerate oriented rectangle. No angle is stored in the label file.

The canonical parser is strict by default and rejects out-of-range
coordinates. Dataset and validation ingestion may clip coordinates to `[0, 1]`
for otherwise valid crop-boundary labels, then still reject degenerate boxes.
Parsing is task-aware: nine fields mean `obb` only in `obb` mode, while in
`segment` mode they may be a four-point polygon.

Internally, normalized corners are converted to canonical `xywhr`, with the
angle in radians representing rotation of the width side around the box center.
Public results expose OBB detections as `xywhr, conf, cls` rows.

Native COCO JSON OBB loading accepts annotations in this priority order:
`obb` as eight pixel-space corners; `obb` as `[cx, cy, w, h, angle]` with the
angle in radians; a COCO `segmentation` polygon or RLE, refit to a
minimum-area rectangle; and a COCO `bbox`, read as axis-aligned and
canonicalized.

Mosaic and mixup are disabled for OBB training until corner-aware OBB
augmentation exists.

The canonical row parser is `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Each image pairs with a dense single-channel mask in a lossless format,
typically PNG, instead of a `.txt` file:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

The mask is single channel, and palette-mode PNGs are read as palette indices.
Each pixel value is a class ID in `0..nc-1`, pixel value `255` means ignore and
is excluded from loss and metrics, and the mask resolution must equal the
image resolution.

Two optional YAML keys sit on top of the common contract. `masks_dir` is the
mask directory name substituted for `images` in each image path, defaulting to
`masks`. `label_mapping` is a `{source_id: train_id}` remap applied to mask
pixel values at load time, where unmapped source values become ignore and
train IDs must fall in `0..nc-1`.

When `masks_dir` is omitted, masks are rasterized at load time from `segment`
polygon labels resolved through the `images` to `labels` convention, and a
`background` class is appended after the object classes, so `nc` grows by one.

Canonical loader: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO adopts the COCO-panoptic format verbatim (Kirillov et al., CVPR
2019). There is no LibreYOLO-specific panoptic format.

One RGB PNG per image, at the image resolution, encodes each pixel's segment
ID in its color:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Every pixel belongs to exactly one segment and segments never overlap. Segment
ID `0`, RGB black, is void: unlabeled pixels excluded from the metric.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` names the segment-ID PNG inside `panoptic_dir`, and
`segments_info[].id` matches a value in that PNG. `iscrowd` marks group
regions: they are never false negatives, and a prediction mostly covering one
is not a false positive.

Thing-versus-stuff is a per-category property. `isthing` lives on
`categories`, never on `segments_info`.

COCO-panoptic `category_id` values are the dataset's raw IDs and are typically
non-contiguous. Models predict contiguous `0..nc-1`, so raw IDs are remapped
through the YAML `names` by category name, the same rule the native COCO JSON
detect loader follows. A JSON category absent from `names` is an error rather
than a silent drop, because it would otherwise score as a permanent false
negative.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` and `panoptic_dir` accept either a single path or a per-split
mapping.

Validation reports Panoptic Quality, computed at the ground-truth resolution
and averaged over the categories that appear, then split into `PQ_things` and
`PQ_stuff`. Matching is unique: a predicted and a ground-truth segment of the
same category match when IoU is above 0.5.

Canonical loader: `libreyolo.data.PanopticDataset`.

## depth

Each image pairs with a dense single-channel depth map:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

The map is a single-channel PNG or TIF, or a `.npy` file, at the image
resolution. Values are plain depth in a dataset-consistent unit. Zero,
negative, NaN and infinite values mark invalid pixels and are excluded from
loss and metrics.

| Key | Default | Meaning |
|---|---|---|
| `depths_dir` | `depths` | Depth directory substituted for `images` |
| `depth_stem_suffix` | | Suffix appended to the image stem; when omitted both the same stem and a `_depth` suffix are tried |
| `depth_mask_suffix` | `_mask` | Suffix for a validity mask; mask values at or below zero, NaN and infinite invalidate the depth pixel |
| `depth_scale` | `256.0` | Divisor for integer-typed depth maps, the common 16-bit PNG convention |

Float `.npy` maps are used as-is and do not apply `depth_scale`.

Canonical loader: `libreyolo.data.DepthDataset`.

## edge

Each RGB image pairs with a same-stem single-channel lossless map and an
optional validity mask:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

The map is single-channel PNG or TIF, not an RGB visualization, at the image
resolution. Integer maps are divided by their dtype maximum; float maps must
already be finite and in `[0, 1]`. `0` means non-edge and `1` means edge.
Optional mask pixels are valid when nonzero. Resizing uses nearest-neighbor
interpolation for targets and masks, and padded pixels are invalid and do not
contribute to validation.

| Key | Default | Meaning |
|---|---|---|
| `edges_dir` | `edges` | Edge-map directory substituted for `images` |
| `edge_stem_suffix` | | Suffix appended to image stems |
| `edge_extension` | `.png` | Lossless target extension |
| `edge_invert` | | Set true when source maps store black edges over white |
| `masks_dir` | `masks` | Optional validity-mask directory |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Validation thins continuous predictions with four-direction gradient
non-maximum suppression and reports ODS and OIS F-measures over a configurable
threshold sweep. Predicted and ground-truth pixels are matched one-to-one
within `edge_max_dist * image_diagonal`, with a default normalized tolerance
of `0.0075`.

Canonical loader: `libreyolo.data.EdgeDataset`. The loader is format-only: it
does not download or redistribute benchmark data.

## normal

Each image pairs with a same-stem three-channel 16-bit PNG, plus an optional
same-stem validity mask:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

The PNG is exactly three-channel `uint16` with channels stored as RGB, at the
image resolution. Decode with `n = png / 65535 * 2 - 1`, then renormalize each
vector. Decoded vectors use the OpenCV camera frame, `+x` right, `+y` down,
`+z` into the scene, and face the camera. The optional mask is a
single-channel PNG where nonzero means valid; without a mask, every finite,
nonzero decoded vector is valid. Invalid and padded target pixels are
represented internally by `(0, 0, 0)`. Resizing interpolates the three
components bilinearly and then renormalizes, validity masks use
nearest-neighbor interpolation, and a horizontal flip also negates the x
component.

| Key | Default | Meaning |
|---|---|---|
| `normals_dir` | `normals` | Normal-map directory substituted for `images` |
| `masks_dir` | `masks` | Optional validity-mask directory |

Validation reports mean and median angular error in degrees and the percentage
of valid pixels within 11.25, 22.5 and 30 degrees.

Canonical loader: `libreyolo.data.NormalDataset`.

## restore

Each degraded input image pairs with a clean RGB target:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Input and target are RGB-compatible image files and their resolutions must
match exactly. Validation keeps native resolution and pads only enough to
stack a batch, and metrics are computed on the original image canvas. Training
applies a coupled crop and horizontal flip to the input and target pair.

| Key | Default | Meaning |
|---|---|---|
| `input_dir` | `inputs` | Degraded-input directory used in split paths |
| `target_dir` | `targets` | Clean-target directory substituted for `input_dir` |
| `target_stem_suffix` | | Suffix appended to the input stem before target lookup |
| `target_stem_suffixes` | | List form of `target_stem_suffix` |
| `degradation` | | Metadata label such as `deblur` or `denoise` |
| `dataset` | | Dataset or provenance label |

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: image}`. Restore models expose `Results.restored`, not detections.

Canonical loader: `libreyolo.data.RestoreDataset`.

## matte

Each RGB image pairs with a single-channel ground-truth matte sharing
the same stem, where 0 is background and 255 is foreground:

```text
images/subject.jpg -> mattes/subject.png
```

Two layouts are accepted. A directory root containing `images/` and a matte
directory, auto-detected among `mattes/`, `matte/`, `gt/`, `masks/`, `mask/`
and `alpha/`, passed as `data=`. Or a YAML with `path` plus per-split
`val_images` and `val_mattes`, and optionally `train_images` and
`train_mattes`, each relative to `path` or absolute.

The matte is grayscale and read as opacity in `[0, 1]`, and it is resized to the
prediction canvas with bilinear interpolation when the shapes differ. Metrics
are MAE and S-measure (Fan et al., ICCV 2017) on the original image canvas,
with S-measure as the best-checkpoint fitness.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: matte}`. Matte models expose `Results.matte`.

Validation is inference-only in this version. Canonical pair resolver:
`libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Labels are one JSONL file per split, one JSON object per image:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` is a four-point quad in absolute pixel coordinates, ordered
top-left, top-right, bottom-right, bottom-left. Regions with unreadable text
use `"text": "###"`, the ICDAR do-not-care convention: they are excluded from
recognition scoring, and predictions overlapping them are ignored rather than
penalized in detection matching.

Metrics are detection hmean with one-to-one polygon matching above IoU 0.5,
end-to-end F1 requiring both IoU above 0.5 and an exact transcript after NFKC
normalization and whitespace removal, case-sensitive, and 1-NED on matched
pairs. Best-checkpoint fitness is end-to-end F1.

Two layouts are accepted: a directory root containing `images/<split>/` and
`labels/<split>.jsonl`, passed as `data=`, or a YAML with `path` plus optional
`images` and `labels` directory names.

The class-like YAML fields are schema placeholders: use `nc: 1` and
`names: {0: text}`. OCR models expose `Results.ocr`.

Validation is inference-only in this version. Canonical sample resolver:
`libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

An ImageFolder-style directory tree, not label files:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` is required for training and defines the class-to-index mapping by
sorted folder name. `val/` is required for validation. `test/` may be present
but the default train and val commands do not use it. Non-training splits must
contain the same class folder names as the expected train or checkpoint class
set. Supported image extensions are defined in
`libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze and point

No training or validation dataset-file contract is implemented for `gaze`.

`point` is a model-output task rather than a dataset-label schema. Point
families may adapt existing labels internally, for example by deriving object
centers from box rows, but a point-only text label format is not defined.
