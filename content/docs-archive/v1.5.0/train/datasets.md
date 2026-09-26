---
title: Datasets
seo_title: "Training datasets in LibreYOLO"
description: "The dataset YAML LibreYOLO reads, the folder layout it expects, how autodownload works, and the doctor command that checks a dataset before training."
lead: "A LibreYOLO dataset is a YAML file naming a root, its splits and its class names. Everything else, including where the label files live, is derived from that file by convention."
keywords:
  - yolo dataset format
  - data.yaml
  - custom dataset training
  - yolo label format
  - coco json dataset
  - dataset autodownload
  - libreyolo doctor
  - class imbalance check
  - train val split leakage
last_verified: "1.5.0"
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # A bundled name, a relative path or an absolute path all work.
        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Check a dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Fail a CI job on warnings too
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Skip the image decode pass
      language: bash
      code: |
        # Reads labels and YAML only. Corruption, duplicate and split-leakage
        # checks all need the pixels, so they are skipped.
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
---

## Point train at a dataset

`data=` takes a YAML path or the name of a config that ships with the package.

<code-tabs name="train" />

The name is resolved in a fixed order: an absolute path that exists, then the
name as given relative to the working directory, then the same name with
`.yaml` appended, then the bundled config directory. When nothing matches, the
error names every directory that was searched and lists the bundled configs.

## Bundled configs

Thirteen dataset configs ship inside the package, under
`libreyolo/config/datasets/`.

| Config | Task | Notes |
|---|---|---|
| `coco8.yaml` | detect | 8 images, downloads from a plain URL |
| `coco128.yaml` | detect | 128 images |
| `coco1000.yaml` | detect | 800 train, 200 val |
| `coco5000.yaml` | detect | 4000 train, 1000 val |
| `coco.yaml` | detect | full COCO 2017 |
| `coco-val-only.yaml` | detect | val2017 only |
| `coco8-pose.yaml` | pose | 8 images, COCO-17 keypoints |
| `coco-pose.yaml` | pose | COCO 2017 keypoints |
| `ade20k.yaml` | semantic | 150 classes |
| `cityscapes.yaml` | semantic | 19 classes, download by hand |
| `cocostuff.yaml` | semantic | 182 classes, download by hand |
| `gopro.yaml` | restore | deblurring pairs |
| `sr8.yaml` | restore | super-resolution pairs |

Only `coco8.yaml` and `coco128.yaml` carry a plain download URL. The rest either
carry a Python download block, which needs the opt-in described below, or expect
the data to already be on disk.

## Where a dataset lives on disk

The YAML `path` key names the dataset root. An absolute `path` is used as
written. A relative one is looked for under the datasets directory first, then
beside the YAML file itself, and a dataset that is about to be downloaded goes
under the datasets directory.

That directory is `~/datasets`, overridden by the `LIBREYOLO_DATASETS_DIR`
environment variable. There is no settings file for it.

## The YAML keys

```yaml
path: my-dataset        # dataset root
train: images/train     # required to train
val: images/val         # required to validate
test: images/test       # optional
nc: 3                   # optional; must agree with names
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # optional
```

`train`, `val` and `test` each accept an image directory, a `.txt` file listing
one image path per line, or a list mixing both. Lines in a `.txt` list may be
relative, in which case they resolve against the list file's own directory, and
lines starting with `#` are skipped.

`names` may be a list or an integer-keyed mapping. `nc` is optional; when both
are present and disagree, the doctor reports it as an error.

## Directory layout and label files

Detection, segmentation, pose and oriented boxes all share one layout. The label
path is derived from the image path by rewriting an `images` directory component
to `labels` and changing the extension to `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Only a whole `images` path component is rewritten, so a directory named
`images_old` is left alone.

A detection row is five fields, all normalized to `[0, 1]` against the original
image width and height:

```text
<class_id> <cx> <cy> <w> <h>
```

A missing or empty label file means the image has no objects, and it trains as
background rather than raising. A row with more than five fields is read as a
polygon and its box becomes the polygon's extent, so a segmentation export used
for detection training loads without complaint. The doctor reports how many rows
took that path.

## Other tasks

Segmentation keeps the same layout with polygon rows,
`<class_id> <x1> <y1> ... <xN> <yN>`, at least three points. A five-field
detection row is accepted and means a rectangular instance.

Pose adds `kpt_shape: [K, D]` and an optional `flip_idx` permutation to the YAML.
Each row is exactly `5 + K * D` fields: the box, then `K` keypoints of `x y` or
`x y v`, with visibility `0`, `1` or `2`.

Oriented boxes use exactly nine fields, the class followed by four corner points
in normalized coordinates. No angle is stored in the file.

Semantic segmentation pairs each image with a single-channel mask of the same
resolution, resolved by substituting `masks_dir` (default `masks`) for `images`.
Pixel value `255` means ignore. `label_mapping` remaps source ids to train ids at
load time.

Classification uses an ImageFolder tree instead of label files, with `train/` and
`val/` each containing one directory per class. The class-to-index mapping is the
sorted folder name order.

Restoration pairs a degraded input with a clean target of identical resolution
through `input_dir` and `target_dir`. Depth, surface normals and edges each pair
an image with a dense map through their own directory key.

The full per-task contract, including the depth scale conventions and the
panoptic segment-id PNG encoding, is `docs/dataset_schema.md` in the library
repository.

## Native COCO JSON

A COCO JSON annotation file can be used directly. Add an `annotations` mapping,
and the split path becomes the image root:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

When `names` is present, the JSON category names must match it, and `names`
defines the label ids the model predicts. Without `names`, COCO category ids are
sorted and mapped densely to `0..N-1`.

This path expects one image directory per split. A list of paths or a `.txt`
image list raises rather than silently loading a different set.

## Autodownload

A dataset counts as present when its `train` or `val` path resolves to a
non-empty directory or an existing file. When it does not, and the YAML has a
`download` key, the value decides what happens next.

An `http` or `https` URL is fetched and, if it is a zip, extracted into the
dataset root. Anything else is treated as an embedded Python script and runs only
when `allow_download_scripts=True`. Without that, the script is skipped with a
warning and training continues against whatever is on disk.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

The flag is a code-execution gate, not a network gate. URL downloads happen
either way; it is the `download: |` blocks that need it. The CLI prints a warning
when the flag is on, and the doctor never enables it.

## Check the dataset before you train

`libreyolo doctor` reads a detection dataset and reports what would go wrong
before a GPU is involved. It exits 1 when it finds errors, so it works as a CI
gate.

<code-tabs name="doctor" />

The checks come in six families:

| Family | Looks for |
|---|---|
| `config` | missing `names`, `nc` that disagrees with `names`, missing or empty splits, duplicate class names |
| `files` | images with no label file, labels with no image, missing images listed in a split, stem collisions |
| `labels` | malformed rows, class ids outside `[0, nc)`, coordinates outside `[0, 1]`, zero-area boxes, tiny or huge boxes, duplicate boxes, byte-identical label files |
| `balance` | classes with zero or few instances, class imbalance ratio, classes present in one split only, background image share |
| `images` | undecodable files, EXIF rotation, odd channel layouts, uniform images, exact and near duplicates |
| `splits` | the same image appearing in two splits, exactly or near-identically |

`--only` and `--skip` take a check id or a family prefix, so
`skip=images,labels.tiny_object` is valid. `--fast` drops every check that needs
to decode pixels, which is the `images` and `splits` families.

Two behaviors are worth knowing. `--strict` makes warnings fail the exit code as
well as errors. And the doctor covers detection datasets only: a pose, segment or
oriented-box dataset is rejected with a message naming what it detected, rather
than being checked against the wrong contract.

## Related

- [Hyperparameters](/docs/train/hyperparameters) for the arguments `train()`
  takes once the data is in place.
- [Validation and metrics](/docs/train/validation) for evaluating on the `val`
  or `test` split.
