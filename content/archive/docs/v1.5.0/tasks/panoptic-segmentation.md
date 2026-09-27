---
title: Panoptic segmentation
seo_title: "Panoptic segmentation in LibreYOLO"
description: "Assign every pixel one segment in LibreYOLO: the families that serve the task, the COCO-panoptic dataset format, and the predict and validate calls."
lead: "Panoptic segmentation assigns every pixel to exactly one non-overlapping segment, unifying countable object instances with amorphous background regions. The task key is panoptic."
keywords: [panoptic segmentation python, panoptic quality, things and stuff segmentation, COCO panoptic format, segment id map, PQ metric]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # The -panoptic suffix in the filename selects the task, so no task
        # argument is needed.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) segment ids
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: One segment at a time
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # boolean (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: A smaller checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() returns a plain dict, not an object.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
---

## Definition

Panoptic segmentation is the union of the other two segmentation tasks. Every
pixel gets exactly one segment, segments never overlap, and a segment is either
a thing, a countable object instance, or stuff, an amorphous region such as sky
or road. That makes it stricter than
[instance segmentation](/docs/tasks/instance-segmentation), which leaves
background pixels unassigned and lets masks overlap, and stricter than
[semantic segmentation](/docs/tasks/semantic-segmentation), which labels every
pixel but merges touching instances of one class.

`panoptic` is the canonical task key, and the `-panoptic` suffix in a
checkpoint filename selects it, so `task=` is not needed when loading published
weights.

`predict()` fills `result.panoptic`. `.data` is an `(H, W)` integer
segment-id map on the original image canvas. `.segments_info` is a list of
dicts, one per segment, each carrying at least `{"id", "category_id"}`, where
`id` matches a value in the map and `category_id` indexes `result.names`.
`.segment_ids` lists the ids present in sorted order and `.segment_mask(id)`
returns the boolean `(H, W)` selection for one segment. Segment id `0` is the
void value: unlabeled pixels, excluded from the metric and left out of
`.segment_ids`.

Thing versus stuff is a property of the category, not of the individual
segment. It is carried on the label set's category metadata, and a prediction
payload may copy it onto each segment as `"isthing"` for convenience, but the
category metadata remains authoritative.

## Models

[EoMT](/docs/models/eomt) is the family that serves this task through
`LibreYOLO()`. It runs on the base package and ships panoptic checkpoints in
three sizes, s, b and l, trained on COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) also emits panoptic maps. It
is a prompted generative model with its own factory, `LibreVLM`, and its own
extra; with no vocabulary set it falls back to the COCO panoptic categories it
was tuned on. Its weights are non-commercial. Per-image latency is far higher
than a purpose-built segmenter, because every prediction is a diffusion decode.

## Predict

Weights download from Hugging Face on first use and are cached locally.

<code-tabs name="predict" />

`conf` filters query selection. See [prediction](/docs/predict) for sources,
streaming and result handling.

## Dataset format

LibreYOLO adopts the COCO-panoptic format verbatim, from Kirillov et al.,
CVPR 2019. There is no LibreYOLO-specific panoptic layout.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Each image is paired with one RGB PNG at the same resolution, where each
pixel's color encodes the id of the segment it belongs to:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Segment id `0`, RGB black, is void: unlabeled pixels that neither reward nor
penalize a prediction. Every other pixel belongs to exactly one segment.

The JSON lists, per image, the segment-id PNG and the segments inside it:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` names the PNG inside the panoptic directory, and
`segments_info[].id` matches a value in that PNG. `iscrowd` marks group
regions: they are never counted as false negatives, and a prediction that
mostly covers one is not a false positive. `isthing` lives on `categories` and
never on an individual segment.

The YAML points at both:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` and `panoptic_dir` each accept a single path or a per-split
mapping. Raw COCO category ids are typically non-contiguous, while models
predict a contiguous `0..nc-1`, so ids are remapped through `names` by category
name. A JSON category missing from `names` is an error rather than a silent
drop, because dropping it would score as a permanent false negative.

The canonical loader is `libreyolo.data.PanopticDataset`.

## Train

No family trains panoptic segmentation in LibreYOLO today: EoMT's `train()`
raises `NotImplementedError`, so panoptic checkpoints are used as published.

## Validate

`val()` returns a plain dictionary of `metrics/` keys, computed at the ground
truth resolution over the split named by `val` in the dataset YAML. A predicted
and a true segment of the same category match when their IoU exceeds 0.5, and
that match is unique.

<code-tabs name="val" />

`metrics/PQ` is Panoptic Quality, the headline number. Within one category it
is the product of two factors. Segmentation quality is the mean IoU over
matched segments and says how well the matched shapes line up. Recognition
quality is `TP / (TP + 0.5 FP + 0.5 FN)`, the F1 score of the matching itself,
and says how many segments were found at all. All three figures are then
averaged over the categories that appeared, and reported as `metrics/PQ`,
`metrics/SQ` and `metrics/RQ`, so the reported PQ is the mean of per-category
products rather than the product of the two reported means.

`metrics/PQ_things` and `metrics/PQ_stuff` average the same per-category PQ
over thing categories and stuff categories separately, and
`metrics/categories` counts the categories that appeared and were therefore
averaged over. The dictionary also carries `fitness`, a copy of the PQ value.

## Export

Panoptic checkpoints do not export. `export()` raises `NotImplementedError` for
this task, because the query-mask output has no runtime export contract yet.
EoMT's semantic task does export; see
[semantic segmentation](/docs/tasks/semantic-segmentation) and
[export and deploy](/docs/export).
