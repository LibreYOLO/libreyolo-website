---
title: Working with results
seo_title: "The LibreYOLO Results object"
description: "One Results object per image, with a slot per payload type: boxes, masks, keypoints, probs, depth, panoptic, OCR and more. Plotting, saving and JSON."
lead: "Every prediction returns a Results object per image. It has one named slot per kind of payload, all of them empty except the ones the model produces, plus the same slots on an exported artifact."
keywords:
  - yolo results object python
  - results.boxes xyxy
  - results to json
  - save annotated image
  - segmentation masks python
  - keypoints results
  - depth map results
  - results summary
  - onnx same results
last_verified: "1.5.0"
verification: "Payload classes, slots, move semantics, summary(), to_json(), plot(), save() and cutout() read from libreyolo/utils/results.py. Annotation and disk-writing behavior from InferenceRunner._save_annotated_image in libreyolo/models/base/inference.py and resolve_save_path in libreyolo/utils/general.py. Suffix dispatch from LibreYOLO() in libreyolo/models/__init__.py."
snippets:
  basic:
    - label: Boxes
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # (height, width) of the source image
        print(result.path)         # source path, None for in-memory input

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Normalized coordinates
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # pixels, x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # pixels, center x, center y, w, h
        print(result.boxes.xyxyn[:1])   # same box divided by width and height
        print(result.boxes.xywhn[:1])
    - label: NumPy and devices
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Each of these returns a new Results; the original is unchanged.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary and to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Same content as a string, with the same keyword arguments.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Annotated images
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # save=True draws the payload and writes it under runs/detect/predict*.
        result = model(SAMPLE_IMAGE, save=True)
        print(result.saved_path)
  exported:
    - label: Install the export extra
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: The same Results from an exported artifact
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # returns the written path

        # LibreYOLO() dispatches on the file suffix.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
---

## One object, one slot per payload

A prediction on one image returns one `Results`. It carries eighteen payload
slots, and a model fills only the ones its task produces. Every other slot is
`None`, so reading `result.masks` on a detector is `None` rather than an error.

| Slot | Class | Shape | Produced by |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` plus scores and classes | Detection, and any task that localizes first |
| `masks` | `Masks` | `(N, H, W)` | Instance segmentation |
| `keypoints` | `Keypoints` | `(N, K, 2)` or `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Classification |
| `obb` | `OBB` | `(N, 7)` or `(N, 8)` | Oriented boxes |
| `gaze` | `Gaze` | `(N, 2)` pitch and yaw in radians | Gaze estimation |
| `points` | `Points` | `(N, 4)` as x, y, class, confidence | Point localization |
| `semantic_mask` | `SemanticMask` | `(H, W)` class ids | Semantic segmentation |
| `panoptic` | `PanopticSegmentation` | `(H, W)` segment ids plus `segments_info` | Panoptic segmentation |
| `depth_map` | `DepthMap` | `(H, W)` floats | Depth estimation |
| `normal_map` | `NormalMap` | `(H, W, 3)` unit vectors | Surface normals |
| `edges` | `EdgeMap` | `(H, W)` floats in `[0, 1]` | Edge detection |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restoration and super-resolution |
| `matte` | `Matte` | `(H, W)` floats in `[0, 1]` | Alpha matting and background removal |
| `ocr` | `OCRRegions` | `(N, 4, 2)` polygons plus transcripts | Text detection and recognition |
| `embeddings` | `Embeddings` | `(N, D)` L2-normalized rows | The `embed` task |
| `identities` | `Identities` | N names and scores | The `embed` task with a gallery |
| `meshes` | `Meshes` | Body parameters and optional vertices | Body mesh recovery |

Alongside them sit the fields every result has: `orig_shape` as
`(height, width)`, `path` (the source path, or `None` for in-memory input),
`names` mapping class id to class name, `frame_idx` for video and live frames,
`track_id` when tracking, and `restore_scale`, the integer upscale factor of a
restoration result.

`result.normals` is an alias for `result.normal_map`.

`result.speed` exists on every result but is populated only by
[ensembles](/docs/predict/ensembling), where its keys are `member_0`,
`member_1` and `fusion` in milliseconds. For a single model it stays an empty
dict.

## Boxes

<code-tabs name="basic" />

`Boxes` keeps coordinates and scores as separate arrays rather than one packed
tensor.

| Attribute | Contents |
|---|---|
| `xyxy` | `(N, 4)` absolute pixels, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` absolute pixels, center x, center y, width, height |
| `xyxyn`, `xywhn` | The same divided by image width and height |
| `conf` | `(N,)` confidence |
| `cls` | `(N,)` class id, as a float |
| `id` | `(N,)` track id, or `None` |
| `is_track` | Whether `id` is set |
| `data` | Everything concatenated: boxes, optional id, conf, cls |

`cls` is a float array, so use it as `result.names[int(cls)]`.

`xyxyn` and `xywhn` need `orig_shape`, which `Results` fills in for you.

## Dense payloads

Payloads covering the whole image behave differently from per-instance ones,
and it matters when slicing.

`SemanticMask` holds `(H, W)` class ids on the original canvas, with `255`
reserved as the ignore value that never counts as a class. `classes` lists the
ids present and excludes it; `class_mask(id)` returns a boolean `(H, W)`.

`PanopticSegmentation` holds `(H, W)` segment ids, with `0` as the void id, and
a `segments_info` list of dicts carrying at least `id` and `category_id`.
`segment_ids` lists the ids present, `segment_mask(id)` selects one.

`DepthMap` holds `(H, W)` relative inverse depth: higher means closer, and the
values are not metric meters. It exposes `min`, `max`, `mean` over finite
values, and `normalized()` rescaling to `[0, 1]`.

`NormalMap` holds `(H, W, 3)` unit vectors in the OpenCV camera frame, with
`+x` right, `+y` down and `+z` into the scene, so a surface facing the camera
is `(0, 0, -1)`. `assert_normalized()` checks every pixel is finite and unit
length.

`EdgeMap` holds `(H, W)` float32 in `[0, 1]`. The continuous map is kept rather
than thresholded, so `binary(threshold=0.5)` is where you choose a cutoff.

`Matte` holds `(H, W)` float32 in `[0, 1]`, where `1` is fully foreground.
`array` returns it clipped as float32.

`RestoredImage` holds `(H, W, 3)` uint8 RGB, with `array` for the raw ndarray
and `save(path)` to write it.

`Probs` holds one probability vector for the image. `top1` and `top5` are class
indices, `top1conf` and `top5conf` the matching scores.

`Embeddings` holds `(N, D)` rows that are already L2-normalized, so cosine
similarity is a dot product. `similarity(other)` returns `(N, M)` against a
gallery or `(N,)` against a single vector, and `verify(i, j, threshold=0.4)`
compares two rows.

`OCRRegions` holds `(N, 4, 2)` polygons in reading order, corners ordered
top-left, top-right, bottom-right, bottom-left. Transcripts are in `texts`,
recognition scores in `conf`, detection scores in `det_conf`. Because these are
genuine rotated polygons they do not populate `boxes`; `ocr.xyxy` gives
axis-aligned hulls when you need rectangles.

## Slicing and moving

`result[i]` returns a new `Results` holding one instance. Per-instance payloads
are sliced; whole-image payloads are carried through unchanged, so slicing a
classification result cannot truncate its probability vector to a single class,
and slicing a depth result cannot corrupt the `(H, W)` layout.

`len(result)` counts instances: boxes, points, embeddings, OCR regions or
meshes. Any dense whole-image payload counts as `1`. A result with nothing in
it is `0`.

`to()`, `cpu()`, `cuda()` and `numpy()` each return a new `Results` with every
populated slot converted. They do not modify the original.

`update()` is the one method that mutates in place, replacing named slots and
returning the same object.

## JSON

<code-tabs name="json" />

`summary()` returns a list of plain dicts, and `to_json()` is that list passed
through `json.dumps`. Both take the same three arguments: `normalize=False`
switches coordinates to `[0, 1]`, `decimals=5` sets rounding, and
`embeddings=False` controls whether embedding vectors are included.

The row shape follows the payload. Detection rows carry `name`, `class`,
`confidence` and a `box` dict, and pick up `segments` when masks are present,
`obb` and `corners` for oriented boxes, `gaze` angles in both radians and
degrees, `track_id` when tracking, and `mesh` parameters when meshes are
present.

Where there are no boxes, one payload decides the rows: OCR emits one row per
region with its `text`, points one row per point, panoptic one row per segment
with `pixel_count` and `pixel_fraction`, semantic one row per class present,
classification the top five classes. Depth, normals, edges, restoration and
matting each emit a single summary row describing the map rather than its
pixels.

Two payloads are deliberately abbreviated. An embedding vector is reported as
`embedding_dim` only, because a 512-float row is about 2 KB per face; pass
`embeddings=True` to include the values. Mesh vertices are never included at
all, since that is tens of thousands of coordinates per person. Read
`result.meshes.vertices` or call `result.meshes.save_obj(path)` for geometry.

## Drawing and saving

<code-tabs name="saving" />

`predict(save=True)` is the path that annotates and writes. It picks the
drawing routine from whichever slot is filled, so a semantic result is written
as a colored mask, a depth result as a depth visualization, a panoptic result
with its segments, a matte as a transparent-background RGBA PNG, and a detector
as boxes with masks underneath them. The written path is attached to the result
as `result.saved_path`.

`Results.plot()` is narrower than its name suggests. It is defined for normal
maps and edge maps only, and raises `NotImplementedError` for anything else.
Use `save=True` for the other tasks.

`Results.save(path)` is likewise narrow: it writes a matte result as a
transparent-background RGBA PNG cutout and raises `NotImplementedError`
otherwise. `Results.cutout()` returns that same RGBA array without writing it.
Both need the source image, taken from `result.path` or passed as `image=`.

Two payloads carry their own writers: `result.restored.save(path)` for a
restored image, and `result.meshes.save_obj(path, index=0)` for a mesh.

For where files land and how `output_path` and `output_file_format` behave, see
[Prediction sources](/docs/predict/sources).

## Exported artifacts return the same object

<code-tabs name="exported" />

`LibreYOLO()` dispatches on the file suffix, so an exported artifact loads
through the same call as a `.pt` checkpoint and returns the same `Results`.
`.onnx`, `.engine`, `.pte` and `.mnn` files are recognized by suffix, as are
OpenVINO, Paddle and ncnn directories and a Triton model URL. Code that reads
`result.boxes.xyxy` does not change when a model is swapped for its exported
build. See [Export](/docs/export) for the full set of formats.

Reaching for the runtime's own API instead means owning preprocessing,
postprocessing and class names yourself.
