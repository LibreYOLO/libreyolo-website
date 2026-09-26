---
title: Results types
seo_title: "LibreYOLO Results object reference"
description: "Every payload a LibreYOLO Results object can carry, one slot per task shape: boxes, masks, keypoints, probs, obb, depth, ocr, embeddings and ten more."
lead: "Results is the single per-image return type of every LibreYOLO model. It carries eighteen optional payload slots, one per task shape, and populates only the ones the model produced."
keywords:
  - libreyolo results object
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - libreyolo results to_json
last_verified: "1.5.0"
verification: "Slot names, shapes, properties and defaults read from libreyolo/utils/results.py at v1.5.0. Semantics quoted from the payload class docstrings."
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Every payload moves together.
        result = result.cpu().numpy()

        # Rows, as plain dicts, then as JSON.
        print(result.summary()[:1])
        print(result.to_json())
---

## The Results object

One `Results` describes one image. A single image source returns one of them,
a list source or a directory returns a list, and `stream=True` returns a
generator that yields them.

| Attribute | Type | Meaning |
|---|---|---|
| `orig_shape` | `(int, int)` | Original image height and width |
| `path` | `str` | Source path when the input came from disk |
| `names` | `dict[int, str]` | Class index to class name |
| `speed` | `dict[str, float]` | Per-stage milliseconds |
| `track_id` | tensor | Track IDs when the result came from `track()` |
| `frame_idx` | `int` | Frame index for video and stream sources |
| `restore_scale` | `int` | Output-to-input upscale factor of a restore result; `1` everywhere else |

<code-tabs name="usage" />

## Payload slots

Each slot is `None` unless the model produced it. The slot a family fills is
decided by its task.

| Slot | Class | Task |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, with a gallery |
| `meshes` | `Meshes` | mesh |

`result.normals` is a read-write alias for `result.normal_map`.

More than one slot can be set at once. A segmentation model fills both `boxes`
and `masks`; a gaze model fills `boxes` with the face boxes and `gaze` with the
angles; a mesh model fills `boxes` with person boxes and `meshes` row-aligned
to them.

## Boxes

Detection boxes for one image.

| Member | Returns |
|---|---|
| `xyxy` | Corner coordinates in original-image pixels |
| `xywh` | Center and size in pixels |
| `xyxyn` | Corners normalized to `[0, 1]` |
| `xywhn` | Center and size normalized to `[0, 1]` |
| `conf` | Confidence per box |
| `cls` | Class index per box |
| `id` | Track ID per box, or `None` |
| `is_track` | `True` when track IDs are present |
| `data` | The packed tensor |

`with_id(id)` and `with_orig_shape(orig_shape)` return a new `Boxes` with that
field replaced.

## Masks

Instance masks for one image. `data` is the mask tensor; `xy` returns
per-instance contours in pixels and `xyn` returns them normalized.

## Keypoints

Pose keypoints, row-aligned with `boxes`. `xy` is the coordinate pair per
keypoint and `xyn` the normalized pair. `conf` is the third channel when the
data carries one, otherwise `None`. `has_visible` is a boolean array, true
where `conf > 0`, and all-true when there is no confidence channel.

## Points

Point localization for one image. `data` has shape `(N, 4)` with rows
`x, y, class, confidence`. Coordinates are absolute pixels; `xy` and `cls` and
`conf` split the columns, and `xyn` normalizes the coordinates.

## Probs

Classification scores. `top1` is the winning index, `top5` the five best
indices, and `top1conf` and `top5conf` their scores.

## OBB

Oriented boxes. `data` holds 7 or 8 values per row: `xywhr`, an optional track
ID, then confidence and class.

| Member | Returns |
|---|---|
| `xywhr` | Center, size and rotation in radians |
| `xyxyxyxy` | The four corners in pixels |
| `xyxyxyxyn` | The four corners normalized |
| `xyxy` | Axis-aligned hull in pixels |
| `conf`, `cls`, `id`, `is_track` | As on `Boxes` |

## Gaze

Per-face gaze angles in radians, shape `(N, 2)`, row-aligned with the face
boxes in `boxes`. Column 0 is pitch and column 1 is yaw, in the L2CS
convention: positive yaw rotates the gaze toward the subject's left and
positive pitch rotates it downward. `pitch_deg` and `yaw_deg` convert to
degrees, and `direction_3d` returns the unit direction vector.

## SemanticMask

Dense semantic map, shape `(H, W)` of integer class IDs on the original image
canvas. `255` is the ignore value and never counts as a class
(`SemanticMask.IGNORE_INDEX`). `classes` lists the class IDs present, and
`class_mask(class_id)` returns the boolean mask for one class.

## PanopticSegmentation

Every pixel gets exactly one non-overlapping segment, unifying stuff regions
and thing instances. `data` is a `(H, W)` integer segment-ID map; segment ID
`0` is unlabeled (`PanopticSegmentation.IGNORE_INDEX`). `segments_info` is a
list of dicts, one per segment, each with at least `{"id": int,
"category_id": int}`, where `id` matches a value in the map and `category_id`
indexes `names`. `segment_ids` lists the IDs present and
`segment_mask(segment_id)` returns one segment's boolean mask.

Thing-versus-stuff is a property of the category, not of the segment. A
payload may denormalize it onto each segment as `"isthing": bool`, and when it
does, the value must agree with the category-level map.

## DepthMap

Dense relative inverse-depth map, shape `(H, W)` of floats on the original
image canvas. Higher values mean closer to the camera. Values are relative,
not metric meters. `min`, `max` and `mean` are computed over finite values,
and `normalized()` rescales the map to `[0, 1]`.

## NormalMap

Dense surface-normal field, float32 `(H, W, 3)` on the original image canvas,
in the OpenCV camera frame: `+x` right, `+y` down, `+z` into the scene.
Normals face the camera, so a fronto-parallel surface is `(0, 0, -1)`. Every
pixel is a unit vector. `assert_normalized(atol=1e-4)` checks that invariant.

## EdgeMap

Dense edge-probability map, float32 `(H, W)` on the original image canvas,
where `0` is non-edge and `1` is edge. The continuous map is kept so the
threshold stays the caller's choice: `binary(threshold=0.5)` applies one, and
`array` returns the numpy view.

## RestoredImage

The restored RGB image, `(H, W, 3)` uint8. For super-resolution the canvas is
`Results.restore_scale` times the input. `array` returns the numpy view and
`save(path)` writes the image.

## Matte

Soft opacity matte, float32 `(H, W)` in `[0, 1]` on the original image canvas.
`1` is fully foreground and `0` is fully background. A soft matte subsumes a
hard background-removal mask, thresholded at 0.5, and keeps the anti-aliased
edges that a binary mask discards. `array` returns the numpy view.

On a matte result, `Results.cutout(image=None)` returns an RGBA `(H, W, 4)`
uint8 array whose fourth channel is the matte, and `Results.save(path, image=None)`
writes that cutout as a transparent-background PNG. Both take the RGB from
`image` when given, otherwise they reload it from `Results.path`.

## OCRRegions

Located text with transcripts. `data` is `(N, 4, 2)` float polygons in
original-image pixels, ordered top-left, top-right, bottom-right,
bottom-left, and regions come in reading order, top to bottom then left to
right. `texts` is the list of N transcripts. `conf` is the per-region
recognition score and `det_conf` the detection score, both `(N,)`.

Detection quads are genuine polygons, so they do not populate
`Results.boxes`. `xyxy` gives the axis-aligned hulls.

## Embeddings

L2-normalized vectors from the `embed` task, always shape `(N, D)`. A
whole-image result carries one row and no boxes; region embeddings are
row-aligned with `boxes`. Because each row is normalized, cosine similarity is
a dot product.

| Member | Returns |
|---|---|
| `dim` | `D` |
| `normalized` | The rows, renormalized |
| `similarity(other)` | Pairwise cosine similarity against another `Embeddings` or tensor |
| `verify(i, j, threshold=0.4)` | `True` when rows `i` and `j` match |

## Identities

Named gallery matches, row-aligned with `embeddings`. Produced when a
`Gallery` is passed to an `embed` prediction. `name` is a list where an entry
is `None` below the match threshold, and the nearest below-threshold name is
never guessed. `score` is the match score array and `data` pairs them.

## Meshes

Parametric human body meshes, row-aligned with the person boxes in `boxes`.
Everything is in the camera frame of the original image. `transl` is metric in
meters with `+z` pointing away from the camera; `vertices` and `joints3d` are
metric and already include `transl`; `joints2d` is in pixels on the original
image canvas, not on the crop the network saw. No field carries a world or
gravity frame.

Parameter layouts differ between body models, so nothing about the shapes is
hard-coded. `body_model` names the parameterization and the counts are read
back from the tensors: `num_vertices`, `num_joints`, `num_betas`, and
`has_vertices`. `params` returns the parameter dict, and `save_obj(path,
index=0)` writes one mesh. Fields are `global_orient`, `body_pose`, `betas`,
`transl`, `vertices`, `faces`, `joints3d`, `joints2d`, `conf`,
`focal_length` and `extras`.

For `body_model="mhr"` the rotations are Euler angles in radians rather than
axis-angle, `body_pose` is a flat per-joint parameter vector rather than one
triplet per joint, and `betas` are identity blendshape coefficients. Skeleton
scale, hand pose and facial expression live in `extras`.

## Conversion and selection

Every payload carries `to(*args, **kwargs)`, `cpu()`, `cuda()` and `numpy()`,
and calling one of them on the `Results` applies it to every populated slot at
once.

<code-tabs name="convert" />

`result[idx]` selects rows across the row-aligned payloads. `len(result)` is
the number of detections, or of points when there are no boxes.
`result.update(...)` returns a copy with the named slots replaced; it accepts
every slot plus `track_id` and `restore_scale`.

## summary and to_json

`summary(normalize=False, decimals=5, embeddings=False)` returns a list of
plain dicts, one row per detection, segment, point or region depending on
which slots are set. `to_json(**kwargs)` passes its arguments to `summary`
and returns the JSON string.

`plot()` renders a dense normal or edge result in its canonical
visualization; it raises for other result types. Annotated images for the
other tasks come from `predict(save=True)`.
