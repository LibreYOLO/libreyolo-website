---
title: Object tracking
seo_title: Object tracking in LibreYOLO
description: >-
  Track objects across video frames in LibreYOLO with ByteTrack, BoT-SORT,
  OC-SORT or Deep OC-SORT, over any detection, segmentation or pose model.
lead: >-
  Tracking assigns a stable identity to each detection across video frames.
  LibreYOLO does not model it as a task with its own weights: it is a predict
  mode, model.track(), that runs a chosen tracker over the per-frame output of a
  detection, segmentation or pose model.
keywords:
  - object tracking python
  - multi object tracking
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - reid tracking
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() is a generator: one Results per processed frame.
        for result in model.track("video.mp4"):
            print(result.track_id)        # (N,) int tensor, aligned with boxes
            print(result.boxes.xyxy)
    - label: Choose a tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (default), "botsort", "ocsort" or "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Save an annotated video
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Without output_path, the file lands in runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Tune a tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # The config type selects the tracker, so tracker= is redundant here.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Or pass the same fields as keyword arguments and let track() build it.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definition

Tracking is not one of LibreYOLO's task keys, and there is no tracking
checkpoint to download. It is a method on the model, `model.track(source)`,
which runs detection on each frame and associates the results across time. The
method is a generator: it yields one `Results` per processed frame, with
`result.track_id` set to an `(N,)` integer tensor aligned with `result.boxes`.
The same IDs are also on `result.boxes.id`.

Only confirmed, currently tracked objects are yielded. A track the association
loses stays alive for a configured number of frames before it is dropped,
`track_buffer` for ByteTrack and BoT-SORT and `max_age` for the two OC-SORT
variants, so an object recovered inside that window keeps its original ID.

Because association happens after detection, the frame's other payloads survive
it: the tracked `Results` is the detection `Results` sliced to the matched rows,
so masks and keypoints come through with the boxes.

## Models

Two independent choices go into a tracking run: the model that produces boxes
each frame, and the tracker that links them.

Any native LibreYOLO model whose task is detection, segmentation or pose exposes
`track()`, so the choice of detector is the ordinary one. See
[the model index](/docs/models) for the full list, or start from
[YOLO9](/docs/models/yolov9), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) or [RTMDet](/docs/models/rtmdet). Tasks whose
results have no box to associate refuse the call rather than returning
meaningless IDs: classification, oriented boxes, points, depth, surface normals,
edges, semantic and panoptic segmentation, restoration, OCR and body mesh all
raise from `track()`.

Two of LibreYOLO's model tiers also decline it. Models loaded through
`LibreSAM` are image segmenters, and models loaded through `LibreOpenVocab`
are per-frame detectors; both raise from `track()` and are used with `predict()`
per frame instead.

Tracking runs on native PyTorch models. An exported artifact loaded through
`LibreYOLO("model.onnx")` returns a runtime backend object, which carries
`predict()` but not `track()`.

Four trackers ship with the library, selected by the `tracker` argument:

`"bytetrack"` is the default. It is motion only, with a Kalman filter and a
three-stage association: high-confidence detections first, then a second pass
that gives low-confidence detections a chance to match an existing track before
they are discarded, then unconfirmed tracks. Configured with `TrackConfig`.

`"botsort"` keeps ByteTrack's three-stage lifecycle but uses a
center-width-height Kalman state and compensates predicted tracks for camera
motion before matching. This is the motion-only variant of BoT-SORT; it runs no
appearance model. Configured with `BoTSortConfig`, which adds `enable_cmc`,
`cmc_method` and `cmc_downscale`.

`"ocsort"` is also motion only, and adds a velocity-direction term to the
association cost, a second association pass against each track's last real
observation, and a smoothing of the Kalman state along a virtual trajectory when
a track is re-found. Configured with `OCSortConfig`.

`"deepocsort"` extends OC-SORT with appearance. Each track keeps a
confidence-weighted moving average of re-identification embeddings, and a cosine
similarity term joins the association cost, so identities survive long
occlusions and crossing targets. It costs one small embedding network forward
per frame, and its OSNet weights download on first use. Configured with
`DeepOCSortConfig`.

## Predict

<code-tabs name="predict" />

`track_conf` sets the threshold for the first association stage:
`track_high_thresh` for ByteTrack and BoT-SORT, `det_thresh` for OC-SORT and
Deep OC-SORT. It is not `predict()`'s `conf`, and for ByteTrack, BoT-SORT and
OC-SORT the detector runs at a lower threshold internally so weak detections
stay available for the recovery pass. Deep OC-SORT runs the detector at
`det_thresh` itself. For ByteTrack and BoT-SORT, `track_conf` must be at or
above `track_low_thresh`, which defaults to 0.1.

Tracker settings arrive in one of two ways. Pass a config instance to
`tracker_config=`, and its type selects the tracker, making `tracker=` redundant.
Or pass the fields as keyword arguments and let `track()` build the config for
the tracker you named; unknown keys warn rather than being applied silently.
Either way, `track_conf` is ignored once the matching key is set explicitly.

The remaining arguments mirror prediction: `iou`, `imgsz`, `classes`, `max_det`,
`vid_stride`, `show`, and `save` with `output_path`. The source is a video file
path. See [prediction](/docs/predict) for result handling.

## Train

Trackers are not trained. Three of the four are pure motion models with no
learned parameters at all, and Deep OC-SORT's appearance network is a published
re-identification checkpoint that downloads on first use. Improving tracking
quality means improving the detector, or tuning the association thresholds
above.


