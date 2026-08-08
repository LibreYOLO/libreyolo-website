---
title: Body mesh
seo_title: "Body mesh recovery in LibreYOLO"
description: "Recover a parametric 3D body mesh per person in LibreYOLO. Predict from person boxes or a detector, and read vertices, joints and camera translation."
lead: "Body mesh recovery turns a single image and a set of person boxes into a parametric 3D body per person: shape and pose parameters, posed vertices, 3D joints, and the camera translation that places them in front of the lens."
keywords: [human mesh recovery python, body mesh, 3d body pose, SAM 3D Body, MHR, parametric body model, libreyolo mesh task]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # This family is not registered with the LibreYOLO() factory, so it
        # is constructed directly. model_path=None triggers the gated
        # Hugging Face download; a string is treated as an existing local
        # checkpoint and is never fetched. Inference requires CUDA.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # the parameterization these tensors use
        print(meshes.vertices.shape)  # (N, V, 3), camera frame, meters
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2), pixels on the source image
    - label: With a person detector
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector accepts a constructed LibreYOLO detector, a plain
        # callable, or a PersonDetector instance. There is no name shortcut.
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
---

## Definition

Body mesh recovery returns a `Meshes` payload per image, row-aligned with
`result.boxes`: row `i` describes the person in box `i`, the same contract the
pose task uses for keypoints.

Everything is expressed in the camera frame of the original image.
`transl` is metric, in meters, with +z pointing away from the camera.
`vertices` and `joints3d` are metric and already include `transl`, so they need
no further composition. `joints2d` is in pixels on the original image canvas,
not on the crop the network saw. `faces` holds the mesh topology once for the
whole image rather than per row, because every person shares it. There is no
world or gravity frame in this version, and no field silently stands in for one.

Parameter layouts differ between body models, so nothing about the shapes is
fixed: `body_model` names the parameterization and the counts are read back from
the tensors. For `"mhr"`, the Momentum Human Rig, rotations are Euler angles in
radians rather than axis-angle, `body_pose` is a flat per-joint parameter vector
rather than one triplet per joint, and `betas` are identity blendshape
coefficients. Skeleton scale, hand pose and facial expression live in `extras`.

The canonical task key is `mesh`. `body-mesh`, `hmr` and `human-mesh-recovery`
normalize to it.

## Models

[SAM 3D Body](/docs/models/sam-3d-body) is the only family serving this task,
and it is a wrapper rather than a port: Meta's `sam-3d-body` package is
published under the SAM License, which LibreYOLO's own code may not derive
from, so none of it is vendored. Two backbones share the same MHR body model,
`d3` on a DINOv3 ViT-H/16+ encoder and `h` on the original ViT-H.

Three requirements apply before a first prediction, and none of them is
optional.

The upstream package is installed by you, not by LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Point the library at the clone with `sam_3d_body_path=` or the
`SAM_3D_BODY_PATH` environment variable. A user who never constructs this
family never triggers the import.

The checkpoint mirror is gated. Accept the license on the Hugging Face model
page and authenticate with `hf auth login`, or the first download fails. The
MHR body model itself is a separate Apache-2.0 release, fetched from its own
public location and cached locally.

Inference needs a CUDA device. The upstream estimator moves its batch to the
GPU without checking, so there is no CPU path to fall back to and
`device="cpu"` raises.

## Predict

<code-tabs name="predict" />

People reach the model in one of two ways. `person_boxes` passes boxes you
already hold, for a single image only: a fixed set of boxes cannot follow people
across video frames, so passing it with a video source raises instead of
silently reusing frame one's boxes. `person_detector` accepts a constructed
LibreYOLO detector, a callable, or a `PersonDetector`, and is the path for
video. `focal_length` supplies a known camera intrinsic; left unset, the model
uses its own estimate, which is what `meshes.focal_length` reports.

This family is not wired into the `LibreYOLO()` factory or the
`libreyolo predict` CLI command. `LibreSAM3DBody` is the only entry point. See
[prediction](/docs/predict) for sources, streaming and result handling.

## Train

No family in this task trains inside LibreYOLO. `LibreSAM3DBody.train()`
raises: train at the upstream project and load the resulting checkpoint here.

## Validate

There is no mesh validator, and `val()` raises. The usual benchmarks are
research-license only, so none is bundled and none can be fetched for you.

The metrics themselves are available as `libreyolo.validation.mesh_metrics`, for
evaluating against a dataset you already hold. It takes predicted and target
joints, optionally predicted and target vertices, and returns a dictionary keyed
exactly like a validator's:

`metrics/mpjpe` is mean per-joint position error after aligning the root joint,
in millimeters. `metrics/pa_mpjpe` is the same error after a Procrustes
alignment, which removes global rotation, translation and scale, so it measures
pose independently of how the body is placed. `metrics/pve` is per-vertex error
over the full mesh surface, and appears only when both vertex arrays are
supplied. Inputs are assumed metric, in meters; `scale_to_mm` converts them to
the millimeters the literature reports.

## Export

Mesh export is not implemented. LibreYOLO has not defined an exported-graph
metadata contract for this task, including how to carry the MHR parameter layout
outside PyTorch, so `export()` raises rather than emitting a graph whose output
could not be interpreted.
