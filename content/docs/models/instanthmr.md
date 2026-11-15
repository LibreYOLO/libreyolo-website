---
title: InstantHMR
families: [instanthmr]
seo_title: "InstantHMR in Python: real-time human body meshes from one image"
description: "Use InstantHMR in LibreYOLO to recover a 3D body mesh for every person in an image. Sizes n, t, s and m, plus an Apache-2.0 commercial m checkpoint."
lead: "InstantHMR, by Naturalpad and LAAS-CNRS, is a real-time human mesh recovery model. For each person box it returns MHR body pose and identity parameters, 70 keypoints in 2D and 3D, the camera translation and an 18,439-vertex mesh. LibreYOLO ports it natively in PyTorch and finds the people with RF-DETR."
keywords: [InstantHMR, human mesh recovery, HMR, body mesh, 3D human pose, MHR, Momentum Human Rig, real-time]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")   # people found with LibreRFDETRm.pt
        result = model("people.jpg")

        meshes = result.meshes            # one row per box in result.boxes
        print(meshes.body_model)          # "mhr"
        print(meshes.vertices.shape)      # (people, 18439, 3) metres, camera frame
        print(meshes.joints2d.shape)      # (people, 70, 2) pixels
    - label: Your own boxes
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh-commercial.pt")
        boxes = [[40, 20, 260, 600]]      # x1, y1, x2, y2 per person
        result = model("people.jpg", person_boxes=boxes)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Fine-tune a pretrained checkpoint on a YOLO-pose COCO-17 dataset.
        model = LibreYOLO("LibreInstantHMRs-mesh.pt")
        model.train(data="my-pose-dataset.yaml", epochs=20)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreInstantHMRm-mesh.pt")
        model.export(format="onnx")   # inputs image, cliff_cond; faces in metadata
---

## Install

InstantHMR needs no optional extra. The default person detector,
`LibreRFDETRm.pt`, needs the RF-DETR extra.

```bash
pip install "libreyolo[rfdetr]"
```

## Predict

Weights download from the LibreYOLO Hugging Face mirrors on first use, after
a licence notice.

<code-tabs name="predict" />

People are found by an RF-DETR detector (`LibreRFDETRm.pt` by default).
Pass `person_boxes=` to use your own boxes, or `person_detector=` to swap the
detector. `result.meshes` rows line up with `result.boxes`. Everything is in
the camera frame of the original image: `transl` is in metres, `vertices` and
`joints3d` are metric and include `transl`, and `joints2d` is in pixels on the
original image. `global_orient`, `body_pose` and `betas` hold the MHR
parameters; MHR rotations are Euler angles in radians. See the
[body mesh](/docs/tasks/body-mesh) task page for the full payload.

## Variants

Four sizes, `n`, `t`, `s` and `m`, all at a 288 px person crop, plus
`LibreInstantHMRm-mesh-commercial.pt`, an `m` trained only on commercially
usable images. Upstream publishes ONNX only; LibreYOLO rebuilds PyTorch
checkpoints that match the upstream ONNX graphs within 8e-5 for all five
releases.

## Train

`train()` fine-tunes a pretrained checkpoint on a YOLO-pose dataset with
COCO-17 person keypoints, using 2D keypoint, reprojection and frozen-prior
losses. Training from scratch and `resume` are not supported. `val()` raises:
the usual body-mesh benchmarks (3DPW, EMDB) are research-license only, so
none is bundled; the MPJPE, PA-MPJPE and PVE metrics are in
`libreyolo.validation.mesh_metrics` for a dataset you hold.

<code-tabs name="train" />

## Export

<export-matrix />

`export(format="onnx")` writes a graph with `image` and `cliff_cond` inputs
and the mesh faces in its metadata.

<code-tabs name="export" />

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

InstantHMR is by Naturalpad and LAAS-CNRS
([Naturalpad-studio/InstantHMR](https://github.com/Naturalpad-studio/InstantHMR)).
The code is Apache-2.0. The weights licence depends on the checkpoint:

- `n`, `t`, `s` and `m`: CC BY-NC 4.0, **non-commercial only**, because part
  of their training images (SA-1B, AI Challenger) are non-commercial.
- `m-commercial`: Apache-2.0, trained on Harmony4D and commercially licensed
  COCO images only.

Every checkpoint was also trained on the SAM 3D Body dataset annotations,
released by Meta under the SAM License, whose terms pass through with the
weights. The mesh decode was written clean-room, so no SAM-licensed
sam-3d-body code is used. The checkpoints are mirrored as
`LibreYOLO/LibreInstantHMR{n,t,s,m}-mesh` and
`LibreYOLO/LibreInstantHMRm-mesh-commercial`, not gated.

</provenance-box>

## Citation

<citation-block />
