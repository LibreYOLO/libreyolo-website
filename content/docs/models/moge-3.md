---
title: MoGe-3
families: [moge3]
seo_title: "MoGe-3 in Python: metric depth, point map and camera from one image"
description: "Use MoGe-3 in LibreYOLO for metric depth, a camera-space point map, camera intrinsics and surface normals from one RGB image. ViT-L and ViT-g, MIT."
lead: "MoGe-3 is Microsoft's monocular geometry model. From one RGB image it predicts metric depth in metres, a camera-space point map, the camera intrinsics, a validity mask and surface normals. It is MoGe-2 plus a sparse 3D UNet that refines the predicted depth on a voxel grid. LibreYOLO ships the official ViT-L and ViT-g checkpoints for inference, validation and normal export."
keywords: [MoGe-3, MoGe 3, metric depth estimation, monocular geometry, point map, camera intrinsics, surface normals, DINOv2]
last_verified: "1.8.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-depth.pt")
        depth = model(SAMPLE_IMAGE).depth_map

        print(depth.data.shape)        # (H, W) metres, inf outside the valid mask
        print(depth.point_map.shape)   # (H, W, 3) camera-frame points in metres
        print(depth.intrinsics)        # (3, 3) OpenCV pixel intrinsics
        print(depth.valid_mask.shape)  # (H, W) bool
    - label: Normals
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe3l-normal.pt")
        normal = model(SAMPLE_IMAGE).normal_map
        print(normal.array.shape)      # (H, W, 3) unit vectors, OpenCV frame
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreMoGe3l-depth.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/guggenheim-bilbao.jpg save=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Only the normal graph exports; depth export is blocked.
        model = LibreYOLO("LibreMoGe3l-normal.pt")
        model.export(format="onnx")
        model.export(format="torchscript")
---

## Install

MoGe-3 needs no optional extra. The sparse refiner is plain PyTorch and camera
recovery uses SciPy, which is in the base install.

```bash
pip install libreyolo
```

## Predict

Weights download from the LibreYOLO Hugging Face mirrors on first use.

<code-tabs name="predict" />

With a `-depth` checkpoint, `result.depth_map` holds metric depth in metres
(`encoding="depth"`, `metric=True`). The point map, intrinsics and validity
mask are attached to it as `depth_map.point_map`, `depth_map.intrinsics` and
`depth_map.valid_mask`; pixels outside the mask are `inf`. Points use OpenCV
axes, and the intrinsics put pixel centres at integer coordinates. With a
`-normal` checkpoint, `result.normal_map` holds unit normals in the OpenCV
camera frame.

`imgsz` sets the ViT token budget, `(imgsz // 14) ** 2` tokens laid out to the
image aspect ratio. The default 840 is upstream's highest resolution level
(3600 tokens); 490 is close to its lowest. Inputs keep their native
resolution and every output is interpolated back to it. Inference runs in
FP32 and always applies three refinement steps, as upstream does by default.
One image runs per forward pass; there is no stacked-batch path.

CPU, CUDA and MPS all work without a compiled extension. The refiner's memory
grows with `imgsz`, not with the input resolution.

## Variants

| Weight file | Backbone | Task |
| --- | --- | --- |
| `LibreMoGe3l-depth.pt` | DINOv2 ViT-L/14 | depth, point map, intrinsics, mask |
| `LibreMoGe3l-normal.pt` | DINOv2 ViT-L/14 | normals |
| `LibreMoGe3g-depth.pt` | DINOv2 ViT-g/14 | depth, point map, intrinsics, mask |
| `LibreMoGe3g-normal.pt` | DINOv2 ViT-g/14 | normals |

The `-depth` and `-normal` files of one size hold the same tensors; the
suffix picks the task. Against upstream on an RTX 3090 in FP32, the
single-forward graph matches bit for bit and refined depth matches to
floating-point rounding (p99 relative difference 4.7e-6 for `l`, 3.7e-5 for
`g`).

MoGe-3 is not trainable in LibreYOLO: `train()` raises. Zero-shot `val()`
runs the shared depth and normal validators.

## Export

<export-matrix />

Only the normal graph exports, to ONNX and TorchScript, on a fixed square
canvas. Depth export is blocked: the refiner builds data-dependent voxel sets
and camera recovery is a CPU solve.

<code-tabs name="export" />

## Checkpoints

<checkpoint-table />

## Licensing

<provenance-box>

MoGe-3 is by Microsoft ([microsoft/MoGe](https://github.com/microsoft/MoGe)).
Code and weights are MIT, so commercial use is fine. The DINOv2 backbone keeps
its Apache-2.0 attribution. LibreYOLO mirrors the official
`Ruicheng/moge-3-vitl` and `Ruicheng/moge-3-vitg` checkpoints unchanged as
`LibreYOLO/LibreMoGe3{l,g}-{depth,normal}`, not gated.

</provenance-box>

## Citation

<citation-block />
