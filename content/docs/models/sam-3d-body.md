---
title: SAM 3D Body
families: [sam3dbody]
seo_title: "SAM 3D Body: full-body mesh recovery in LibreYOLO"
description: "Use SAM 3D Body in LibreYOLO for full-body human mesh recovery. Install and predict; Meta's SAM License gates the checkpoints, CUDA required."
lead: "SAM 3D Body is Meta's promptable model for recovering a full-body 3D mesh, including hands and feet, from a single image and person boxes. LibreYOLO wraps the upstream package rather than porting it."
keywords: [SAM 3D Body, human mesh recovery, body mesh, MHR, Momentum Human Rig, 3D pose]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # This family is not registered with the LibreYOLO() factory, so it
        # is constructed directly. model_path=None is what triggers the
        # gated Hugging Face download; a string is instead treated as an
        # existing local checkpoint path and is never fetched automatically.
        # Inference requires a CUDA device; there is no CPU path.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.vertices.shape)    # (N, V, 3), camera frame, meters
        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: With a person detector
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # No named-string shortcut here: pass a constructed LibreYOLO
        # detector, a plain callable, or a PersonDetector instance.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
---

## Install

```bash
pip install libreyolo
```

That gives you LibreYOLO's adapter only. SAM 3D Body itself is not bundled,
because its license is not one LibreYOLO's own code may be derived from: clone
the upstream repository and install its dependencies yourself, then point
LibreYOLO at the clone.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

or set the `SAM_3D_BODY_PATH` environment variable instead of passing
`sam_3d_body_path` on every call. A user who never constructs this family
never triggers the import, and never encounters the SAM License. This family
is not wired into the `LibreYOLO()` factory or the `libreyolo predict` CLI
command; `LibreSAM3DBody` is the only entry point.

## Predict

<code-tabs name="predict" />

The checkpoint download is gated: it requires accepting Meta's license on the
Hugging Face model page and authenticating with `hf auth login` before the
first download succeeds. Inference itself needs a CUDA device unconditionally:
the upstream estimator moves its batch to the GPU without checking, so a
CPU-only machine raises rather than falling back. `result.meshes` is a
`Meshes` payload, row-aligned with `result.boxes` (one row per detected
person): `vertices` and `joints3d` are metric and already include the
estimated camera translation, `joints2d` is in pixels on the original image,
and rotations follow MHR's convention, Euler angles rather than axis-angle.
See [prediction](/docs/predict) for sources, streaming and result handling.

## Variants

Two backbones behind the same MHR body model: `d3` uses a DINOv3 ViT-H/16+
encoder, and `h` uses the original ViT-H encoder.

## Export

<export-matrix />

Body-mesh export is not implemented: LibreYOLO has not yet defined an
exported-graph contract for the mesh task, including how to represent the MHR
parameter layout outside PyTorch.

## Checkpoints

Every published weight file for this family.

<checkpoint-table />

## Licensing

<provenance-box>

The body model the checkpoints drive, MHR (Momentum Human Rig), is a separate
Meta release under Apache-2.0. LibreYOLO fetches its TorchScript asset from
MHR's own public release at runtime and caches it locally; that file is not
mirrored by LibreYOLO and carries its own Apache-2.0 terms, not the SAM
License.

</provenance-box>

## Citation

<citation-block />
