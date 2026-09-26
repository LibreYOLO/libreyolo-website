---
title: RKNN
seo_title: "Export to RKNN for Rockchip NPUs"
description: "Compile a LibreYOLO detector to a Rockchip .rknn artifact: the vendor SDK you install yourself, the four validated RK3588 variants, and simulator parity."
lead: "RKNN is Rockchip's compiled NPU format. LibreYOLO exports an opset-19 ONNX intermediate, compiles it with the RKNN Toolkit2 SDK, and can compare the compiled graph against ONNX Runtime in Toolkit2's host simulator without a board."
keywords:
  - yolo rknn export
  - rockchip npu
  - rk3588
  - rknn-toolkit2
  - rknn simulator parity
  - orange pi rockchip inference
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Writes
    value: "One .rknn file, a .rknn.metadata.json sidecar, and a .rknn.parity.json report when verify=True"
  - label: Extra
    value: "None on PyPI. rknn-toolkit2 is a vendor SDK you install yourself."
  - label: Loads back
    value: "Not through LibreYOLO. The artifact runs on the board with Rockchip's runtime."
  - label: Shapes
    value: "Fixed square, batch 1, opset 19. All three are enforced."
  - label: Precision
    value: "The vendor floating build. half=True and int8=True are rejected."
  - label: Scope
    value: "Four detection variants on RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s and YOLO-NAS-s"
verification: "Read from libreyolo/export/rknn.py, libreyolo/export/exporter.py, libreyolo/export/support.py and docs/rknn.md on the dev branch. The measured parity numbers come from the validation record dated 2026-08-04 in docs/rknn.md."
snippets:
  install:
    - label: LibreYOLO side
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Vendor SDK, installed by you
      language: bash
      code: |
        # rknn-toolkit2 is a Rockchip SDK under a separate license. LibreYOLO
        # neither bundles nor installs it. x86_64 Linux only; on Windows use
        # WSL2 or a Linux container.
        #
        # Toolkit2 2.3.2 needs setuptools<81 and fails on ONNX 1.19 or newer,
        # whose removal of onnx.mapping its compiler still imports.
        pip install "setuptools==80.9.0" "onnx==1.18.0"

        # Then install the matching rknn-toolkit2 wheel from Rockchip's own
        # wheel repository, and confirm it imports:
        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.rknn and weights/LibreYOLO9t.rknn.metadata.json
        path = model.export(format="rknn", name="rk3588", imgsz=640, verify=True)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Arguments
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # target platform; target= and target_platform= also work
            imgsz=640,         # must match the variant's recorded canvas
            batch=1,           # any other value raises NotImplementedError
            dynamic=False,     # True raises ValueError
            opset=19,          # any other value raises NotImplementedError
            verify=False,      # True runs the PC simulator and gates on parity
        )
  parity:
    - label: Board-free parity against an existing ONNX artifact
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Check one family and task before compiling
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

Compilation needs Rockchip's RKNN Toolkit2, which is distributed as a vendor SDK
under Rockchip's own license and is not a LibreYOLO dependency. There is no
`libreyolo[rknn]` extra, and nothing about this format installs from a single
line.

<code-tabs name="install" />

A board is not needed to compile or to check numerical parity. An RK3588 board is
needed for latency, power and thermal measurements, none of which have been
recorded.

## Export

<code-tabs name="export" />

The request is validated against a list of exact model variants before anything
is compiled, and the canvas is validated too: passing an `imgsz` other than the
one the variant was recorded at raises rather than silently compiling something
untested. LibreYOLO writes an opset-19 ONNX intermediate, compiles it, optionally
simulates it, and removes the intermediate afterwards.

Metadata is a sidecar named `<model>.rknn.metadata.json`, because the RKNN format
has no portable metadata field.

`verify=True` runs Toolkit2's PC simulator inside the same session that compiled
the artifact, compares every output against ONNX Runtime on the same input, and
writes `<model>.rknn.parity.json` with per-output error metrics. The gates are
cosine similarity of at least 0.9999 and normalized RMSE of at most 0.02, applied
to any output that is not already elementwise close; the vendor floating build
lowers internal tensors to half precision, so strict `allclose` does not hold even
when the decoded boxes are stable. A failing run writes
`<model>.rknn.failed.parity.json`, discards the candidate, and leaves any earlier
successful export at that path untouched.

To compare an ONNX artifact you already have, without exporting again:

<code-tabs name="parity" />

Toolkit2's simulator runs the in-memory graph produced by `load_onnx` and `build`.
It cannot reload a target-specific `.rknn` file without a board, which is why
`verify=True` does compilation, export and simulation in one session.

## Run the artifact

There is no RKNN entry in `libreyolo/backends`, so `LibreYOLO()` does not load a
`.rknn` file. The compiled artifact is deployed to the board and executed by
Rockchip's own runtime, and preprocessing, decoding, NMS and coordinate rescaling
are the application's responsibility there.

`<model>.rknn.metadata.json` carries the class names, input size, task and target
platform, which is what an application needs to reproduce LibreYOLO's
postprocessing. Ship it alongside the compiled model.

For a host-side check that does not need the board, keep an ONNX artifact at the
same fixed shape and compare it in the simulator, as above.

## Constraints

Four combinations compile, and they are model variants rather than families:

| Variant | Task | Canvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Everything else is refused before compilation, with the message that RKNN in this
version is limited to the exact simulator-tested detection variants. Compile-only
results for other models exist but are deliberately not presented as support: on
the same measurement run, RF-DETR left two decoder `GridSample` nodes unlowered,
and D-FINE, RT-DETR, RT-DETRv2, RT-DETRv4, DEIM, DEIMv2 and EC compiled and
simulated with decoded outputs that were materially wrong.

Batch 1, static shapes, opset 19. `half=True` is rejected, because RKNN does not
expose LibreYOLO's `half` contract, and `int8=True` is rejected until
representative calibration and task-accuracy results exist.

Other Rockchip targets are rejected: `rk3588` is the only validated platform.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
