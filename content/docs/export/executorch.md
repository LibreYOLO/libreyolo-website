---
title: ExecuTorch
seo_title: "Export to ExecuTorch from LibreYOLO"
description: "Export a LibreYOLO model to an ExecuTorch .pte program with XNNPACK delegation: fixed shape, batch 1, FP32, and the metadata sidecar it needs."
lead: "ExecuTorch runs PyTorch programs on edge targets. LibreYOLO captures the model with torch.export in strict mode, lowers it to XNNPACK, and commits the .pte program together with a JSON metadata sidecar as one unit."
keywords:
  - yolo executorch export
  - .pte program
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - edge pytorch inference
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="executorch")'
    mono: true
  - label: Writes
    value: "One .pte program plus a .pte.json metadata sidecar"
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t.pte")'
    mono: true
  - label: Shapes
    value: "Fixed. dynamic=True and batch != 1 are rejected."
  - label: Precision
    value: "FP32 only. half=True and int8=True are rejected."
  - label: Delegate
    value: "XNNPACK, CPU. delegate='xnnpack' is the only accepted value."
verification: "Read from libreyolo/export/executorch.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/executorch.py and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # Kept out of libreyolo[all] on purpose: ExecuTorch constrains which
        # Torch version it can be paired with.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes weights/LibreYOLO9t.pte and weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Arguments
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, or (height, width)
            batch=1,               # any other value raises ValueError
            dynamic=False,         # True raises ValueError
            delegate="xnnpack",    # the only accepted value
            device="cpu",          # any other device raises ValueError
            output_path=None,      # None writes weights/<stem>.pte
        )
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Bare ExecuTorch runtime
      language: python
      code: |
        import json
        from pathlib import Path

        import torch
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        print(runtime.backend_registry.is_available("XnnpackBackend"))

        program = runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())
        method = program.load_method("forward")

        # Preprocessing and postprocessing are yours on this path.
        outputs = method.execute((torch.zeros(1, 3, 640, 640),))
        print([tensor.shape for tensor in outputs])

        meta = json.load(open("weights/LibreYOLO9t.pte.json"))
        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

This extra is deliberately outside `libreyolo[all]`, because ExecuTorch pins which
Torch version it works with and installing it would drag the whole environment
onto that pair. Install it into an environment you are willing to constrain.

On Windows the lowering step calls the `flatc` executable that ships with
ExecuTorch. If it is not on `PATH` the export raises a `RuntimeError` saying so,
and running from a Visual Studio 2022 Developer PowerShell is the fix.

## Export

<code-tabs name="export" />

Capture is `torch.export.export(..., strict=True)`, which is a real graph capture
with guards rather than a recorded trace. Host scalar reads and data-dependent
control flow are rejected instead of being silently baked in, so several families
fail here that trace successfully elsewhere; the reasons are recorded per
combination in the support matrix.

Lowering runs `to_edge_transform_and_lower` with the XNNPACK partitioner. If the
result contains zero delegate partitions the export raises rather than labeling a
portable-kernel-only program as XNNPACK.

The program and the sidecar are committed together. Both are staged, both are
swapped in, and a failure rolls back to whatever was there before, so a partial
pair never reaches disk.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` dispatches on the `.pte` suffix and returns the same `Results` object
as the checkpoint. The sidecar is mandatory on load: without
`<program>.pte.json` the backend raises `FileNotFoundError`, because the program
carries no class names, task or input size of its own. The backend also checks
that the installed runtime provides `XnnpackBackend` before loading, and reads the
program from bytes rather than mapping the file, which avoids holding a Windows
file lock for the backend's lifetime.

The second snippet is the bare-runtime path. Preprocessing, decoding, NMS and
coordinate rescaling become yours there.

## Constraints

Batch 1, fixed shape, FP32, CPU. `batch != 1` and `dynamic=True` both raise
`ValueError` before the export mutates anything, `half=True` and `int8=True` are
rejected during validation, and a device other than CPU is refused.

`delegate` accepts `"xnnpack"` and nothing else in this version.

Classification exports carry two extra metadata keys, `crop_pct` and
`interpolation`, so the runtime can reproduce the family's resize and center-crop
policy.

The blocked entries name the concrete failure rather than a category. D-FINE
detection and segmentation reach an unsupported `ContextVar` read in deformable
attention under strict capture, and forcing the manual grid-sample path serializes
but then fails at run time on an invalid delegated tensor dimension order. DEIM
and DEIMv2 capture, lower and serialize, then fail during execution. EoMT semantic
segmentation fails on a data-dependent symbolic expression in the mask path.
BiRefNet matting captures at 1024 by 1024 but has no out variant for
`torchvision::deform_conv2d`. SwinIR restoration reloads and then fails in
`aten::alias_copy.out` on mismatched dimension orders.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
