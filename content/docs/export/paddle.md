---
title: Paddle
seo_title: "Export to PaddlePaddle from LibreYOLO"
description: "Convert a LibreYOLO detector to a PaddlePaddle inference model through X2Paddle: the pinned toolchain, static batch-1 FP32 graphs, and CPU inference."
lead: "PaddlePaddle inference models are a model.pdmodel graph beside a model.pdiparams weight file. LibreYOLO exports a static opset-15 ONNX graph, converts it with X2Paddle, and packages the result with a metadata.yaml so it loads through the same factory as every other runtime."
keywords:
  - yolo paddle export
  - paddlepaddle inference
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: "1.5.0"
meta:
  - label: Flag
    value: 'export(format="paddle")'
    mono: true
  - label: Writes
    value: "A directory with model.pdmodel, model.pdiparams and metadata.yaml"
  - label: Extra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Loads back
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Shapes
    value: "Static, batch 1, opset 15. All three are enforced."
  - label: Precision
    value: "FP32 only, CPU only."
  - label: Toolchain
    value: "PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 or earlier, checked exactly"
verification: "Read from libreyolo/export/paddle.py, libreyolo/export/exporter.py, libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md and pyproject.toml on the dev branch."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        # Python 3.10 to 3.12. WSL2 with Ubuntu 22.04 is the validated Windows path.
        pip install "libreyolo[paddle]"
    - label: Confirm the pinned versions
      language: bash
      code: |
        python -c "from importlib.metadata import version; print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Writes the directory weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Arguments
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; this family's square canvas
            batch=1,          # any other value raises ValueError
            dynamic=False,    # True raises ValueError
            simplify=True,    # False raises ValueError
            opset=15,         # any other value raises ValueError
            output_path=None, # None writes weights/<stem>_paddle
        )
  run:
    - label: Through LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Bare Paddle
      language: python
      code: |
        import numpy as np
        import paddle.inference as paddle_infer
        import yaml

        directory = "weights/LibreYOLO9t_paddle"
        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(config)
        handle = predictor.get_input_handle(predictor.get_input_names()[0])
        handle.reshape([1, 3, 640, 640])
        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))
        predictor.run()
        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))
        print(meta["model_family"], meta["task"], meta["names"])

        # Preprocessing and postprocessing are yours on this path.
  support:
    - label: Check one family and task before exporting
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## Install

<code-tabs name="install" />

The extra pins the exact stack the parity work measured: PaddlePaddle 2.6.2,
X2Paddle 1.6.0 and ONNX 1.17 or earlier. Those pins are checked at export time,
not just at install time, and a different version raises an `ImportError` naming
the expected one. Newer Paddle releases reject parts of the static code X2Paddle
1.6.0 generates, so failing early is better than producing an artifact nobody has
validated.

## Export

<code-tabs name="export" />

Four arguments are fixed rather than defaulted. `dynamic` must be `False`, `batch`
must be 1, `simplify` must be `True` for a fully static conversion graph, and
`opset` must be 15, which is the ceiling X2Paddle 1.6.0 accepts. Passing anything
else raises before tracing.

One normalization runs on the intermediate graph. ONNX defines an omitted MaxPool
dilation as one, PyTorch writes the explicit all-ones attribute, and X2Paddle
1.6.0 rejects it, so the exporter removes that redundant default and leaves the
specified operation unchanged.

The artifact is a directory: `model.pdmodel`, `model.pdiparams` and
`metadata.yaml`. The Python that X2Paddle generates during conversion is not part
of it.

## Run the artifact

<code-tabs name="run" />

`LibreYOLO()` recognizes any directory holding both `model.pdmodel` and
`model.pdiparams`, reads `metadata.yaml`, and returns the same `Results` object as
the checkpoint. A device other than `auto` or `cpu` raises: this backend is CPU
only.

The bare-runtime snippet mirrors what the backend configures, and the three
disabled options are deliberate. The Paddle 2.6 CPU fusion pipeline can crash
while optimizing the large gather and scatter graphs emitted for deformable
attention, so the portable unfused static graph is the one parity was measured
against. Preprocessing, decoding, NMS and coordinate rescaling become yours on
that path.

## Constraints

No dynamic shapes, no FP16, no INT8, no embedded NMS, no GPU runtime.

Validated combinations are YOLO9 detection, YOLO9-E2E and YOLO9-P2 detection, EC
detection, pose and segmentation, RT-DETRv4, D-FINE, DEIM and DEIMv2 detection,
and YOLO-NAS detection and pose. Each is covered by conversion, a CPU runtime
reload, raw-output parity and matched public results.

Blocked, with the reason recorded per combination:

| Combination | Why |
|---|---|
| RF-DETR, all tasks | Needs ONNX opset 17 and GridSample; X2Paddle 1.6.0 accepts opset 15 or lower and has no GridSample mapper |
| RT-DETR and RT-DETRv2 detection | The trained graphs need GridSample at opset 16 or newer |
| D-FINE segmentation | Converts and reloads, but mask-logit relative RMS error is 3.52% and minimum matched-mask IoU is 0.582 |
| YOLO9 segmentation | YOLO9 is detection only in LibreYOLO |
| RTMDet-Ins segmentation | The dynamic-kernel mask decode has no exported-runtime contract |

Anything not listed as validated or blocked is refused with the note that it has
not been validated through the ONNX-to-Paddle conversion path.

For the full family and task grid, see
[the export matrix](/docs/reference/export-matrix). For one combination:

<code-tabs name="support" />
