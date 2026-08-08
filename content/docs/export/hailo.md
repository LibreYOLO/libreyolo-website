---
title: Hailo
seo_title: "Run LibreYOLO models on Hailo accelerators"
description: "Deploy a LibreYOLO model to a Hailo-8 or Hailo-8L: the static ONNX export, the Dataflow Compiler stage you run yourself, and which architectures compile."
lead: "Hailo accelerators are compiled with the Hailo Dataflow Compiler, a proprietary SDK distributed through Hailo's Developer Zone. LibreYOLO's part of the flow is a plain static ONNX export; parsing, quantization and compilation to a HEF happen in the DFC afterwards."
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - hef compile
  - hailortcli
last_verified: "1.5.0"
meta:
  - label: LibreYOLO step
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Not a format
    value: 'There is no format="hef". The DFC cannot be a pip dependency.'
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Compile host
    value: "Linux x86_64, including WSL2 Ubuntu 22.04. Compilation cannot run on ARM."
  - label: Compiles
    value: "Pure-CNN, fixed-shape graphs. Attention, dynamic shapes and LayerNorm-dominated designs do not."
  - label: Status
    value: "No LibreYOLO family has been taken end to end through the DFC to a running HEF yet."
verification: "Read from skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py and libreyolo/cli/commands/export.py on the dev branch. The DFC constraints are the ones recorded in that skill; no LibreYOLO HEF has been compiled and measured."
snippets:
  install:
    - label: LibreYOLO side
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Hailo side, installed by you
      language: text
      code: |
        Prerequisites, none of them installable from PyPI:

        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo needs batch 1, a fixed resolution and no dynamic axes.
        # The Python API defaults to dynamic=True, so turn it off explicitly.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # The CLI already defaults to static shapes.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Confirm the graph is static before compiling
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: Parse, quantize and compile
      language: python
      code: |
        from pathlib import Path

        import numpy as np
        from hailo_sdk_client import ClientRunner
        from PIL import Image

        ONNX = "weights/LibreYOLOXs.onnx"
        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h
        IMGSZ = 640

        runner = ClientRunner(hw_arch=HW_ARCH)

        # For YOLOX, translate once without end_node_names: the DFC log prints
        # the end nodes it suggests. Re-run with those.
        runner.translate_onnx_model(ONNX)

        # Normalization must match LibreYOLO preprocessing. YOLOX and YOLO9
        # need no mean or standard deviation, only the 0-255 to 0-1 scale.
        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

        # Optional: let Hailo own NMS. The configuration is specific to both the
        # class count and the input size, so a COCO-80 config is wrong for a
        # fine-tuned three-class model. Without this line the HEF emits raw head
        # tensors and the application decodes them.
        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox, engine=cpu)\n'

        runner.load_model_script(script)

        # Calibration images must be representative of deployment data.
        # Random images compile and silently destroy accuracy.
        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]
        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])

        runner.optimize(calib)
        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: YOLO9 end nodes
      language: python
      code: |
        # LibreYOLO graphs use a "/head/..." prefix, not the "model.N" prefix
        # seen in configurations written for other exports. A copied config will
        # not match. Confirm the names in your own graph if parsing fails.
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 with the AI Kit or AI HAT+
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # device check, and it names the arch
        hailortcli run libreyoloxs.hef       # smoke test and throughput
---

## Install

There is no `format="hef"` in LibreYOLO and there will not be one. The Hailo
Dataflow Compiler is a proprietary SDK distributed as a private wheel behind
Developer Zone registration, so it cannot be a dependency or an extra. Deployment
is two stages: LibreYOLO writes a static ONNX file, and you run the DFC over it.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Export

<code-tabs name="export" />

Do not pass `half=True`. The DFC ingests FP32 ONNX and does its own INT8
quantization. Do not pass `nms=True` either: Hailo either owns NMS through
`nms_postprocess` or the application does, and an NMS subgraph is dead weight past
the end nodes. The default opset works; if the DFC parser objects, re-export with
`opset=11`.

The DFC cuts the graph at the end nodes you supply, which are the detection-head
convolutions, and discards everything downstream. LibreYOLO's ordinary decoded
ONNX is therefore acceptable input: the decode tail is simply ignored by the
parser.

## Compile

<code-tabs name="compile" />

Pick `hw_arch` for the target: `hailo8` for Hailo-8, the 26 TOPS AI HAT+ and the
M.2 and PCIe modules; `hailo8l` for Hailo-8L, the Raspberry Pi AI Kit and the 13
TOPS AI HAT+; `hailo10h` for Hailo-10H, which needs a matching newer DFC and Model
Zoo. `hailortcli fw-control identify` on the device answers the question when you
are unsure.

Two families map onto a HailoRT NMS meta-architecture, so Hailo can own
suppression inside the compiled pipeline: YOLOX through `meta_arch=yolox`, and
YOLO9 through Hailo's decoupled-head meta-architecture, whose head layout is
identical. Take the matching `nms_postprocess` configuration from the Hailo Model
Zoo and adjust it for your class count and input size. Every other convolutional
detector compiles as a graph with no matching meta-architecture: the HEF emits raw
head tensors and the application runs decode and NMS on the CPU.

Keep the compile log when something fails. Every fix hinges on the exact failing
layer or operator name.

## Run the artifact

<code-tabs name="device" />

Application inference uses the `hailo_platform` Python API. With
`nms_postprocess` compiled in, the output is `(batch, num_classes, max_dets, 5)`
carrying `[y1, x1, y2, x2, score]` in model coordinates, which you scale back to
the source image yourself. LibreYOLO's `Results` pipeline is not involved at run
time; the HEF is a standalone artifact, and preprocessing and postprocessing are
the application's.

## Constraints

Whether a model can target Hailo-8 or Hailo-8L is a property of its architecture,
not its name, so the rule below applies to families added after this page was
written.

A model will not compile if it contains any of these:

- Attention of any kind, self, cross, deformable or windowed. That rules out every
  DETR-style detector, every open-vocabulary or text-conditioned detector, every
  ViT backbone, and every language or vision-language tower. Hailo's own zoo ships
  a few hand-tuned transformer HEFs; that is bespoke vendor work and is not
  evidence that an arbitrary attention graph compiles.
- Dynamic shapes or data-dependent control flow. The DFC compiles one fixed input
  shape and a static graph, so variable query counts, text prompts, dynamic top-k,
  `NonZero`, `Gather` or `TopK` with dynamic indices, and `grid_sample` are all
  out.
- A LayerNorm-dominated or GELU-dominated design. BatchNorm folds into
  convolutions cleanly; LayerNorm support is poor and GELU is not a native
  activation, so a ConvNeXt-style stack is a bad fit even though it is nominally
  convolutional.
- Native-resolution image-to-image work. Restoration models run at full input
  resolution and exceed practical Hailo SRAM budgets.

A family is a candidate when it is convolution only, uses BatchNorm with ReLU or
SiLU, and has a fixed input size. In this library that means the CNN single-stage
detectors, with YOLOX and YOLO9 as the primary targets; other convolutional
detectors such as PicoDet, YOLO-NAS and RTMDet, with application-side decode; the
CNN classifiers ResNet, MobileNetV4-conv and EfficientNetV2, of which ResNet is
best supported because Hailo's Model Zoo ships recipes for it; and small
convolutional task heads such as FOMO point detection and L2CS gaze on a ResNet
backbone, which are compilable in principle but have no Hailo recipe.

One status caveat, which is the reason nothing on this page is presented as
supported: no LibreYOLO family has been taken end to end through the DFC to a
running HEF. The rules above predict compilability from architecture. Parser
behavior, quantization and accuracy remain unproven until a HEF is compiled and
measured, so treat every candidate as requiring its own recorded evidence: a
compiled HEF from the exact checkpoint with DFC, Model Zoo and HailoRT versions
recorded, documented calibration, and an on-device accuracy comparison against the
FP32 baseline rather than a throughput number.

If the model is disqualified, the alternatives are the runtimes with recorded
parity: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) and
[OpenVINO](/docs/export/openvino).
