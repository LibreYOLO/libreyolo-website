---
title: NVIDIA DeepStream
seo_title: "Run YOLO models on NVIDIA DeepStream"
description: "Export a LibreYOLO model for NVIDIA DeepStream: an ONNX graph plus a generated nvinfer config. Exact commands for the parser build and the pipeline."
lead: "NVIDIA DeepStream runs inference through its nvinfer element, which needs an ONNX graph, a matching config file and a bounding-box parser. Setting deepstream=True on the ONNX export writes the first two and wires them to the third."
keywords:
  - NVIDIA DeepStream
  - DeepStream YOLO
  - nvinfer
  - custom bounding box parser
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - TensorRT engine
  - Jetson
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Writes
    value: "An ONNX graph, config_infer_primary_<stem>.txt, and <stem>_labels.txt"
  - label: Coverage
    value: "43 family and task combinations across nine tasks"
  - label: Parser
    value: "NvDsInferParseYolo, from the MIT-licensed DeepStream-Yolo project by Marcos Luciano. Built once per device."
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: https://github.com/marcoslucianops/DeepStream-Yolo
  - label: Availability
    value: "The deepstream-export branch, commit 5f81e11e. No published release carries it."
    links:
      - label: branch
        href: https://github.com/LibreYOLO/libreyolo/tree/deepstream-export
      - label: issue 648
        href: https://github.com/LibreYOLO/libreyolo/issues/648
  - label: Runtime validated
    value: "DeepStream 8.0.0 on an RTX 5070 Ti, detection only, 2026-08-08"
verification: "Written from the runtime validation of 2026-08-08 and read against the deepstream-export branch at commit 5f81e11e. Family lists, config keys and defaults come from libreyolo/export/deepstream.py and libreyolo/export/exporter.py on that commit."
snippets:
  install:
    - label: Install from the branch
      language: bash
      code: |
        pip install "libreyolo[onnx] @ git+https://github.com/LibreYOLO/libreyolo@5f81e11e"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, LibreDFINE

        # Writes libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt
        # and libreyolo9s_labels.txt into the working directory.
        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx", deepstream=True)

        # Keep each detection model in its own directory: every detection
        # config names the same engine cache file. See "Known traps".
        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx", deepstream=True)
    - label: Arguments
      language: python
      code: |
        model.export(
            format="onnx",     # deepstream=True is rejected for every other format
            deepstream=True,
            conf=0.25,         # seeds pre-cluster-threshold (and classifier-threshold,
                               # segmentation-threshold on those tasks)
            iou=0.45,          # seeds nms-iou-threshold, omitted at cluster-mode=4
            batch=1,           # seeds batch-size and the engine cache filename
            half=False,        # True marks the config network-mode=2 (fp16 build)
            int8=False,        # True marks the config network-mode=1
            dynamic=True,      # dynamic batch axis in the ONNX graph
            imgsz=640,         # seeds infer-dims=3;H;W
        )

        # deepstream=True and nms=True are mutually exclusive: DeepStream runs
        # suppression in its clustering stage, so nothing is embedded in the graph.
    - label: Fetch D-FINE weights first
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Confirm GPU passthrough before anything else
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: build_parser.sh, run inside the DeepStream container
      language: bash
      code: |
        set -e
        git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo.git

        # /usr/local/cuda-12 on this image is a stub and the build dies on it with
        # "fatal error: crt/host_defines.h: No such file or directory". Resolve a
        # toolkit that actually carries the header; on the 8.0 image that is cuda-12.5.
        CUDA_DIR=$(readlink -f /usr/local/cuda)
        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # The image ships libcublas.so.12 and libcublas.so.12.8.4.1 but not the
        # unversioned libcublas.so that -lcublas needs, so the link step fails with
        # "/usr/bin/ld: cannot find -lcublas". Give the linker the names it wants.
        mkdir -p /tmp/cudalibs
        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done
        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"

        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Instance segmentation uses a different parser
      language: bash
      code: |
        git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: |
        [application]
        enable-perf-measurement=1
        perf-measurement-interval-sec=5
        gie-kitti-output-dir=kitti

        [tiled-display]
        enable=0

        [source0]
        enable=1
        type=3
        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
        num-sources=1
        gpu-id=0

        [streammux]
        gpu-id=0
        batch-size=1
        batched-push-timeout=40000
        width=1920
        height=1080
        live-source=0

        [primary-gie]
        enable=1
        gpu-id=0
        gie-unique-id=1
        config-file=config_infer_primary_libreyolo9s.txt

        [osd]
        enable=1
        border-width=2
        text-size=15

        [sink0]
        enable=1
        type=1
        sync=0

        [tests]
        file-loop=0
    - label: Run it
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Both steps in one container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
---

## Availability

The flag lives on the `deepstream-export` branch of the LibreYOLO repository, at
commit `5f81e11e`. No published release carries it, so a plain
`pip install libreyolo` does not have it. Install from the branch instead.

<code-tabs name="install" />

The branch was rebased onto `dev` and force-pushed on 2026-08-08, so a clone
taken before that date has different history and does not carry the fix that
lets these exports run on a CUDA machine at all. Pin the commit above rather
than tracking the branch name.

## What the export writes

`model.export(format="onnx", deepstream=True)` writes three files side by side.
For `libreyolo9s.pt`:

- `libreyolo9s.onnx`, the detection graph, one output tensor of shape
  `(batch, num_detections, 6)`, each row `[x1, y1, x2, y2, score, class_id]` in
  network-input pixel coordinates.
- `config_infer_primary_libreyolo9s.txt`, an `nvinfer` configuration carrying
  the family's preprocessing constants, class count, thresholds and parser
  wiring.
- `libreyolo9s_labels.txt`, one class name per line.

A labels file appears whenever the checkpoint carries class names. Depth models
have none, so they get neither the file nor a `labelfile-path` key.

LibreYOLO does not emit a `.so`. The `.so` that DeepStream loads is the
bounding-box parser from `marcoslucianops/DeepStream-Yolo`, compiled once per
device, and it is the same binary whichever LibreYOLO detector you point it at.
The model is the ONNX. Classification and semantic segmentation need no parser
at all, because `nvinfer` post-processes those itself.

## Export the model

<code-tabs name="export" />

`LibreDFINE._load_weights` raises `FileNotFoundError` when the file is not
already on disk, without attempting a download, so fetch `LibreDFINEs.pt`
yourself first. That gap is tracked as
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). YOLO9 weights
download on first use.

The flag is Python only. `libreyolo export` on this branch has no `deepstream`
option, and the CLI builds its export arguments from a fixed list rather than
passing unknown keys through.

## Build the bounding-box parser

Detection needs the parser library, instance segmentation needs a different one,
and the remaining tasks need none. Two things on the DeepStream 8.0 image break
the documented build command, and both are environmental rather than LibreYOLO
problems.

The image ships `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` and `cuda-12.9`
under `/usr/local`. Only `cuda-12.5` has a complete toolkit. It also ships
`libcublas.so.12` and `libcublas.so.12.8.4.1` but not the unversioned
`libcublas.so` that `-lcublas` resolves against. The script below works around
both.

<code-tabs name="parser" />

Then point `custom-lib-path` in the generated config at the built
`libnvdsinfer_custom_impl_Yolo.so`. The generated value is the relative path
`nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, which resolves
when `deepstream-app` runs from the `DeepStream-Yolo` checkout and needs editing
otherwise.

## Run the pipeline

Check that the container can see the GPU before spending time on anything else.
This is the check the validation run made first, on a Blackwell card under WSL2.

<code-tabs name="gpu" />

The validation run drove `deepstream-app` with one file source, no display sink,
the on-screen display on, and `gie-kitti-output-dir` set so every frame's
detections landed on disk as KITTI text. A config with those settings:

<code-tabs name="run" />

`nvinfer` builds the TensorRT engine from the ONNX on first run and caches it
next to the model, so the first run pays for the engine build and later ones
load the cache.

## The generated config

Both configs below were written by the exporter for the validation run, not
edited afterwards.

| Key | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

The two configs differ in three places: `maintain-aspect-ratio`, `cluster-mode`,
and whether `nms-iou-threshold` is present at all. D-FINE's config omits that key
entirely, which is what `cluster-mode=4` calls for.

Heads that emit at most one prediction per object get `cluster-mode=4`, so
DeepStream runs no clustering over them; clustering would merge genuinely
distinct detections. That covers `rfdetr`, `dfine`, `deim`, `deimv2`, `ec`,
`rtdetr`, `rtdetrv2`, `rtdetrv4` and `yolo9_e2e`. Grid and anchor heads get
`cluster-mode=2` plus `nms-iou-threshold`.

Detection configs also carry `engine-create-func-name=NvDsInferYoloCudaEngineGet`,
which hands engine building to the parser library. That is what fixes the engine
cache filename, and it is the source of the collision described under known
traps.

## Supported tasks and families

Forty-three family and task combinations export. `deepstream_supported_tasks()`
and `deepstream_supported_families(task)` in `libreyolo/export/deepstream.py`
return the same lists at runtime.

| Task | `network-type` | Parser library | Families |
|---|---|---|---|
| Detection | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Classification | 1 | None needed | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Semantic segmentation | 2 | None needed | pidnet, eomt, dinov2, lingbotvision |
| Instance segmentation | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | None needed | yolo9, yolonas, rfdetr, ec |
| Depth | 100 | None needed | depth_anything, zipdepth |
| Restoration | 100 | None needed | nafnet, realesrgan, swinir |
| Matting | 100 | None needed | birefnet |
| Gaze | 100 | None needed | l2cs |

`network-type=100` means DeepStream has no post-processor for the task. Those
configs set `output-tensor-meta=1`, the graph's native outputs pass through
untouched, and the application decodes them from the tensor metadata. Multi-output
graphs are fine there: every output layer reaches the metadata with the same
output names and dynamic axes as a plain ONNX export.

Instance segmentation rows are the detection row followed by that instance's
mask, flattened at `(netH / 4, netW / 4)`, which is the resolution the seg parser
hardcodes, as probabilities for `segmentation-threshold`.

Classification and gaze run as secondary inference. Set `process-mode=2` and
`operate-on-gie-id` in the generated config to put a classifier behind a
detector. Gaze is a head-only contract, one face crop per input, so it needs a
face detector in front of it.

Three families are absent on purpose. `segformer` is not wired to the shared
semantic export contract and cannot export to ONNX in any format. RTMDet-Ins and
YOLO9 have their instance segmentation export blocked in LibreYOLO itself.
`depth_anything3` has no export implementation.

Two rows in the table have checkpoint gaps behind them. Only the `l` EoMT
semantic checkpoint is published, and DINOv2 classification has no published
checkpoint at all, so that combination needs your own fine-tuned weights.

## Preprocessing differences

`nvinfer` computes `net-scale-factor * (x - offsets)` per channel with a scalar
scale, which cannot express per-channel standard deviation. Families that need
one (`rfdetr`, `ec`, the DINO-backboned `deimv2` sizes, `rtmdet`, `picodet`, and
every classification family) have the normalization baked into the exported
graph, and the generated config feeds the graph the matching raw input space.

The geometry is where LibreYOLO's own Python pipelines and `nvinfer` still
diverge:

- Letterbox families (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) pad with gray natively. `nvinfer` pads black.
- `yolonas` detection natively resizes the longest side to 636 inside its 640
  canvas. `nvinfer`'s `maintain-aspect-ratio` uses the full 640.
- Classification natively resizes the shortest side then center-crops. `nvinfer`
  stretches the frame or object ROI to the network input, so tightly cropped
  subjects differ.
- EoMT natively runs sliding-window tiles for semantic segmentation. The exported
  graph is a single stretched canvas, which is faster and less accurate.
- `pidnet` emits a class map at 1/8 of the input resolution and `lingbotvision`
  at 1/16. DeepStream upsamples the class map for display.

The ONNX parity gate feeds already-preprocessed tensors, so it checks graph
outputs and cannot catch a wrong color order or padding policy in the config.
Validate on your own data before deploying an exact-parity workload.

## Known traps

### Two detection models in one directory load each other's engine

Every detection config carries the same line:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

The parser's engine builder requires that basename and it does not vary by
model. Export a second detection model into the same directory and the second
run loads the first model's cached engine. Nothing crashes; the boxes are just
wrong. Give each detection model its own directory. The validation run had to
isolate D-FINE into one before it could be tested at all.

### A box can only carry one class

`nvinfer`'s row format is `[x1, y1, x2, y2, score, class_id]`, one class per
box, so the export collapses class scores to their argmax. A box that `predict`
reports under two classes survives under one. Measured case: LibreYOLO reports
`vase 0.773` and `bottle 0.383` on the same box, and the DeepStream graph keeps
`vase`. This follows from the parser's row format and cannot be changed without
leaving that contract, so it is expected behavior rather than a regression.

## Validated

`deepstream-app` ran to EOS with `App run successful` on both detector head
types, over NVIDIA's bundled `sample_1080p_h264.mp4` (1443 frames), with
per-frame KITTI dumps enabled.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Head type | grid | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frames with detections | 1443 | 1443 |
| Total detections | 18031 | 71105 |

Class histograms over all 1443 frames put cars first and people second for both
models, which is right for a street scene. The four-fold gap in detection count
is the `cluster-mode` difference doing its job: D-FINE at `cluster-mode=4` runs
no clustering, so every query above threshold survives, near-duplicates
included.

Two independently trained models put the dominant object in the same place:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

That run establishes five things: TensorRT builds an engine from the exported
ONNX on sm_120, `nvinfer` accepts every key in the generated config,
`NvDsInferParseYolo` reads the tensor layout correctly, boxes land in
source-resolution 1920x1080 coordinates, and labels resolve against the
generated labels file.

The environment it ran in:

| Component | Value |
|---|---|
| Host OS | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Container runtime | Docker Desktop 29.4.3, WSL2 backend |
| DeepStream image | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| DeepStream version | 8.0.0 |
| Container CUDA | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` at HEAD |

Alongside the pipeline run, `tests/unit/test_deepstream_export.py` covers the
graph adapters and the generated config keys, and its 35 tests pass on this
commit.

## Not validated

Stated so the scope above is not read wider than it is.

- Jetson and aarch64. The export contract does not depend on the architecture,
  but the pipeline has only been run on an x86 discrete GPU.
- Forty-one of the 43 combinations. Only detection with `yolo9` and detection
  with `dfine` went through DeepStream. Classification, semantic segmentation,
  instance segmentation and the raw-tensor tasks are covered by unit tests and
  ONNX parity checks, not by a pipeline run.
- FP16 and INT8. Only `network-mode=0` was exercised.
- Multi-stream and batching. One source, `batch-size=1`.
- Accuracy against a ground-truth dataset. Detections were checked for semantic
  plausibility and cross-model agreement, not scored as mAP through DeepStream.
