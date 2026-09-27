---
title: Triton Inference Server
seo_title: "Serve a LibreYOLO model on NVIDIA Triton"
description: "Serve a LibreYOLO ONNX export through NVIDIA Triton: the model repository layout, the generated config.pbtxt, and predicting against an HTTP model URL."
lead: "Triton Inference Server hosts a model repository and answers inference requests over HTTP. LibreYOLO exports the ONNX graph, generates a config.pbtxt that carries the export metadata as one Triton parameter, and treats a model URL as a loadable model path."
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - model repository
  - remote yolo inference
last_verified: "1.5.0"
meta:
  - label: Call
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Helper
    value: "create_triton_config(onnx_path, config_path, model_name=..., max_batch_size=8)"
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protocol
    value: "HTTP and HTTPS V2 inference only. No gRPC, authentication, shared memory, or model load and unload."
  - label: Timeouts
    value: "Connection and network timeouts default to 30 seconds"
verification: "Read from libreyolo/backends/triton.py, libreyolo/models/__init__.py, docs/triton.md and pyproject.toml on the dev branch. Container commands are the pinned ones from docs/triton.md."
snippets:
  install:
    - label: Install
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Export into the repository layout
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Generate config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Resulting layout
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Start the server
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Wait for readiness
      language: bash
      code: |
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do sleep 1; done
    - label: Stop it
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Predict against the served model
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Compare with the local model
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: Pin a version, or change the timeout
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # A second path segment selects the model version. Without it,
        # Triton's configured version policy chooses.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Connection and network timeouts default to 30 seconds.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
---

## Install

<code-tabs name="install" />

The `triton` extra installs `tritonclient[http]`. gRPC and shared-memory extras
are excluded on purpose: this integration is HTTP and HTTPS V2 inference only.
`onnx` is needed because the served artifact and the config generator both work
from an ONNX graph.

## Build the model repository

Export with a dynamic batch axis, into the directory layout Triton expects.

<code-tabs name="repo" />

Triton does not preserve ONNX custom metadata in its model-config response, so the
complete exported metadata has to travel some other way. `create_triton_config`
encodes it as one JSON string parameter named `libreyolo_metadata` in
`config.pbtxt`, emits the input and output declarations in graph order, handles the
JSON escaping, and pins the model to `KIND_CPU`.

The helper validates before writing. It requires exactly one ONNX graph input, at
least one output, resolvable tensor shapes, and metadata whose `names` map defines
every class index from 0 to `nc - 1`. A model that fails any of those checks is
rejected at config time rather than at the first request.

`max_batch_size: 8` matches a dynamic export and lets the server batch up to eight
images per request. For a fixed batch-1 ONNX graph use `max_batch_size=0`;
LibreYOLO then sends images sequentially.

## Start the server

<code-tabs name="serve" />

The commands pin Triton Server 26.04 and deliberately omit Docker GPU flags, since
`KIND_CPU` in the generated config prevents GPU placement anyway.

## Run the artifact

A Triton model URL is a model path. `LibreYOLO()` checks for an `http` or `https`
scheme before any local path handling and returns a backend that speaks to the
server, so the call site is identical to a local checkpoint and so is the `Results`
object that comes back.

<code-tabs name="run" />

The URL form is `http(s)://host:port/model` with an optional version segment. The
port must be explicit. Embedded credentials, a query string and a fragment are all
rejected, as is a path with more than two segments.

`device` is accepted and ignored with a log line, because placement is the server's
decision.

## Constraints

The backend fails with a direct error rather than a degraded result when the
contract is not met: missing LibreYOLO metadata in the model config, more than one
model input, a mismatch between the configured outputs and the model metadata, an
input datatype it does not support, or a server or model that is not ready.

Outside the contract in this version: gRPC, authentication, shared memory, and
loading or unloading models through the API.

Any format Triton itself supports can be served, but the metadata parameter and
the generated config are ONNX-shaped here, so the LibreYOLO path is
[ONNX](/docs/export/onnx) into the repository. For a full video pipeline rather
than a request-response server, see [DeepStream](/docs/export/deepstream).
