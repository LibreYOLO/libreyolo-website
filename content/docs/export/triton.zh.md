---
title: Triton Inference Server
seo_title: "在 NVIDIA Triton 上部署 LibreYOLO 模型"
description: "把 LibreYOLO 的 ONNX 导出产物交给 NVIDIA Triton 部署：模型仓库的目录结构、生成的 config.pbtxt，以及对着一个 HTTP 模型 URL 做预测。"
lead: "Triton Inference Server 托管一个模型仓库，通过 HTTP 响应推理请求。LibreYOLO 导出 ONNX 图，生成一份把导出元数据装进单个 Triton 参数的 config.pbtxt，并把模型 URL 当作一条可加载的模型路径。"
keywords:
  - libreyolo triton
  - triton 推理服务器
  - config.pbtxt 生成
  - tritonclient http
  - triton 模型仓库
  - yolo 远程推理
last_verified: "1.5.0"
meta:
  - label: 调用
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: 辅助函数
    value: "create_triton_config(onnx_path, config_path, model_name=..., max_batch_size=8)"
    mono: true
  - label: 额外依赖
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: 协议
    value: "仅支持 HTTP 与 HTTPS 的 V2 推理。不支持 gRPC、认证、共享内存，也不支持模型的加载与卸载。"
  - label: 超时
    value: "连接超时与网络超时默认为 30 秒"
verification: "依据 dev 分支上的 libreyolo/backends/triton.py、libreyolo/models/__init__.py、docs/triton.md 和 pyproject.toml 校对。容器命令沿用 docs/triton.md 中固定的那一套。"
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: 导出到仓库目录结构
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
    - label: 生成 config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: 生成的目录结构
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: 启动服务
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: 等待就绪
      language: bash
      code: |
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do sleep 1; done
    - label: 停止服务
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: 对部署好的模型跑预测
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 与本地模型对比
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 固定版本，或修改超时
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # 第二段路径用于选择模型版本，不写的话，
        # 由 Triton 配置的版本策略决定
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # 连接超时与网络超时默认为 30 秒
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
---

## 安装

<code-tabs name="install" />

`triton` 这个额外依赖会装上 `tritonclient[http]`。gRPC 和共享内存相关的额外依赖是刻意排除的：这套集成只做 HTTP 与 HTTPS 的 V2 推理。`onnx` 是必需的，因为被部署的产物和配置生成器都基于 ONNX 图工作。

## 构建模型仓库

以动态批次轴导出，写进 Triton 期望的目录结构里。

<code-tabs name="repo" />

Triton 在自己的模型配置响应里不保留 ONNX 自定义元数据，所以完整的导出元数据必须走另一条路传过去。`create_triton_config` 把它编码成 `config.pbtxt` 里一个名为 `libreyolo_metadata` 的 JSON 字符串参数，按图中的顺序写出输入与输出声明，处理好 JSON 转义，并把模型固定为 `KIND_CPU`。

这个辅助函数在写入前先做校验。它要求 ONNX 图恰好有一个输入、至少一个输出、张量形状可解析，并且元数据的 `names` 映射定义了从 0 到 `nc - 1` 的每一个类别索引。任何一项检查不通过的模型，都会在生成配置时就被拒绝，而不是等到第一次请求。

`max_batch_size: 8` 与动态导出相匹配，让服务端每次请求最多批处理八张图像。如果 ONNX 图是固定的 batch 1，就用 `max_batch_size=0`；LibreYOLO 会改成逐张顺序发送图像。

## 启动服务

<code-tabs name="serve" />

这些命令把 Triton Server 固定在 26.04，并刻意省掉了 Docker 的 GPU 标志，因为生成的配置里的 `KIND_CPU` 本来就会阻止分配到 GPU。

## 运行产物

Triton 的模型 URL 就是一条模型路径。`LibreYOLO()` 会在处理任何本地路径之前先检查 `http` 或 `https` 协议头，然后返回一个与服务端通信的后端，所以调用处和用本地检查点（checkpoint）时完全一样，拿回来的 `Results` 对象也一样。

<code-tabs name="run" />

URL 的形式是 `http(s)://host:port/model`，版本段可选。端口必须显式写出。内嵌的凭据、查询字符串和片段都会被拒绝，超过两段的路径也一样。

`device` 会被接受，但只留一行日志然后忽略，因为设备分配是服务端的决定。

## 约束

契约不满足时，后端直接报错，而不是给出一个降级的结果：模型配置里缺少 LibreYOLO 元数据、模型输入多于一个、配置的输出与模型元数据不匹配、输入数据类型不受支持，或者服务端或模型尚未就绪。

本版本契约之外的东西：gRPC、认证、共享内存，以及通过 API 加载或卸载模型。

Triton 自己支持的任何格式都可以部署，但这里的元数据参数和生成的配置都是按 ONNX 的形状来的，所以 LibreYOLO 的路径是把 [ONNX](/docs/export/onnx) 放进仓库。如果你要的是完整的视频流水线，而不是请求-响应式的服务，见 [DeepStream](/docs/export/deepstream)。
