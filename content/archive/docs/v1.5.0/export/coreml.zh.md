---
title: Core ML
seo_title: 从 LibreYOLO 导出到 Core ML
description: >-
  把 LibreYOLO 检测器导出为 Core ML 的 .mlpackage：ImageType 输入契约、FP16、compute units、内嵌
  NMS，以及支持的四个家族。
lead: >-
  Core ML 是 Apple 的端侧模型格式。LibreYOLO 会在一层按家族区分的预处理包装器后面对检测器做 trace，让转换后的图始终接受规范的
  RGB 图像输入，然后写出一个附带模型元数据的 ML Program 格式 .mlpackage。
keywords:
  - yolo 导出 coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml 内嵌 nms
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="coreml")
    mono: true
  - label: 输出
    value: 一个 ML Program 格式的 .mlpackage bundle（一个目录）
  - label: 额外依赖
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: 加载方式
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") on macOS
    mono: true
  - label: 形状
    value: 固定。输入是形状写死的 ct.ImageType。
  - label: 精度
    value: FP32、FP16（half=True）。没有 INT8。
  - label: 家族
    value: 仅检测，支持 yolox、yolo9、rtdetr 和 rfdetr
verification: >-
  读自 dev 分支上的
  libreyolo/export/coreml.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/coreml.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 输出 bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: 完整参数
      language: python
      code: |
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True 会以 FLOAT16 计算精度转换
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None 输出 weights/<stem>.mlpackage
        )

        # dynamic 会被接受，但输入是固定形状的 ct.ImageType，
        # 而且内嵌的元数据无论如何都记录 dynamic=False
  nms:
    - label: 内嵌 Apple 的 NMS 层
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 仅 YOLOX 和 YOLO9 的检测，batch 为 1
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 在 macOS 上通过 LibreYOLO 运行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # 或用 cpu_and_ne 固定到 Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接使用 coremltools
      language: python
      code: |
        import coremltools as ct
        from PIL import Image

        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")
        print(mlmodel.user_defined_metadata["model_family"])
        print(mlmodel.user_defined_metadata["names"])

        # 输入是一张名为 "image" 的图像，尺寸是导出时的固定尺寸
        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))
        out = mlmodel.predict({"image": image})
        print({name: value.shape for name, value in out.items()})

        # 这条路上 letterbox 和后处理都归你自己做
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## 安装

<code-tabs name="install" />

预测需要 macOS。在其他任何平台上，`LibreYOLO()` 都会拒绝 `.mlpackage`，并在消息里
指出当前平台；支持矩阵把这些组合记为可用，理由是运行时一致性需要一台 macOS runner。

## 导出

<code-tabs name="export" />

bundle 会以检查点（checkpoint）的 stem 写入 `weights/`，`half=True` 时追加 `_fp16`。
`.mlpackage` 是一个目录，所以要把整棵目录树一起复制。

每个家族都在一层预处理包装器后面做 trace，因此转换后的图只接受一种规范输入：RGB、
`scale=1/255`、不设 bias，声明为 `ct.ImageType`。包装器吸收了各家族自己的约定，即
YOLOX 是 0 到 255 范围的 BGR，RF-DETR 是 ImageNet 的均值和标准差，YOLO9 和 RT-DETR
则是恒等变换。这就是为什么 Core ML 这一侧喂进去的是一张普通图像，而不是某个家族专用
的张量。

转换的目标是 ML Program，最低部署目标为 iOS 15。`compute_units` 会存在转换后的模型
上，加载产物时还可以再次覆盖。

模型元数据以字符串形式写入 `user_defined_metadata`，后端就是从那里读取家族、任务、
类别名、输入尺寸和姿态 schema 的。

### 内嵌 NMS

<code-tabs name="nms" />

`nms=True` 会把模型包进一条 Core ML 流水线，末尾接上 Apple 的
`NonMaximumSuppression` 层。结果有两个输出：`confidence`，形状是 `N` 乘类别数；
`coordinates`，形状是 `N` 乘 4，为归一化的 `xywh`。

它只适用于 YOLOX 和 YOLO9 的检测，并且要求 batch 为 1。DETR 系的家族会被按名字拒绝，
因为集合预测（set prediction）是在 queries 和类别上取 top-k，没有 IoU 这一步，用不了
那一层。`max_det` 在这里也没有暴露出来；当检测数量上限重要时，改用
[ONNX 内嵌 NMS](/docs/export/onnx)。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 会识别后缀为 `.mlpackage` 的目录，并返回和检查点一样的 `Results` 对象。
`compute_units` 是这个格式下工厂唯一透传的参数，它接受 `all`、`cpu_and_gpu`、
`cpu_and_ne` 和 `cpu_only`。`device` 参数会被忽略，因为 Core ML 走的是 compute units。

第二个片段是裸运行时的路径。在那条路上，letterbox、解码、NMS 和坐标缩放都归你自己
做，类别名则存在 `user_defined_metadata` 里。

## 限制

四个家族，只有检测：`yolox`、`yolo9`、`rtdetr` 和 `rfdetr`。其他的都会在预检阶段被
拒绝，因为正是这层区分家族的预处理包装器才让固定图像输入的契约成立，而不在其中的家族
会以错误的归一化被转换。报错会点名 ONNX 和 TorchScript 作为替代。

输入形状被 `ct.ImageType` 写死，所以 `dynamic=True` 不会改变任何东西，元数据记录的是
`dynamic=False`。要第二个分辨率，就再导出一个 bundle。

`half=True` 会以 FP16 计算精度转换。这个导出器没有 INT8 的路径。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。Apple 更新的端侧格式
见 [Core AI](/docs/export/coreai)。想查单个组合：

<code-tabs name="support" />
