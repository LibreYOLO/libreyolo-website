---
title: TFLite
seo_title: 从 LibreYOLO 导出到 TFLite (LiteRT)
description: >-
  通过 onnx2tf 把 LibreYOLO 模型导出为 .tflite FlatBuffer：静态形状、仅 FP32、NHWC
  输入，以及能干净转换的那些家族。
lead: >-
  TFLite 是 LiteRT 在移动端和嵌入式目标上执行的 FlatBuffer 格式。LibreYOLO 会导出一个静态 ONNX 图，用
  onnx2tf 的 flatbuffer-direct 模式转换它，并把模型元数据以 JSON sidecar 的形式写在产物旁边。
keywords:
  - yolo 导出 tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - tflite nhwc 输入
  - 边缘推理
last_verified: 1.5.0
meta:
  - label: 标志
    value: export(format="tflite")
    mono: true
  - label: 输出
    value: 一个 .tflite 文件，外加一个 .tflite.json 元数据 sidecar
  - label: 额外依赖
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: 形状
    value: 仅静态。dynamic=True 会被拒绝。
  - label: 精度
    value: 仅 FP32。half=True 和 int8=True 会被拒绝。
  - label: 环境要求
    value: Python 3.12 或更高版本，因为 onnx2tf 2.4.x 没有发布更早的 wheel
verification: >-
  读自 dev 分支上的
  libreyolo/export/tflite.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/tflite.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # LiteRT 是 Google 现在给 TensorFlow Lite 用的名字，两个 extra 装的
        # 是同一套工具链，产出的 .tflite 也一样
        pip install "libreyolo[tflite]"
    - label: 先确认 Python 版本
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写入 weights/LibreYOLO9t.tflite 和 weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" 也是可接受的别名，解析到同一个导出器
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: 参数
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int，或者 (height, width)
            batch=1,
            simplify=True,    # 对 ONNX 中间产物跑 onnxsim
            output_path=None, # None 输出 weights/<stem>.tflite
            verbose=False,    # True 会实时打印 onnx2tf 日志
        )

        # dynamic=True 会抛出 ValueError：转换器需要静态形状
        # half=True 和 int8=True 在 trace 之前就会被拒绝
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接使用 LiteRT
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC，不是 NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # 类别名、任务和输入尺寸都在 sidecar 里

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # 预处理、NCHW 到 NHWC 的转置和后处理都归你自己做
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## 安装

<code-tabs name="install" />

这个额外依赖会拉进 `onnx2tf` 做转换，拉进 `ai-edge-litert` 来运行结果，两者都挂在
Python 3.12 的版本标记后面。在更老的解释器上，导出会抛出一个点明版本要求的
`ImportError`，而不是在转换器内部失败。

`libreyolo[litert]` 装的是完全一样的东西。格式字符串 `litert` 是 `tflite` 的别名，
两种写法输出的都是 `.tflite` 文件。

## 导出

<code-tabs name="export" />

家族和任务会在其他任何事情之前先被检查，所以不支持的组合会立刻失败，报出的是当初把
它挡在外面的那个具体的转换器或运行时错误，而不是一句笼统的提示。转换本身是在一个静态
ONNX 中间产物上，以 `flatbuffer_direct` 模式对 `onnx2tf` 的一次子进程调用。

元数据是一个 sidecar。`weights/LibreYOLO9t.tflite.json` 带着家族、任务、类别名、输入
尺寸和姿态 schema；FlatBuffer 本身没有 LibreYOLO 的元数据字段，所以这两个文件要一起走。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.tflite` 后缀分发，返回和检查点（checkpoint）一样的 `Results` 对象。
后端会读取 sidecar，在解释器要求 channels-last 输入时把 NCHW 的 blob 转置成 NHWC，在存
在量化 scale 和 zero point 时套用解释器给出的值，并把输出转置回 LibreYOLO 后处理所期望
的布局。

第二个片段是裸运行时的路径。在那条路上，预处理、布局转置、解码、NMS 和坐标缩放都归你
自己做，而最容易被漏掉的正是布局这个细节：onnx2tf 产出的是 channels-last 输入，所以形
状为 `(1, 3, 640, 640)` 的 blob 绑不上去。

## 限制

只支持静态形状。`dynamic=True` 会在 trace 之前抛出 `ValueError`，导出画布固定为 `imgsz`
最终解析出来的尺寸。

只支持 FP32。`half=True` 和 `int8=True` 都会在验证阶段被拒绝，所以今天还没法从这个导出
器走到量化部署。

这里的覆盖面比图格式要窄，而且是按实测、而不是按家族来定的。已验证的组合包括 YOLO9、
YOLOX 和 YOLO-NAS 的检测，PIDNet 的语义分割，四个 CNN 分类家族，DINOv2 和 SigLIP2 的
嵌入向量，SigLIP2 的分类，TEED 和 DexiNed 的边缘检测，以及 Real-ESRGAN 和 SwinIR 的复
原。SwinIR 还多一条注意事项：只有当源尺寸和导出画布完全一致时一致性才成立，更小的源会
先被填充到画布尺寸再交给 transformer，这可能和原生的可变尺寸推理有出入。

被拦下的条目会点明具体的失败原因，在动手找绕法之前值得先读一遍。举几个例子：RF-DETR
的检测能在它原生的 384 画布上转换，但 LiteRT 分配不了，因为 `STRIDED_SLICE` 收到的输入
超出了它支持的 5 维秩；PicoDet 被拒是因为一个 `RESHAPE` 要把 19,200 个输入元素映射到
9,600 个输出元素；D-FINE 会在 `GatherElements` 的形状处理上把转换器搞崩；RTMDet 能导出
也能重新加载，原始输出一致性完好，但公开检测框掉到 0.911 IoU，坐标漂移 29.9 px。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合，包括被
拦下时背后的原因字符串：

<code-tabs name="support" />
