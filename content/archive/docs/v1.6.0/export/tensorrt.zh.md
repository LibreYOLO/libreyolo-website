---
title: TensorRT
seo_title: 从 LibreYOLO 导出到 TensorRT
description: >-
  从一个 LibreYOLO 模型构建 TensorRT engine：ONNX 中间产物、FP16 与 INT8 构建、动态 batch
  的优化配置文件，以及 engine 的可移植性限制。
lead: >-
  TensorRT 会把一张图编译成针对某一块 GPU 调优的 engine。LibreYOLO 先导出一个 ONNX 中间产物，用 TensorRT 的
  ONNX 解析器解析它，构建出 engine，并把模型元数据作为 JSON 附属文件（sidecar）写在它旁边。
keywords:
  - yolo 导出 tensorrt
  - tensorrt engine
  - trt fp16
  - tensorrt int8 校准
  - tensorrt 优化配置文件
  - tensorrt 动态 batch
  - tensorrt 硬件兼容性
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="tensorrt")
    mono: true
  - label: 输出
    value: 一个 .engine 文件，外加一个 .engine.json 元数据附属文件
  - label: 额外依赖
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: 加载方式
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: 形状
    value: 默认固定；dynamic=True 会加上一条沿 batch 轴的优化配置文件
  - label: 精度
    value: FP32、FP16（half=True）、INT8（int8=True 并配合 data=）
  - label: 环境要求
    value: 构建时和运行时都需要一块 NVIDIA GPU。engine 不能在不同 GPU 架构之间迁移。
verification: >-
  读自 dev 分支上的
  libreyolo/export/tensorrt.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/tensorrt.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # engine 是从 ONNX 中间产物构建出来的，所以两个 extra 都要装
        pip install "libreyolo[onnx,tensorrt]"
    - label: 构建前确认工具链
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # 输出 weights/LibreYOLO9t_fp16.engine 和
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: 完整参数
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # int8=True 时必填
            dynamic=False,
            workspace=4.0,                  # 构建时的临时显存，单位 GiB
            min_batch=1,                    # 动态优化配置文件的上下界
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # 或 "ampere_plus"
            gpu_device=0,                   # 多卡主机上用于构建的设备
            verbose=False,
        )
  dynamic:
    - label: 动态 batch 的 engine
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ONNX 中间产物必须带上动态 batch 轴，
        # 优化配置文件才有东西可以绑定
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: 带校准数据的 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # 必填：这个格式没有默认值
            fraction=1.0,
        )
  run:
    - label: 通过 LibreYOLO 运行
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接使用 TensorRT
      language: python
      code: |
        import json

        import tensorrt as trt

        path = "weights/LibreYOLO9t_fp16.engine"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # 类别名、任务和输入尺寸都在附属文件里，不在 engine 里
        # 缓冲区分配、预处理和后处理在这条路上都归你自己做
        print(json.load(open(path + ".json"))["names"])
  support:
    - label: 构建前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## 安装

构建和运行都需要一块 NVIDIA GPU，以及一套可用的 CUDA 技术栈。这个格式没有 CPU
回退路径。

<code-tabs name="install" />

`tensorrt` 这个 extra 会锁定 `tensorrt-cu12` 和 `pycuda`，而 marker 会在 macOS 上把
两者都去掉。在 Jetson 上不要用这个 extra：它会把一个 CUDA 12 的构建锁到 CUDA 13 的
平台上。改用 JetPack 装好的那份 TensorRT，具体见
[NVIDIA Jetson](/docs/export/jetson)。

## 导出

<code-tabs name="export" />

导出分两步走。第一步把一个 ONNX 中间产物写到临时路径，第二步解析它并构建 engine，
之后中间产物会被删掉。`workspace` 是构建时的临时显存，单位是 GiB；值越大，构建器
能试的 kernel 越多，它不影响推理时的显存占用。

元数据附属文件以 `<engine>.json` 的名字写在 engine 旁边，记录这次构建实际达成的
精度。当 GPU 缺少快速 FP16 或快速 INT8 时，构建器会告警并回退，附属文件报告的是最终
得到的那个精度，而不是当初要求的那个。

在 FP16 下，图里的 ViT 骨干会被检测出来，它的浮点层会被固定到 FP32。DINOv2 这一类
骨干在 FP16 下会溢出并产生 NaN，所以构建会设置 `OBEY_PRECISION_CONSTRAINTS`，并报告
`FP16 (FP32 ViT backbone)`。这个 pass 在 CNN 骨干上不做任何事。

### 动态 batch

<code-tabs name="dynamic" />

`dynamic=True` 会加上一条优化配置文件（optimization profile），覆盖 `min_batch` 到
`max_batch` 的范围，在 `opt_batch` 处做优化，并把这三个值记进附属文件。只有当 ONNX
中间产物确实带着动态 batch 维度时，这条优化配置文件才会被加上；否则构建会记一条日志，
说明自己用的是静态优化，然后继续。

### INT8

<code-tabs name="int8" />

INT8 走的是 TensorRT 的熵校准器，接在一个 LibreYOLO 的校准数据加载器上，而且 `data`
是必填的：这个格式没有八张图的兜底。校准需要 `cuda-python` 或 `pycuda` 来准备设备端
缓冲区。校准缓存以 ONNX 字节流的哈希为键，所以一个模型的缩放系数绝不会被另一个碰巧
写到同一个输出路径的模型复用。

`half=True` 和 `int8=True` 一起给出时会告警，并按 INT8 构建，这会为 TensorRT 无法量化
的层保留一条 FP16 回退路径。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.engine` 后缀分发，从附属文件里读取类别名、任务和姿态 schema，并返回
和检查点（checkpoint）一样的 `Results` 对象。当机器上没有 CUDA 设备时，它会立即抛错。

第二个片段是裸运行时的路径。主机端和设备端的缓冲区分配、预处理、解码、NMS 和坐标缩放
都归你自己做，而 engine 本身不带类别名，所以附属文件必须跟着它一起走。

## 限制

序列化后的 engine 绑死在构建它的那套 GPU 架构、驱动栈和 TensorRT 版本上。在工作站上
构建的 engine 无法在另一种架构上加载，这也是构建这一步要在部署机器上跑的原因。
`hardware_compatibility="ampere_plus"` 用一部分性能，换来在 Ampere 及更新架构之间的
可移植性。`"same_compute_capability"` 这个取值会映射到 `NONE` 并给出告警：engine 只针对
当前这块 GPU 做了优化，导出会如实说出来，而不是宣称一种它并没有施加的可移植性。

只有 batch 轴会进优化配置文件。带动态空间维度的构建不在这份契约之内，FCOS 因此被拦下：
它需要动态的填充高和宽，才能保住自己 800 乘 1333 的宽高比变换。

在 trace 之前就被拦下的有：YOLO9 分割、RTMDet-Ins 分割，SSD、Faster R-CNN 和 RetinaNet
的检测，以及 BiRefNet 或 FeyNobg 的抠图，在这些路径上，TensorRT 10.16 会走到共享的
ONNX `DeformConv` 节点，而插件注册表（registry）里没有 `ModulatedDeformConv2d`，它解析
不了这个节点。

当某个组合既没有被验证也没有被拦下时，转换路径是通的，只是项目没有记录过它在 TensorRT
运行时上的一致性结果。这是关于证据的陈述，而不是关于构建能不能成功。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
