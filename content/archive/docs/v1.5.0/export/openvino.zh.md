---
title: OpenVINO
seo_title: 从 LibreYOLO 导出到 OpenVINO IR
description: >-
  把 LibreYOLO 模型转换成 OpenVINO IR：model.xml 与 model.bin 这一对文件、FP16 权重压缩、NNCF
  INT8，以及 CPU、GPU 或 NPU 推理。
lead: >-
  OpenVINO IR 是英特尔的运行时格式，一个 model.xml 计算图加上一个 model.bin 权重 blob。LibreYOLO 先导出一个
  ONNX 中间产物，用 ov.convert_model 转换，再往同一个目录里写一个 metadata.yaml。
keywords:
  - yolo 导出 openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - nncf int8 量化
  - openvino npu 推理
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="openvino")
    mono: true
  - label: 输出
    value: 一个包含 model.xml、model.bin 和 metadata.yaml 的目录
  - label: Extra
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: 形状
    value: 跟随 ONNX 中间产物：dynamic=True 时为动态 batch
  - label: 精度
    value: FP32、FP16 权重压缩（half=True）、通过 NNCF 的 INT8（int8=True 并配合 data=）
verification: >-
  读自 dev 分支上的
  libreyolo/export/openvino.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/openvino.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # IR 是从 ONNX 中间产物转换来的，所以两个 extra 都需要
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 还需要 NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写出目录 weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: 参数
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True 会在 IR 中保留动态 batch 轴
            half=False,       # True 会存储 FP16 权重
            int8=False,       # True 会运行 NNCF 训练后量化
            data=None,        # int8=True 时必填
            output_path=None, # None 会写到 weights/<stem>_openvino
        )
  int8:
    - label: 带校准数据的 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # 必填：这个格式没有默认值
            fraction=1.0,
        )
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 选择设备
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" 和 "cpu" 映射到 CPU，"gpu" 和 "cuda" 映射到 GPU，
        # 其他值一律转成大写后原样传入，例如 "npu" -> NPU
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: 直接使用 OpenVINO
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # 类别名、任务和输入尺寸都在 IR 旁边的 metadata.yaml 里

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # 这条路径上的预处理和后处理要你自己做
  support:
    - label: 导出前检查单个家族与任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## 安装

<code-tabs name="install" />

转换要经过一个 ONNX 中间产物，所以 `onnx` extra 属于必需项，而不是可选搭配。NNCF
需要单独安装，只有 `int8=True` 时才用得到。

## 导出

<code-tabs name="export" />

产物是一个目录，不是一个文件。`weights/LibreYOLO9t_openvino` 里放着 `model.xml`、
`model.bin` 和 `metadata.yaml`，`half=True` 时会在后缀前插入 `_fp16`。要移动或复制就
整个目录一起；这三个文件是一个产物。

`half=True` 会在保存时设置 `compress_to_fp16`。那是 IR 里的权重压缩，不是改变设备在
运行时选择的推理精度。

### INT8

<code-tabs name="int8" />

`int8=True` 会用 mixed 预设，在 LibreYOLO 的校准加载器上跑 NNCF 训练后量化，而且
`data` 是必填的：这个格式没有八张图的兜底。缺少 NNCF 会抛出一个 `ImportError`，里面
写明了安装命令。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 能识别任何含有 `model.xml` 的目录，返回和检查点（checkpoint）一样的
`Results` 对象，类别名、任务、输入尺寸和姿态 schema 都从 `metadata.yaml` 读取。

设备字符串是经过映射的，而不是直接透传。`auto` 和 `cpu` 都编译到 CPU，`gpu` 和
`cuda` 都编译到 GPU，其他任何值都会转成大写后交给 OpenVINO——NPU 目标就是这样指定的。

第三段代码是给没有安装 LibreYOLO 的读者的。在那条路径上，预处理、解码、NMS 和坐标
缩放都要你自己做，而类别名只存在于 `metadata.yaml` 里。

## 限制

缺少 `metadata.yaml` 的 IR 仍然能加载，但后端会退回到 80 个类别和检测任务，对其他任何
情况来说都是错的。请保持目录完整。

在 tracing 之前就被拦下：YOLO9 分割、RTMDet-Ins 分割，SSD、Faster R-CNN 和 RetinaNet
检测，以及 BiRefNet 或 FeyNobg 抠图——OpenVINO 2026.2 无法转换共享抠图解码器里那个标准
ONNX `DeformConv-19` 算子。

如果一个组合既没有被验证也没有被拦下，那么转换路径是通的，只是项目没有记录它在
OpenVINO 上的运行时一致性。有几个组合是带着明确上下文验证的，例如 DeepLabV3 语义分割
在固定 520×520 输入、OpenVINO 2026.2、CPU 默认推理精度下，以及 L2CS 视线估计在固定
448×448 的人脸裁剪上。`libreyolo formats` 会按组合打印这段上下文。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。要查单个组合：

<code-tabs name="support" />
