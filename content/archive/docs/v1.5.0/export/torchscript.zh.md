---
title: TorchScript
seo_title: 从 LibreYOLO 导出到 TorchScript
description: >-
  把 LibreYOLO 模型导出为 TorchScript：一个 trace 出来的 .torchscript 归档，LibreYOLO
  元数据就放在里面，可以从 Python 或 libtorch 加载。
lead: >-
  TorchScript 是 PyTorch 自己的序列化图格式。LibreYOLO 用 torch.jit.trace 追踪模型，并把结果和一个
  libreyolo_metadata.json 附加文件（extra file）一起保存，所以归档里带着家族、任务、类别名和输入尺寸。
keywords:
  - yolo 导出 torchscript
  - torch.jit.trace
  - torch.jit.load
  - libtorch 部署
  - torchscript 元数据
  - extra_files
last_verified: 1.5.0
meta:
  - label: 标志
    value: export(format="torchscript")
    mono: true
  - label: 输出
    value: 一个 .torchscript 归档，里面带一个 libreyolo_metadata.json 附加文件
  - label: 额外依赖
    value: 无。TorchScript 随 PyTorch 一起提供。
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: 形状
    value: 固定。图是在单一输入形状下 trace 出来的。
  - label: 精度
    value: FP32、FP16（half=True）。没有 INT8。
verification: >-
  读取自 dev 分支上的
  libreyolo/export/torchscript.py、libreyolo/export/exporter.py、libreyolo/export/support.py
  和 libreyolo/backends/torchscript.py。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写入 weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: 参数
      language: python
      code: |
        model.export(
            format="torchscript",
            imgsz=640,        # int，或 (height, width)
            batch=1,
            half=False,       # FP16 权重和激活值
            device=None,      # 这个格式下 None 会在 CPU 上 trace
            output_path=None, # None 会写入 weights/<stem>.torchscript
        )

        # dynamic 会被接受，但归档始终是一次固定形状的 trace，
        # 而且内嵌的元数据无论如何都记录 dynamic=False
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接使用 PyTorch
      language: python
      code: |
        import json

        import torch

        extra_files = {"libreyolo_metadata.json": ""}
        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )
        module.eval()

        metadata = json.loads(extra_files["libreyolo_metadata.json"])
        print(metadata["model_family"], metadata["task"], metadata["imgsz"])

        # 这条路径上的预处理和后处理由你自己负责
        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## 安装

<code-tabs name="install" />

TorchScript 不需要基础安装之外的任何东西，因为 `torch.jit` 随 PyTorch 一起提供。它是
唯一一个既没有可选依赖、也没有外部转换器的导出目标，所以当更长的工具链失败时，拿它先
做一次检查很有用。

## 导出

<code-tabs name="export" />

除非指定了设备，否则 trace 在 CPU 上跑；省略 `output_path` 时，归档会以检查点
（checkpoint）的文件名主干写进 `weights/`。

`torch.jit.trace` 平时会做的重复 trace 检查被关掉了。有几个导出包装器会在第一次前向
时缓存与形状相关的锚框，所以第二次 trace 会走到不同的 Python 分支，尽管录下来的固定
形状图是正确的。一致性测试改为直接验证保存下来的 module。

元数据不放在 sidecar 里。`torch.jit.save` 把 `libreyolo_metadata.json` 存进归档内部，
`torch.jit.load` 再通过 `_extra_files` 把它交回来。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.torchscript` 后缀分派，返回和它来源的检查点一样的 `Results` 对象。
用 `device="auto"` 时，module 会在 CUDA 可用时映射到 CUDA，然后是 MPS，最后是 CPU。

第二个代码片段是给没装 LibreYOLO 的读者准备的路径，也用于通过 libtorch 做 C++ 部署
——同一个归档在那边用 `torch::jit::load` 就能加载。预处理、解码、NMS 和坐标缩放在那里
都由你自己负责。元数据附加文件仍然可读，而且类别名只存在于那里。

## 约束

图是在单一输入形状下 trace 出来的。`dynamic=True` 为了接口对称会被接受，但什么都不
改变，内嵌的元数据报告的是 `dynamic=False`，这样后端就不会去假设一个它用不了的轴。
想要第二种分辨率，就再导出一个归档。

`half=True` 会把模型和 trace 用的输入转成 FP16。没有 INT8 路径：`int8=True` 会在校验
阶段抛出 `NotImplementedError`。

矩形 `imgsz` 对 YOLO9 各家族、HRNet、NAFNet 和 Real-ESRGAN 有效，对有固定正方形契约的
家族则会被拒绝。

有五个组合在 trace 之前就被拒绝。YOLO9 分割，因为 YOLO9 在 LibreYOLO 里只做检测。
RTMDet-Ins 分割，它的动态卷积核掩码解码没有导出运行时的契约。SSD、Faster R-CNN 和
RetinaNet 检测，它们的变长图或动态锚框图只有通过 ONNX Runtime 契约才有一致性证据。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
