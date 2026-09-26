---
title: ExecuTorch
seo_title: 从 LibreYOLO 导出到 ExecuTorch
description: >-
  把 LibreYOLO 模型导出为带 XNNPACK 委托的 ExecuTorch .pte 程序：固定形状、批大小 1、FP32，以及它需要的元数据
  sidecar。
lead: >-
  ExecuTorch 在边缘设备上运行 PyTorch 程序。LibreYOLO 用 torch.export 的严格模式捕获模型，把它 lowering
  到 XNNPACK，并把 .pte 程序和一个 JSON 元数据 sidecar 作为一个整体一起提交。
keywords:
  - yolo 导出 executorch
  - .pte 程序
  - xnnpack partitioner
  - torch.export strict
  - executorch 运行时
  - 边缘 pytorch 推理
last_verified: 1.5.0
meta:
  - label: 标志
    value: export(format="executorch")
    mono: true
  - label: 输出
    value: 一个 .pte 程序，外加一个 .pte.json 元数据 sidecar
  - label: 额外依赖
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: 形状
    value: 固定。dynamic=True 和 batch != 1 会被拒绝。
  - label: 精度
    value: 仅 FP32。half=True 和 int8=True 会被拒绝。
  - label: 委托
    value: XNNPACK，CPU。delegate='xnnpack' 是唯一接受的值。
verification: >-
  读取自 dev 分支上的
  libreyolo/export/executorch.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/executorch.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # 特意放在 libreyolo[all] 之外：ExecuTorch 会限制它能搭配的
        # Torch 版本
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写入 weights/LibreYOLO9t.pte 和 weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: 参数
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int，或 (height, width)
            batch=1,               # 任何其他值都会抛出 ValueError
            dynamic=False,         # True 会抛出 ValueError
            delegate="xnnpack",    # 唯一接受的值
            device="cpu",          # 任何其他设备都会抛出 ValueError
            output_path=None,      # None 会写入 weights/<stem>.pte
        )
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接使用 ExecuTorch 运行时
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # 这条路径上的预处理和后处理由你自己负责

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## 安装

<code-tabs name="install" />

这个 extra 特意放在 `libreyolo[all]` 之外，因为 ExecuTorch 会锁定它能配合的 Torch
版本，装上它会把整个环境都拖到那个组合上。请把它装进一个你愿意为此受限的环境里。

在 Windows 上，lowering 这一步会调用 ExecuTorch 附带的 `flatc` 可执行文件。如果它
不在 `PATH` 上，导出会抛出一个 `RuntimeError` 说明这件事，解决办法是从 Visual
Studio 2022 Developer PowerShell 里运行。

## 导出

<code-tabs name="export" />

捕获用的是 `torch.export.export(..., strict=True)`，这是带 guards 的真正图捕获，而
不是录下来的一次 trace。宿主端的标量读取和数据依赖的控制流会被直接拒绝，而不是被
悄悄固化进去，所以有几个家族在这里会失败，尽管它们在别的地方能顺利 trace；具体原因
按组合记录在支持矩阵里。

Lowering 用 XNNPACK partitioner 跑 `to_edge_transform_and_lower`。如果结果里一个委托
分区都没有，导出会直接报错，而不是把一个只用可移植 kernel 的程序标成 XNNPACK。

程序和 sidecar 是一起提交的。两者都先暂存，再一起换入，失败就回滚到原来的状态，所以
磁盘上永远不会出现只写了一半的一对文件。

## 运行产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.pte` 后缀分派，返回和检查点（checkpoint）一样的 `Results` 对象。
加载时 sidecar 是必需的：没有 `<program>.pte.json`，后端会抛出 `FileNotFoundError`，
因为程序本身不带类别名、任务和输入尺寸。后端在加载前还会检查已安装的运行时是否提供
`XnnpackBackend`，并且从字节读取程序而不是映射文件，这样就不会在后端的整个生命周期
里一直占着 Windows 的文件锁。

第二个代码片段是直接使用运行时的路径。预处理、解码、NMS 和坐标缩放在那里都由你自己
负责。

## 约束

批大小 1、固定形状、FP32、CPU。`batch != 1` 和 `dynamic=True` 都会在导出改动任何东西
之前抛出 `ValueError`，`half=True` 和 `int8=True` 在校验阶段被拒绝，CPU 以外的设备也
会被拒绝。

`delegate` 在这个版本里只接受 `"xnnpack"`，别的都不行。

分类导出会多带两个元数据键，`crop_pct` 和 `interpolation`，这样运行时就能复现这个家族
的缩放和中心裁剪策略。

被屏蔽的条目写的是具体的失败原因，而不是一个笼统的类别。D-FINE 的检测和分割在严格捕获
下会走到可变形注意力里一处不支持的 `ContextVar` 读取，而强制走手写的 grid-sample 路径
虽然能序列化，却在运行时因为委托张量的维度顺序无效而失败。DEIM 和 DEIMv2 能捕获、
lowering 并序列化，然后在执行时失败。EoMT 语义分割在掩码路径上因为一个数据依赖的符号
表达式而失败。BiRefNet 抠图能在 1024 乘 1024 下捕获，但 `torchvision::deform_conv2d`
没有 out 变体。SwinIR 复原能重新加载，然后在 `aten::alias_copy.out` 上因为维度顺序不
匹配而失败。

完整的家族与任务网格见[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
