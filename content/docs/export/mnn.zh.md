---
title: MNN
seo_title: "从 LibreYOLO 导出到 MNN"
description: "把 LibreYOLO 检测器通过 ONNX 和 mnnconvert 导出成 MNN：固定的 NCHW 形状、CPU 上的 FP32，以及运行时契约要求的元数据 sidecar。"
lead: "MNN 是阿里巴巴的轻量推理引擎。LibreYOLO 导出一张静态 ONNX 图，用 MNN 包自带的 mnnconvert 工具做转换，并写出一个 JSON sidecar，记录输入和输出名称、固定的输入形状以及类别名。"
keywords:
  - yolo 导出 mnn
  - mnnconvert
  - mnn 推理
  - 移动端目标检测推理
  - nchw 固定输入形状
last_verified: "1.5.0"
meta:
  - label: 参数
    value: 'export(format="mnn")'
    mono: true
  - label: 输出
    value: "一个 .mnn 文件，外加一个 .mnn.json 元数据 sidecar"
  - label: 额外依赖
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: 重新加载方式
    value: 'LibreYOLO("weights/LibreYOLO9t.mnn")'
    mono: true
  - label: 形状
    value: "固定 NCHW。dynamic=True 会被拒绝。"
  - label: 精度
    value: "仅 FP32，仅 CPU。"
  - label: 任务
    value: "本版本仅支持检测"
verification: "读自 dev 分支上的 libreyolo/export/mnn.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/mnn.py 和 pyproject.toml。"
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        # 这个 extra 包含 libreyolo[onnx]：MNN 从一个 ONNX 中间产物做转换
        pip install "libreyolo[mnn]"
    - label: 确认转换器在 PATH 上
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 输出 weights/LibreYOLO9t.mnn 和 weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: 参数
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int，或 (height, width)
            batch=1,          # 固化进产物
            simplify=True,    # 对 ONNX 中间产物跑 onnxsim
            output_path=None, # None 表示写到 weights/<stem>.mnn
            verbose=False,    # True 会实时输出 mnnconvert 日志
        )

        # dynamic=True 会抛 ValueError，half=True 和 int8=True 会被拒绝
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 直接用 MNN
      language: python
      code: |
        import json

        import MNN
        import numpy as np

        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))
        print(meta["mnn_input_names"], meta["mnn_output_names"], meta["mnn_input_shape"])

        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )
        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )

        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)
        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )
        outputs = module.forward([input_var])
        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # 这条路径上的预处理和后处理由你自己负责
  support:
    - label: 导出前检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
---

## 安装

<code-tabs name="install" />

这个 extra 包含 `libreyolo[onnx]`，因为转换要经过一个 ONNX 中间产物。它还会带上
`mnnconvert` 可执行文件，导出器先在当前 Python 解释器旁边找它，再去 `PATH` 上找。
转换器缺失时会抛出 `ImportError` 并给出安装命令，而不是在转换中途失败。

## 导出

<code-tabs name="export" />

把图交出去之前，导出器会读取 ONNX 的输入契约，并拒绝任何它表达不了的东西：多于一个
图像输入，或者输入形状里带符号维度。这个版本的 MNN 要求完全固定的 NCHW 形状，
`batch` 会固化进产物，而不是在加载时协商。

sidecar 不是可有可无的记账。`weights/LibreYOLO9t.mnn.json` 记录了输入和输出名称、
固定的输入形状、批大小、类别名、所用的 MNN 版本，以及产物构建时面向的后端，运行时
在加载时会逐项校验这些字段。

在 Windows 上，MNN 3.6.1 有时会完成转换，然后在进程退出阶段以访问违例或 fail-fast
状态终止。导出器认得这几个特定的退出码，只要输出文件在，就把这次转换当作成功。

## 运行导出的产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.mnn` 后缀分发，返回和检查点（checkpoint）一样的 `Results` 对象。
加载严格是刻意为之：sidecar 必须声明 `format=mnn`、`mnn_backend=cpu`、
`dynamic=false`、`precision=fp32`、一个尺寸、一个检测任务、一个与记录的图像尺寸一致
的固定正值 NCHW 形状，以及覆盖 0 到 `nc - 1` 每一个索引的类别名。任何不匹配都会直接
抛错，而不是去猜。

用和产物构建时不同的 `imgsz` 去预测同样会抛错，`device` 会被忽略并给出警告，因为这里
的 MNN 导出产物跑在 CPU 上。

第二段代码走的是裸运行时路径。预处理、解码、NMS 和坐标缩放在那里都归你自己做，输入和
输出名称来自 sidecar，因为 MNN 的模块加载器要求显式给出。

## 限制

只支持检测。后端在加载时会拒绝其他任何任务，导出侧也一致：超出记录在案的组合，预检会
抛出「MNN v1 has no implemented runtime contract for this family and task.」

FP32、CPU、固定形状。`dynamic=True` 会抛 `ValueError`，`half=True` 和 `int8=True`
在校验阶段会被拒绝。

已验证的检测家族有 YOLO9、YOLO9-E2E、YOLO9-P2、RF-DETR、EC、RT-DETR、RT-DETRv2、
RT-DETRv4、D-FINE、DEIM 和 YOLO-NAS，每一个都覆盖了转换、重新加载新产物、MNN CPU
执行、元数据检查，以及与 PyTorch 模型在 NMS 后检测结果上的一致性对齐。DEIMv2 能转换、
能重新加载、能执行，也保住了 NMS 后的检测结果，但它的 ONNX 中间路线在 query 级分数上
一致性不完整，所以只记为可用，而不是已验证。

完整的家族与任务矩阵见[导出矩阵](/docs/reference/export-matrix)。要查单个组合：

<code-tabs name="support" />
