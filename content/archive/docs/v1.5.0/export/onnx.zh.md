---
title: ONNX
seo_title: 从 LibreYOLO 导出 ONNX
description: 把 LibreYOLO 模型导出成 ONNX：LibreYOLO 按家族挑选的 opset、动态轴、内嵌 NMS、INT8，以及导出的图怎么加载回来。
lead: >-
  ONNX 是一种可移植的图格式。LibreYOLO 用 torch.onnx.export 对模型做
  tracing，可选地简化图，并把家族、任务、类别名和输入尺寸写进文件自身的元数据，这样任何 LibreYOLO 后端都能重建后处理。
keywords:
  - yolo 导出 onnx
  - onnxruntime 推理
  - torch.onnx.export
  - onnx opset 选择
  - onnx 动态轴
  - onnx 内嵌 nms
  - onnx int8 量化 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: 参数
    value: export(format="onnx")
    mono: true
  - label: 输出
    value: 一个 .onnx 文件，元数据内嵌在图里
  - label: 额外依赖
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 重新加载方式
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: 形状
    value: Python 里默认 batch 轴动态；分任务的例外见下文
  - label: 精度
    value: FP32、FP16（half=True）、INT8（int8=True，YOLO9 检测）
verification: >-
  读自 dev 分支上的
  libreyolo/export/onnx.py、libreyolo/export/exporter.py、libreyolo/export/support.py、libreyolo/backends/onnx.py
  和 libreyolo/cli/commands/export.py。
snippets:
  install:
    - label: 安装
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 输出 weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: 参数
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int，或 (height, width)
            batch=1,
            dynamic=True,     # Python 默认值，CLI 默认为 False
            simplify=True,    # 对图跑一遍 onnxsim
            opset=None,       # None 会选 13，DETR 系家族选 17
            half=False,       # FP16 权重和激活
            int8=False,       # QDQ INT8，仅 YOLO9 检测
            data=None,        # 校准用的 data.yaml，仅 INT8
            device=None,      # tracing 设备，None 用模型自身的设备
            output_path=None, # None 输出 weights/<stem>.onnx
        )
  nms:
    - label: 把 NMS 内嵌进图里
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 仅 YOLO9 检测，batch 为 1，dynamic 会被强制设为 False
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: 带校准数据的 INT8
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # 几百张有代表性的图像
            fraction=1.0,
        )
  run:
    - label: 通过 LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: 裸用 ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # 这条路径上的预处理和后处理都得你自己写

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # 图里带着家族、任务、类别名和输入尺寸

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: 导出前先查一个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## 安装

<code-tabs name="install" />

这个额外依赖会拉取 `onnx`、`onnxsim` 和 `onnxruntime`。只有 `onnx` 就足够写出文件；
`onnxsim` 负责跑简化那一趟，`onnxruntime` 负责运行产物并完成 INT8 校准。

## 导出

<code-tabs name="export" />

不传 `output_path` 时，文件会以检查点（checkpoint）的文件名主干落在 `weights/` 下，
请求了对应精度时会追加 `_fp16` 或 `_int8`。

`dynamic` 在 Python 里默认为 `True`，在 CLI 上默认为 `False`。打开时 batch 轴会变成
符号维，还有几个任务会放开得更多：语义分割会同时放开掩码的高和宽，Real-ESRGAN 复原会
放开空间维，两阶段检测器则让源图的高和宽保持动态，因为它们的 resize 发生在图内部。

省略 `opset` 时会按家族选择。DETR 系家族（`detr`、`deformable_detr`、`dinodetr`、
`dfine`、`deim`、`deimv2`、`ec`、`lwdetr`、`rfdetr`、`rtdetr`、`rtdetrv2`、
`rtdetrv4`）加上 `deit`、`midas` 和 `moge2` 用 opset 17，`aten::scaled_dot_product`
正是在这一版才有对应的下降实现。其余都用 13。抠图（matting）无论如何都会提到 19，因为
BiRefNet 的解码器需要 `DeformConv` 算子，而 ONNX 从 opset 19 起才定义它。

`simplify=True` 会跑 `onnxsim`，这一趟失败就保留原图，所以简化出错是一条警告，而不是
导出失败。在 macOS arm64 上，如果 `onnx` 是 1.22 或更高版本、`onnxsim` 是 0.6.5 或
更低版本，这一趟会被整个跳过，因为这个组合可能让 Python 进程直接中止。

### 内嵌 NMS

<code-tabs name="nms" />

`nms=True` 只适用于 YOLO9 检测，并且要求 batch 为 1；和 `dynamic=True` 一起请求会记
一条警告并把 dynamic 关掉。此时图有两个输出：`output`，形状是 `(batch, max_det, 6)`；
以及 `raw`，也就是未解码的检测器张量，LibreYOLO 自己的后端用它，好让后处理和 PyTorch
路径完全一致。

### DeepStream

`deepstream=True` 是只有 ONNX 才有的选项。它会按 NVIDIA DeepStream 的解析器期望的
布局导出图，并在旁边写两个附属文件 `config_infer_primary_<stem>.txt` 和
`<stem>_labels.txt`，这样产物不用手写配置就能塞进流水线。

它和 `nms=True` 互斥，两个一起要会抛 `ValueError`：DeepStream 在它自己的聚类阶段做
抑制。传给 ONNX 以外的任何格式同样会抛错。支持的家族和任务矩阵以及解析器的编译，见
[DeepStream](/docs/export/deepstream)。

### INT8

<code-tabs name="int8" />

`int8=True` 会跑 ONNX Runtime 的静态量化，写出一张输入输出都是 float32 的 QDQ 图。
只有 `Conv` 和 `Gemm` 节点会被量化。把检测 head 的解码留在 float32 是有意为之：那次
拼接把像素尺度的检测框坐标和 0 到 1 之间的类别分数混在一起，而单个 per-tensor 的激活
缩放因子会被检测框的数量级主导，把每个分数都压到零。

这个参数目前只适用于 YOLO9 检测，其他任何情况都会在预检时抛 `NotImplementedError`。
不传 `data` 会带一条警告回退到 `coco8.yaml`；八张图像算不上有代表性的校准集。已经在
PyTorch 里量化过的模型走的是另一条路，见[量化](/docs/export/quantization)。

## 运行导出的产物

<code-tabs name="run" />

`LibreYOLO()` 按 `.onnx` 后缀路由，返回的 `Results` 对象和 `.pt` 检查点一样，因为
类别名、任务、输入尺寸和姿态 schema 在导出时就写进了图的 `metadata_props`。用
`device="auto"` 时，只要 ONNX Runtime 报告有 `CUDAExecutionProvider`，会话就会用它，
否则回退到 CPU。

第二段代码是给没装 LibreYOLO 的读者的。在那条路径上，预处理、解码、NMS 和坐标缩放全
都得你自己来；元数据块仍然在那里，可以读。

## 限制

输出张量的名字按任务固定，不读元数据的消费方必须对上它们：

| 任务 | 输出名 |
|---|---|
| 检测，网格和锚框 head | `output` |
| 检测，DETR 系 | `pred_logits`、`pred_boxes` |
| 检测，RF-DETR | `dets`、`labels` |
| 分类 | `output` |
| 语义分割 | `semantic_logits` |
| 深度 | `depth` |
| 表面法线 | `normal` |
| 边缘 | `edges` |
| 复原 | `restored` |
| 抠图 | `matte` |
| 视线 | `yaw_logits`、`pitch_logits` |

RF-DETR 也是唯一一个输入张量叫 `input` 而不是 `images` 的家族。

这一版里有几个任务带着固定分辨率的运行时约定。深度、表面法线和边缘会拒绝
`batch != 1` 并强制 `dynamic=False`。抠图会强制用原生的 1024 方形输入，因为
BiRefNet 的 Swin 相对位置表和它们的分辨率绑死了。复原对除 Real-ESRGAN 以外的每个
家族都强制固定画布，Real-ESRGAN 的生成器是全卷积的。

矩形 `imgsz` 对 YOLO9 系家族、HRNet、NAFNet 和 Real-ESRGAN 有效。有固定方形约定的
家族（`clip`、`deformable_detr`、`detr`、`dinodetr`、`dfine`、`deim`、`deimv2`、
`ec`、`lwdetr`、`moge2`、`rtdetr`、`rtdetrv2`、`rtdetrv4`、`rfdetr`、`siglip2`、
`ssd`）会直接拒绝。

有两种组合在 tracing 之前就会被拒绝：YOLO9 分割，因为在 LibreYOLO 里 YOLO9 只做检测；
以及 RTMDet-Ins 分割，它那套动态卷积核的掩码解码没有导出运行时的约定。

完整的家族和任务矩阵见[导出矩阵](/docs/reference/export-matrix)。想查某一个组合，直接
问库：

<code-tabs name="support" />
