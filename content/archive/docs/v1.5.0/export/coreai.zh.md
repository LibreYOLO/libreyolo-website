---
title: Core AI
seo_title: 从 LibreYOLO 导出到 Apple Core AI
description: >-
  把 LibreYOLO 模型导出为 Apple Core AI 的 .aimodel 资产：仅限
  macOS、固定画布、FP32，以及使用方必须遵守的具名输出顺序约定。
lead: >-
  Core AI 是 Apple 的端侧推理技术栈。LibreYOLO 用 torch.export 捕获模型，经 Core AI
  转换器完成下降（lowering），并写出一个 .aimodel 资产，其中带有模型元数据和导出的输出名称。
keywords:
  - libreyolo 导出 core ai
  - aimodel
  - coreai-torch
  - torch.export apple
  - apple 端侧推理
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: 导出参数
    value: export(format="coreai")
    mono: true
  - label: 输出
    value: 一个附带元数据的 .aimodel 资产
  - label: Extra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: 回读方式
    value: 不经过 LibreYOLO。使用方直接调用 Core AI 运行时。
  - label: 形状
    value: 固定画布。dynamic=True 会抛出 NotImplementedError。
  - label: 精度
    value: 仅 FP32。half=True 和 int8=True 会被拒绝。
  - label: 环境要求
    value: macOS。工具链在别处既不能转换也不能运行，而且 coreai-torch 把 torch 锁定在 2.11.x。
verification: >-
  读自 dev 分支上的
  libreyolo/export/coreai.py、libreyolo/export/coreai_compat.py、libreyolo/export/exporter.py、libreyolo/export/support.py
  和 pyproject.toml。
snippets:
  install:
    - label: 安装，在 macOS 上
      language: bash
      code: |
        # 故意排除在所有聚合 extra 之外：coreai-torch 把 torch 锁定在
        # 2.11.x，会把整个环境拖到那个版本
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # 写出 weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: 参数
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int，或 (height, width)；这就是运行画布
            batch=1,
            output_path=None, # None 时写出 weights/<stem>.aimodel
        )

        # dynamic=True 会抛出 NotImplementedError
        # half=True 和 int8=True 在验证阶段会被拒绝
  outputs:
    - label: 接入使用方之前先读一遍输出顺序
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # 资产元数据按图顺序把导出的输出名称记录在 "coreai_output_names" 下，
        # 请用这份列表按名称映射 Core AI 返回的字典，
        # 绝不要按位置与 eager 前向的元组配对
  support:
    - label: 导出前先检查某个家族和任务
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## 安装

这个格式仅限 macOS。`coreai-torch` 这条依赖带着 `sys_platform == 'darwin'` 标记，
工具链在别处既不能转换也不能运行。

<code-tabs name="install" />

这个 extra 位于所有聚合 extra 之外，包括 `libreyolo[all]`，因为 `coreai-torch`
把 torch 锁定在 2.11 系列。把它装进一个你愿意约束到这一对版本的环境里。

## 导出

<code-tabs name="export" />

捕获走的是 `torch.export`，一次带 guards 的真实图捕获，而不是单次录制下来的
trace。这比 Core ML 那条路更严格：宿主端的标量读取和数据依赖的控制流会被拒绝，
而不是被悄悄固化进去，所以这里有少数几个家族被阻塞，并为它们记录了捕获失败。

三个准备步骤运行在一个作用域里，无论导出成功还是失败，这个作用域都会还原调用方
的活动模型。Darknet 衍生的家族会把推理用的 batch normalization 精确折叠进前面的
卷积，因为 Core AI 0.4.1 不保留 Darknet 那种 epsilon 加在平方根之后的公式。网格
和锚框家族会为固定画布冻结自己的锚框。RF-DETR 会通过重新跑模型自己的固化路径，
为请求的画布重新固化位置嵌入（position embedding），因为转换器对
`aten._upsample_bicubic2d_aa` 没有下降实现。

下降过程会把 PyTorch 为 `aten.grid_sampler_2d` 提供的参考分解折进分解表，因为
Core AI 转换器对 DETR 家族用的 deformable attention 采样器没有下降实现。

资产声明的最低系统版本是 v27，这也是工具链唯一提供的取值。它限制的是部署，而不是
转换：借助 wheel 里自带的运行时，转换和 Python 侧的执行在更早的 macOS 上也能跑，
但不同系统版本之间数值会有差异，所以记录的一致性是在 macOS 27 上测的。

## 运行产物

`libreyolo/backends` 里没有 Core AI 的条目，所以 `LibreYOLO()` 不会加载
`.aimodel`。使用方直接调用 Core AI 运行时，预处理、解码、NMS 和坐标缩放都归他们
自己管。支持矩阵里一行「已验证」，说的是导出的图算出的数值和参考实现一致，而不是
`predict` 会去跑它。

使用方唯一无法自己推导出来的，就是输出顺序：

<code-tabs name="outputs" />

Core AI 返回的是一个具名字典，它的键顺序既不匹配 eager 前向的元组顺序，也不匹配
任何能猜到的顺序。导出的名称正是为此被写进资产元数据，名字叫
`coreai_output_names`。按名称映射。

## 限制

固定画布、FP32、批大小按导出时固定。`dynamic=True` 会抛出 `NotImplementedError`，
`half=True` 和 `int8=True` 在验证阶段会被拒绝。

转换这一侧的覆盖面很广。已验证的组合包括 YOLO9 系列家族、YOLOX、YOLO7、四个
Darknet 时代的检测器、YOLO-NAS、PicoDet、RTMDet、RT-DETR、RT-DETRv2、RT-DETRv4、
D-FINE、DEIM、DEIMv2、EC 和 RF-DETR 的检测；四个 CNN 分类家族，外加冻结类别的
CLIP 和 SigLIP2；Depth Anything V2 和 ZipDepth；NAFNet 和 Real-ESRGAN 的图像修复；
PIDNet 和 LingBotVision 的语义分割；以及 FOMO 点检测。每一项都带有自己记录下来的
上下文，`libreyolo formats` 会把它打印出来。

被阻塞的组合，每一项都记录了原因：

| 组合 | 原因 |
|---|---|
| EoMT 语义分割 | 严格捕获失败并报 `GuardOnDataDependentSymNode`：掩码路径里有地方从张量上读了一个值并据此分支 |
| SegFormer 语义分割 | 捕获路径尚未评估，而且不管什么格式，它公布的权重都是非商用的 |
| L2CS 视线估计 | 模型本身只支持 ONNX、TorchScript、ExecuTorch、TensorRT 和 OpenVINO，这是模型侧的决定 |
| Depth Anything 3 深度估计 | 这个家族对所有格式都拒绝导出 |

RF-DETR 有一条注意事项，值得在比较产物之前先读一遍。它的一致性是对着 Core AI
导出器自己准备的那张图记录的，而不是对着 ONNX；在 640 画布下，RF-DETR 的 ONNX
产物和那张准备好的图对不上。Core AI 的重新固化保留了 eager 模型做的抗锯齿缩放，
而 ONNX 那条路把抗锯齿关掉了。所以在非原生画布下，ONNX 对这个家族不是有效的
参考。

Apple 更早的那个格式见 [Core ML](/docs/export/coreml)。完整的家族与任务网格见
[导出矩阵](/docs/reference/export-matrix)。想查单个组合：

<code-tabs name="support" />
