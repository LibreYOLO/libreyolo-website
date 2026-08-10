---
title: Hailo
seo_title: "在 Hailo 加速器上运行 LibreYOLO 模型"
description: "把 LibreYOLO 模型部署到 Hailo-8 或 Hailo-8L：静态的 ONNX 导出、需要你自己跑的 Dataflow Compiler 环节，以及哪些架构能编译。"
lead: "Hailo 加速器要用 Hailo Dataflow Compiler 编译，那是一套通过 Hailo Developer Zone 分发的专有 SDK。LibreYOLO 在这条流程里负责的部分只是一次普通的静态 ONNX 导出；解析、量化以及编译成 HEF，都在之后的 DFC 里完成。"
keywords:
  - libreyolo hailo
  - hailo-8 部署
  - hailo-8l
  - 树莓派 ai kit
  - ai hat+ 目标检测
  - hailo dataflow compiler
  - hef 编译
  - hailortcli
last_verified: "1.5.0"
meta:
  - label: LibreYOLO 环节
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: 不是一种格式
    value: '没有 format="hef" 这种格式。DFC 不可能作为 pip 依赖存在。'
  - label: 额外依赖
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: 编译主机
    value: "Linux x86_64，包括 WSL2 Ubuntu 22.04。编译无法在 ARM 上运行。"
  - label: 能编译
    value: "纯 CNN、固定形状的计算图。注意力、动态形状以及以 LayerNorm 为主的设计则不行。"
  - label: 状态
    value: "目前还没有任何 LibreYOLO 家族完整跑通 DFC 并得到可运行的 HEF。"
verification: "读自 dev 分支上的 skills/libreyolo-export-hailo/SKILL.md、libreyolo/export/onnx.py 和 libreyolo/cli/commands/export.py。DFC 的各项约束就是那个 skill 里记录的内容；目前还没有编译并实测过任何 LibreYOLO 的 HEF。"
snippets:
  install:
    - label: LibreYOLO 侧
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Hailo 侧，由你自己安装
      language: text
      code: |
        Prerequisites, none of them installable from PyPI:

        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo 需要 batch 1、固定分辨率，并且不能有动态轴
        # Python API 默认 dynamic=True，所以要显式关掉
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLI 默认就是静态形状
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: 编译前确认计算图是静态的
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 解析、量化并编译
      language: python
      code: |
        from pathlib import Path

        import numpy as np
        from hailo_sdk_client import ClientRunner
        from PIL import Image

        ONNX = "weights/LibreYOLOXs.onnx"
        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h
        IMGSZ = 640

        runner = ClientRunner(hw_arch=HW_ARCH)

        # 对 YOLOX，先不带 end_node_names 转换一次：DFC 的日志会打印
        # 它建议的末端节点，再用这些节点重跑一遍
        runner.translate_onnx_model(ONNX)

        # 归一化必须与 LibreYOLO 的预处理一致：YOLOX 和 YOLO9
        # 不需要均值和标准差，只需要 0-255 到 0-1 的缩放
        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"

        # 可选：让 Hailo 接管 NMS，这份配置同时与类别数和输入尺寸绑定，
        # 所以 COCO-80 的配置用在微调过的三类模型上就是错的，
        # 不加这一行的话，HEF 会输出原始的 head 张量，由应用自己解码
        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox, engine=cpu)\n'

        runner.load_model_script(script)

        # 校准图像必须能代表实际部署时的数据，
        # 随机图像照样能编译，但会悄悄毁掉精度
        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]
        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])

        runner.optimize(calib)
        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: YOLO9 的末端节点
      language: python
      code: |
        # LibreYOLO 的计算图用的是 "/head/..." 前缀，而不是那些为其他导出
        # 写的配置里的 "model.N" 前缀，直接抄来的配置对不上，
        # 如果解析失败，请在自己的图里确认这些名字
        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]
        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: 树莓派 5 搭配 AI Kit 或 AI HAT+
      language: bash
      code: |
        sudo apt install dkms hailo-all
        hailortcli fw-control identify       # 检查设备，同时给出架构名
        hailortcli run libreyoloxs.hef       # 冒烟测试与吞吐量
---

## 安装

LibreYOLO 里没有 `format="hef"`，以后也不会有。Hailo Dataflow Compiler 是一套专有
SDK，以私有 wheel 的形式分发，要先在 Developer Zone 注册才拿得到，所以它不可能成为
依赖或额外依赖。部署分两个阶段：LibreYOLO 写出一个静态 ONNX 文件，然后由你自己对它
跑 DFC。

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## 导出

<code-tabs name="export" />

不要传 `half=True`。DFC 吃的是 FP32 的 ONNX，INT8 量化由它自己做。也不要传
`nms=True`：NMS 要么由 Hailo 通过 `nms_postprocess` 接管，要么由应用自己做，而 NMS
子图在末端节点之后只是累赘。默认的 opset 可用；如果 DFC 的解析器有意见，就用
`opset=11` 重新导出。

DFC 会在你指定的末端节点处切断计算图，也就是检测 head 的卷积，并丢弃下游的一切。
所以 LibreYOLO 平常那份带解码的 ONNX 也是可接受的输入：解码尾部会被解析器直接忽略。

## 编译

<code-tabs name="compile" />

按目标设备选 `hw_arch`：`hailo8` 对应 Hailo-8、26 TOPS 的 AI HAT+ 以及 M.2 和 PCIe
模块；`hailo8l` 对应 Hailo-8L、树莓派 AI Kit 和 13 TOPS 的 AI HAT+；`hailo10h` 对应
Hailo-10H，它需要配套的更新版 DFC 和 Model Zoo。拿不准的时候，在设备上跑
`hailortcli fw-control identify` 就能给出答案。

有两个家族能对上 HailoRT 的 NMS 元架构（meta-architecture），因此 Hailo 可以在编译
好的流水线内部接管抑制：YOLOX 走 `meta_arch=yolox`，YOLO9 走 Hailo 的解耦 head 元
架构，两者的 head 布局完全一样。从 Hailo Model Zoo 取来对应的 `nms_postprocess`
配置，再按你的类别数和输入尺寸调整。其他所有卷积检测器编译出来都是一张没有对应元架
构的图：HEF 输出原始的 head 张量，由应用在 CPU 上跑解码和 NMS。

出问题时把编译日志留着。任何修复都取决于出错的那个层或算子的确切名字。

## 运行产物

<code-tabs name="device" />

应用侧的推理使用 `hailo_platform` 这套 Python API。把 `nms_postprocess` 编译进去
之后，输出是 `(batch, num_classes, max_dets, 5)`，里面是模型坐标系下的
`[y1, x1, y2, x2, score]`，需要你自己缩放回原图。运行时不会用到 LibreYOLO 的
`Results` 流水线；HEF 是一个独立产物，预处理和后处理都归应用负责。

## 约束

一个模型能不能以 Hailo-8 或 Hailo-8L 为目标，取决于它的架构而不是它的名字，所以下面
这条规则同样适用于本页写成之后新增的家族。

只要模型里含有下面任何一项，就编译不了：

- 任何形式的注意力，self、cross、deformable 或 windowed 都算。这就排除了所有 DETR
  系检测器、所有开放词汇或文本条件检测器、所有 ViT 骨干，以及所有语言塔和视觉-语言
  塔。Hailo 自家的 zoo 里确实有几个手工调过的 transformer HEF；那是厂商定制的活儿，
  不能拿来当作任意注意力图都能编译的证据。
- 动态形状，或者依赖数据的控制流。DFC 只编译一个固定的输入形状和一张静态图，所以
  可变的 query 数量、文本提示词、动态 top-k、`NonZero`、带动态索引的 `Gather` 或
  `TopK`，以及 `grid_sample` 全都不行。
- 以 LayerNorm 或 GELU 为主的设计。BatchNorm 能干净地折叠进卷积；而 LayerNorm 的
  支持很差，GELU 也不是原生激活函数，所以 ConvNeXt 那类堆叠即便名义上是卷积的，
  也并不合适。
- 原生分辨率下的图像到图像任务。修复类模型按完整输入分辨率运行，会超出 Hailo 实际
  可用的 SRAM 预算。

一个家族要成为候选，得是纯卷积、用 BatchNorm 搭配 ReLU 或 SiLU，并且输入尺寸固定。
在这个库里，这意味着 CNN 单阶段检测器，其中 YOLOX 和 YOLO9 是首要目标；其他卷积检测
器，比如 PicoDet、YOLO-NAS 和 RTMDet，解码放在应用侧；CNN 分类器 ResNet、
MobileNetV4-conv 和 EfficientNetV2，其中 ResNet 支持得最好，因为 Hailo 的 Model Zoo
为它提供了配方；以及小型卷积任务 head，比如 FOMO 点检测和基于 ResNet 骨干的 L2CS
视线估计，它们原则上可编译，但没有现成的 Hailo 配方。

关于状态有一点必须说明，这也是本页任何内容都不以「已支持」呈现的原因：还没有任何
LibreYOLO 家族完整跑通 DFC 并得到可运行的 HEF。上面这些规则只是从架构出发预测可编译
性。在真正编译出一个 HEF 并实测之前，解析器行为、量化和精度都仍未得到验证，所以请把
每个候选都当成需要自己单独留下证据的对象：一个由确切的检查点（checkpoint）编译出来的
HEF，记录下 DFC、Model Zoo 和 HailoRT 的版本，有文档记录的校准过程，以及在设备上与
FP32 基线做的精度对比，而不是一个吞吐量数字。

如果模型不符合条件，替代方案是那些已有一致性验证记录的运行时：
[ONNX](/docs/export/onnx)、[TensorRT](/docs/export/tensorrt) 和
[OpenVINO](/docs/export/openvino)。
