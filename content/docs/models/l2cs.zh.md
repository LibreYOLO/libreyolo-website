---
title: L2CS-Net
families:
  - l2cs
seo_title: L2CS-Net：在 LibreYOLO 里做视线估计
description: 在 LibreYOLO 里用 L2CS-Net 做两阶段的视线俯仰角/偏航角估计。安装、预测和导出；Gaze360 检查点仅限研究使用。
lead: >-
  L2CS-Net 是一个两阶段视线估计器：先由人脸检测器定位人脸，再由一个带两个角度分箱分类 head 的 ResNet
  主干为每张人脸预测俯仰角和偏航角。LibreYOLO 对它的封装仅支持推理。
keywords:
  - L2CS-Net
  - 视线估计 python
  - 眼动追踪 开源模型
  - 俯仰角 偏航角 估计
  - Gaze360
  - 人脸检测
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 不给 face_detector 时，会回退到 OpenCV 自带的人脸检测器
        # （OpenCV 4 上是 Haar，OpenCV 5 上是 YuNet），所以除了 L2CS 检查点
        # 本身之外不需要任何额外下载
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 人脸来源
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # 把你已经跑过的检测器给出的检测框交给 L2CS
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # 或者指定某个自带的人脸检测器
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # 导出的计算图只有 ResNet 主干和那两个角度分箱 head：它接收一份预处理好
        # 的 448x448 人脸裁剪图，返回的是原始的 (yaw_logits, pitch_logits)，
        # 而不是解码后的角度；softmax、分箱期望和角度换算仍留在 Python 里，见
        # libreyolo.models.l2cs.utils.bin_logits_to_angles
        session = ort.InferenceSession("LibreL2CSr50.onnx")
        name = session.get_inputs()[0].name
        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## 安装

只要你手上已经有检查点，构造 L2CS-Net 模型、用它预测或者导出它，都不需要任何
可选 extra。

```bash
pip install libreyolo
```

LibreYOLO 唯一能自动获取的检查点是一个在 Gaze360 上训练的 ResNet-50，它走
`gdown` 下载，而不是普通的 HTTP 镜像，因为它放在作者本人的 Google Drive 上，而
不是 LibreYOLO 组织下。这条路径需要 `gaze` extra：

```bash
pip install "libreyolo[gaze]"
```

没有它的话，LibreYOLO 会打印手动下载说明，而不是悄无声息地失败。

## 预测

<code-tabs name="predict" />

L2CS-Net 是一个两阶段估计器：先跑人脸检测器，再由视线 head 从它返回的每一张人脸
裁剪图里读出俯仰角和偏航角。不做设置时，预测会回退到 OpenCV 自带的检测器，所以
只要 L2CS 检查点本身到手，一次裸调用就能跑起来，不需要额外下载。`face_boxes` 接
受你已经跑过的检测器给出的检测框；`face_detector` 接受 `"auto"`、`"haar"`、
`"yunet"`、一个 LibreYOLO 检测模型，或者一个普通的可调用对象。`result.gaze` 以弧
度给出俯仰角和偏航角，逐行与 `result.boxes`（也就是检测到的人脸框）对齐。数据
源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

五种骨干深度共用一个输入分辨率，接受的参数也相同。唯一一份已发布检查点背后的数
据集 Gaze360 训练的是 ResNet-50；另外四种深度在架构上是支持的，但没有可加载的已
发布权重。

## 导出

<export-matrix />

<code-tabs name="export" />

## 许可证

<provenance-box>

LibreYOLO 不托管也不镜像任何 L2CS 检查点：与本站大多数其他家族不同，LibreYOLO 的
Hugging Face 组织下没有这个家族的任何东西。库能自动获取的那一份检查点直接来自作
者本人的 Google Drive 分发，下载开始前会先打印 Gaze360 的许可证声明作为门槛，它
并不是上面的摘要所暗示的那份「在 huggingface.co/LibreYOLO 重新发布」的副本。

</provenance-box>
