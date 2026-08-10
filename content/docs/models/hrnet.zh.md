---
title: HRNet
families: [hrnet]
seo_title: "HRNet：LibreYOLO 里的自顶向下姿态估计"
description: "在 LibreYOLO 里用 HRNet 做自顶向下的 COCO-17 姿态估计。采用 MIT 许可，可以安装、预测、验证并导出 W32 和 W48 检查点。"
lead: "HRNet 是一个卷积网络，它靠反复的多尺度融合始终保持一路高分辨率特征流，而不是先下采样再把分辨率恢复回来。LibreYOLO 封装了官方的自顶向下姿态变体，用于推理和验证。"
keywords: [HRNet, "人体姿态估计", "hrnet 姿态估计", "自顶向下姿态估计", "COCO-17 关键点"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 没有给出人体来源：HRNet 会自动给自己配一个轻量的 LibreYOLO9t
        # 检测器，并把这个选择记录一次
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreHRNetw32-pose.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 人体来源
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        # 完全跳过检测：把整张图当作一个人
        result = model(SAMPLE_IMAGE, cropped=True)

        # 或者把你已经跑过的检测器给出的检测框交给 HRNet
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        # 或者给它配一个指定的 LibreYOLO 检测器，代替默认的
        # LibreYOLO9t
        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        import numpy as np
        import onnxruntime as ort

        # 导出的计算图只有固定画布的热力图 head：它接收一批已经裁好、
        # 已经归一化的人体裁剪图，返回原始热力图。人体检测、裁剪几何、
        # 热力图解码和 OKS 抑制都不在这个计算图里；在 LibreYOLO 之外
        # 运行它，意味着解码这一步得你自己重新实现一遍
        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")
        name = session.get_inputs()[0].name
        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
---

## 安装

HRNet 不需要基础包之外的任何 extra。

```bash
pip install libreyolo
```

它默认的人体检测器是一个轻量的 LibreYOLO9t 检查点（checkpoint），在 HRNet 第一次
与它配对时自动下载。

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

HRNet 是一个自顶向下的姿态估计器：姿态 head 能跑起来之前需要有一个人体检测框，所
以每次调用都会先解析出一个。不去管它的话，它会在第一次调用时自动给自己配上一个
LibreYOLO9t 检测器，并记录这个选择。`cropped=True` 跳过检测，把整张图当作一个人；
`person_boxes` 接收你已经跑过的检测器给出的检测框；`person_detector` 接收
`"auto"`、`"rfdetr"`、任何 LibreYOLO 检测模型，或者一个普通的可调用对象。
`flip_test=True` 会在水平翻转后的裁剪图上再跑一次模型，并对两张热力图取平均，这是
HRNet 自带的测试时增强（TTA）；通用的 `augment=True` 在这里没有定义。多图数据源按
顺序逐张运行：HRNet 的检测器，以及每张图上不固定的人数，都不支持堆叠预测。数据
源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

两种尺寸，`w32` 和 `w48`，都是从固定分辨率的人体裁剪图预测标准的 COCO-17 关键点集
合；`w48` 是两个骨干里更宽的那个。

上游的 model zoo 用它自己的人体检测器、自己的翻转测试设置和官方的 COCO 评测协议，
给出每种尺寸的姿态精度。LibreYOLO 默认配的是另一个检测器，所以在这里跑一次验证测
的是这个组合，而不是上游那个；要对上上游的数字，需要与原始评测相同的人体检测框、
检测器分数和翻转设置。

## 验证

`val()` 跑的是 COCO 风格的关键点 OKS-AP，接收 YOLO-pose 的 `data.yaml`，或者一份
COCO 关键点 JSON 加一个图像目录。指标后端默认是 faster-coco-eval，没有装
faster-coco-eval 时会自动改用 `pycocotools`；`faster_coco_eval=False` 强制走
`pycocotools` 这条路径。

<code-tabs name="val" />

验证在内部驱动的是 HRNet 自己的 `predict()`，所以它用的是模型构造时或调用时给定的
那个人体检测器。构造模型时显式传入 `person_detector=`，可以让这个来源在多次运行之
间保持固定，而不是让每次调用都重新解析一遍默认值。

## 导出

<export-matrix />

HRNet 的导出约定只覆盖 ONNX、TorchScript、OpenVINO 和 TensorRT；其他任何格式都会在
追踪开始前抛出异常。每次导出的都只是固定画布的热力图 head，批大小为一的 FP32，接收
一张人体裁剪图并返回原始热力图：它前面的仿射裁剪几何，以及它后面的热力图解码、翻转
还原和 OKS 抑制，都留在 Python 里，所以一条完整的「图像进、关键点出」流水线在另一端
仍然需要 LibreYOLO。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
