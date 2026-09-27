---
title: RF-DETR
families:
  - rfdetr
seo_title: RF-DETR：在 MIT 许可下训练、微调并导出
description: 在 LibreYOLO 里用 RF-DETR 做目标检测、实例分割、姿态和旋转框。安装、预测、训练、验证、导出，全部采用 MIT 许可。
lead: 一个检测 transformer，它预测的是一组固定数量的目标，而不是一张稠密网格，所以推理时不需要 NMS。LibreYOLO 支持它做四种任务。
keywords:
  - RF-DETR
  - 实时检测 transformer
  - DETR
  - 目标检测 python
  - 实例分割
  - 姿态估计
  - 旋转框检测
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: LibreRFDETRs，在视频上以 512 px 做检测。
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # 库接受的任何数据源：文件、文件夹、URL、摄像头索引、
        # RTSP 流，或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() 返回的是一个普通 dict，不是对象
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: 在 COCO 上验证
      language: bash
      code: |
        # 内置的 COCO yaml 带着一个内嵌的下载脚本，所以除非数据集已经
        # 在本地，否则需要显式放行
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)

        # 每种格式都接受的参数：
        #
        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"
        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"
        #             | "tflite" | "coreml" | "coreai"。
        #             "engine" 是 tensorrt 的别名，"litert" 是 tflite 的别名
        #   imgsz     int，或 (height, width)。默认为检查点的原生分辨率
        #   batch     int，默认 1
        #   half      bool，以 FP16 导出。默认 False
        #   int8      bool，以 INT8 导出。默认 False。需要 `data`
        #   data      数据集 YAML 的路径，用来校准 int8
        #   fraction  float，取该校准集的多大比例。默认 1.0
        #   dynamic   bool，动态轴。默认 True
        #   simplify  bool，跑一次 ONNX 计算图简化。默认 True
        #   opset     int，ONNX opset。不给时按家族选定
        #   device    str，在哪个设备上做 trace。默认为模型所在的设备
        #   output_path  str，默认是一个由检查点派生出来的名字
        #   verbose   bool，默认 False
        #   allow_download_scripts  bool，默认 False。放行一个必须下载的
        #             数据集 YAML 里内嵌的 Python
        #
        # 少数几种格式还接受自己额外的参数，比如 RKNN 的目标平台。
        # 这些参数记录在各自格式的页面上
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreRFDETRs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
    - label: 不用 LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # 直接跑这张计算图，意味着预处理和后处理都得你自己做。动手

        # 接线之前先看清楚签名

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## 安装

RF-DETR 需要自己的 extra，它会为骨干拉进 `transformers`。

```bash
pip install "libreyolo[rfdetr]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`conf` 和 `max_det` 过滤的是 query 的选择；没有 NMS 步骤需要调。数据源、
流式处理和结果处理见[预测](/docs/predict)。

## 变体

四种尺寸，四种任务共用一套架构：分割、姿态和旋转框复用检测的解码器，只是换了一个
head，所以接受的参数完全一样。这些尺寸的参数量相近，主要差别在输入分辨率。

<benchmark-table task="detect" />

<va-embed />

## 训练

四种任务的训练都从已发布的检查点（checkpoint）开始。RF-DETR 把 `pretrained` 列在
它原生训练器会忽略的参数里，所以在这里传 `pretrained=False` 并不会给你一个随机
初始化的模型。

<code-tabs name="train" />

有两个参数在这里比在 CNN 检测器上更要紧。把 `lr0` 保持在 `1e-4` 或更低，因为 YOLO
模型能扛住的学习率会让检测 transformer 发散。除非你有理由改，否则把 `imgsz` 留在
检查点的原生分辨率上。输入必须能被骨干的 patch 大小乘以窗口数整除；LibreYOLO 会在
训练开始前检查这一点，并给出最接近的合法尺寸。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何与你训练时所用格式相同的数据集上测得。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在一个不装 LibreYOLO 的裸运行时里跑
这张计算图同样是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
