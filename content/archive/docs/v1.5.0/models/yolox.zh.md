---
title: YOLOX
families: [yolox]
seo_title: YOLOX：在 Apache-2.0 许可下预测、训练与导出
description: 在 LibreYOLO 里用 YOLOX 做目标检测：安装、预测、训练、验证、导出，全部采用 Apache-2.0 许可。
lead: YOLOX 是一个无锚框（anchor-free）单阶段检测器，带一个分类与回归解耦的 head，用 SimOTA 标签分配训练。LibreYOLO 支持它做目标检测。
keywords:
  - YOLOX
  - 目标检测
  - 无锚框检测
  - yolox 解耦头
  - SimOTA
  - 实时目标检测
  - yolox 训练自己的数据集
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLOXs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16, lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: 在 COCO 上验证
      language: bash
      code: |
        # 自带的 coco.yaml 带着一个内嵌的下载脚本，所以除非数据集已经在
        # 本地，否则需要显式放行
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640
        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLOXs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## 安装

YOLOX 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。`conf` 设置置信度阈值，`iou` 设置在三个解耦的预测尺度之间施加的 NMS 阈值。
数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

六种尺寸共用同一个 CSP 骨干和 PAFPN neck。最小的两个 `n` 和 `t` 跑在比另外四个更小
的固定输入分辨率上；下面的基准测试表格给出了每一个的确切数值。

<benchmark-table task="detect" />

<va-embed />

## 训练

<code-tabs name="train" />

保持默认时，训练器以 `lr0=0.01` 跑 300 轮，用 SGD 动量 0.9、5 轮预热，并在最后 15
轮关掉 mosaic 和 mixup 数据增强。`train()` 还接受一个 `pretrained` 参数，但方法内部
从不读取这个值：训练总是从构造模型时所用的那份权重继续，所以 `pretrained=False`
并不会重新初始化网络。

`imgsz` 在基础训练配置文件里默认取一个固定值，而不是取加载的检查点（checkpoint）
的原生分辨率。这尤其影响 `n` 和 `t` 这两个检查点：不显式设置 `imgsz` 就继续训练其中
任何一个，都会把它抬到更大的那个默认值，而不是它发布时所用的更小的尺寸。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在不装 LibreYOLO 的裸运行时里跑这张
计算图也是支持的，但那样预处理和后处理就得你自己写。CoreML 导出可以用 `nms=True`
把 NMS 内嵌进计算图；YOLOX 和 YOLOv9 是目前这个参数唯一接受的两个家族。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
