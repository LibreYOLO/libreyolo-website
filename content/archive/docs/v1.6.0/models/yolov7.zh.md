---
title: YOLOv7
families:
  - yolo7
seo_title: LibreYOLO 里的 YOLOv7：预测、训练与导出，采用 MIT 许可
description: 在 LibreYOLO 里用 YOLOv7 做目标检测：安装、预测、训练、验证并导出，代码和权重都采用 MIT 许可。
lead: >-
  YOLOv7 是一个基于锚框的单阶段检测器，它的 head 在最后一层卷积之前加上学到的隐式知识（implicit
  knowledge）偏移量。LibreYOLO 支持它已发布的那一个尺寸，用于目标检测。
keywords:
  - YOLOv7
  - ImplicitA
  - yolov7 目标检测
  - yolov7 pytorch
  - yolov7 训练自己的数据集
  - 隐式知识
  - 实时目标检测
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: 从一个全新模型热启动
      language: python
      code: |
        from libreyolo import LibreYOLO7

        # pretrained=True 总是加载已发布的 LibreYOLO7b.pt 检查点，
        # 无论这个实例构造时用的是什么；直接构造这个类，
        # 而不是经由 LibreYOLO()，一开始不会加载任何权重
        model = LibreYOLO7(None, size="b")
        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO7b.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## 安装

YOLOv7 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行改
动。`conf` 设置置信度阈值，`iou` 设置 NMS 阈值，后者在基于锚框的 head 解码完成之
后施加。数据源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

LibreYOLO 只提供一种尺寸 `b`。上游只发布了一个 YOLOv7 模型，所以没有尺寸可选。

## 训练

<code-tabs name="train" />

`pretrained` 在这里是会被读取的，不像这边另外几个家族上同名的那个空操作：传
`True` 就从已发布的 `LibreYOLO7b.pt` 检查点（checkpoint）热启动（会自动下载），
传路径或名字则加载别的权重。那份已发布的检查点是 80 类的 COCO 权重，所以在一个
已经按别的类别数重建过的模型上要求它，会先重建回 80 类、把它加载进来，再在读到
数据集的类别数之后，把每一个形状匹配的张量迁移到目标 head 的类别数上。
`resume=True` 不能和 `pretrained` 一起用。保持默认的 `None` 时，训练从模型构造时
所带的权重继续，如果什么都没加载过，就从随机初始化开始。

其余不动时，训练器以 `lr0=0.01` 跑 300 轮，用动量 0.937 的 SGD、3 轮预热，以及和
YOLOX 相同的 SimOTA 分配和最后 15 轮不做数据增强的阶段，只是适配到了基于锚框的
head 上。唯一的区别：YOLOX 在收尾那几轮里加了一个 L1 检测框回归精修，v7 跳过了
它，因为 v7 的 SimOTA 损失函数里没有可供精修的原始偏移量 L1 分支。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在不装 LibreYOLO 的裸运行时里跑这
张计算图也是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
