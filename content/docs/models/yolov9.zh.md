---
title: YOLOv9
families:
  - yolo9
seo_title: YOLOv9：在 MIT 许可下预测、训练与导出
description: 在 LibreYOLO 里运行 YOLOv9，包括无 NMS 的端到端 head 和 stride-4 小目标 head。可以安装、预测、训练、验证和导出。
lead: 一个单阶段卷积检测器：一次前向就为密集的检测框网格打分，再由 NMS 去掉重复的那些。LibreYOLO 收录了它的三个变体，其中一个没有 NMS 步骤。
keywords:
  - YOLOv9
  - YOLO9
  - yolov9 目标检测
  - 无 nms 检测
  - 端到端检测
  - 小目标检测
  - yolov9 导出 onnx
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 不用 NMS
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 同样的调用，换一个检查点。端到端 head 返回自己得分最高的预测，
        # 所以不跑 NMS，iou 也被忽略
        model = LibreYOLO("LibreYOLO9E2Es.pt")
        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)

        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: 小目标
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # stride-4 变体没有自己的 COCO 检查点，所以指定一个基础检测的

        # 检查点：它的骨干和 neck 原样加载，stride-4 的那组 head 从随机

        # 初始化开始

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: 在 COCO 上
      language: bash
      code: |
        # 内置的 COCO yaml 带有一段内嵌的下载脚本，所以除非数据集已经
        # 在本地，否则需要显式放行
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: 把 NMS 放进计算图
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreYOLO9s.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## 安装

YOLOv9 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都会返回的那一个，所以换成另一个检测器只是改一行
的事。在基础模型和 stride-4 模型上，`conf` 设置置信度阈值，`iou` 设置 NMS 阈值。端
到端模型不跑 NMS，也忽略 `iou`，所以决定它输出形态的是 `conf` 和 `max_det`。数据
源、流式处理和结果处理见[预测](/docs/predict)。

## 变体

三个变体共用一个骨干。三者都只做检测，接受的参数也相同。

基础模型在三个特征尺度上预测，并用 NMS 清掉重复的检测框。

端到端模型保留那个 head，并在它旁边加了一条一对一匹配分支。推理时只读一对一分支，
取它得分最高的预测，所以不跑 NMS。当你部署的运行时没有 NMS 算子时，选它。

stride-4 模型从骨干上再往上取一层，把 neck 向下延伸到那一层，在四个尺度而不是三个
尺度上预测。多出来的这个尺度是给只占几个像素的目标用的；它已发布的那一个检查点
（checkpoint）是在航拍图像上训练的。基础检测检查点可以迁移进来：骨干和 neck 原样
加载，三组预训练的 head 整体上移一格，stride-4 的那组从随机初始化开始。

<benchmark-table task="detect" />

<va-embed />

## 训练

<code-tabs name="train" />

`pretrained` 决定这次训练从什么开始。传 `True` 会加载同一模型、同一尺寸已发布的检查
点，传名称或路径则可以加载别的。形状对不上的张量会被跳过而不是报错，训练还会记录
加载了多少个，所以在不同类别数上训练出来的检查点仍然是一个可用的起点。

stride-4 模型没有自己已发布的 COCO 检查点，所以 `True` 在这里会解析到一个不存在的
文件，下载会失败。改为指定一个基础检测检查点。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

一个勾对三个变体都成立：三者有差异的地方，矩阵取其中最弱的那个。

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文件
的表现和检查点一样，返回同样的 `Results`。在不装 LibreYOLO 的裸运行时里跑这张计算图
也是支持的，但那样预处理和后处理就得你自己写。

对基础检测模型来说，其中后处理的那一半可以搬进计算图。ONNX 导出加上 `nms=True` 会把
抑制放进模型里，第一个输出变成一个固定的 `(1, max_det, 6)` 张量，每一行是
`x1, y1, x2, y2, score, class`，检测数之后补零。这张计算图是 batch 1 的，也不带动态
轴。端到端模型和 stride-4 模型不接受这个参数。

每种格式安装的 extra 不同，也各自带几个自己的参数。两者都在那个格式的页面上。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

这里有一个检查点不是 MIT 许可。在 VisDrone2019-DET 上训练的 stride-4 模型继承了那个
数据集的 CC BY-NC-SA 3.0 条款：只允许非商业使用，任何由它衍生出来的东西都要以相同
方式共享，这也在这个家族其余部分所采用的宽松许可之外。它预测的是 VisDrone 的航拍
类别，而不是 COCO 的类别。库在下载这个文件之前会把这些全部打印出来。

</provenance-box>

## 引用

<citation-block />
