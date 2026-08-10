---
title: RTMDet
families: [rtmdet]
seo_title: "在 LibreYOLO 里使用 RTMDet：预测、训练与导出"
description: "在 LibreYOLO 里用 RTMDet 做目标检测，用 RTMDet-Ins 做实例分割。安装、预测、训练、验证、导出，代码采用 Apache-2.0 许可。"
lead: "RTMDet 是一个单阶段检测器，它在每个网格位置上只用一个基于点的先验来预测，没有锚框，经过的 head 在各个特征层级之间共享卷积。LibreYOLO 支持它做目标检测和 RTMDet-Ins 实例分割。"
keywords: [RTMDet, "目标检测 python", "实例分割", RTMDet-Ins, "无锚框检测", "rtmdet 不用 mmdetection"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTMDets.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -seg 后缀会选中 RTMDet-Ins 的 mask head，
        # 所以这里不需要传 task 参数
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 掩码
        print(metrics["metrics/mAP50-95(B)"])   # 检测框
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640 epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640
        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreRTMDets.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

除基础包之外，RTMDet 不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`-seg` 文件名自己就能解析到 RTMDet-Ins 任务，此时 `result.masks` 会在检测框
之外带上实例掩码。`conf` 设置置信度阈值，`iou` 设置 NMS 阈值。数据源、流式处理和
结果处理见[预测](/docs/predict)。

## 变体

五种尺寸，从 `t` 到 `x`，在同一个输入分辨率下共用一套架构。这个家族在这里没有基准
测试表：比较尺寸请看下表里检查点（checkpoint）的文件大小。

## 训练

<code-tabs name="train" />

检测通过 `train()` 训练。QualityFocalLoss、GIoU 和 DynamicSoftLabelAssigner 这几个
组件移植自上游 mmdetection，前向传播和 ONNX 导出与它逐位一致，后处理在 val2017 子集
上与 mmdet 的输出相差在 0.001 mAP 以内。

按 `train()` 自己的 docstring，尚未验证的部分有：小数据集微调的收敛性、从头训练与
论文的对齐、多卡训练的行为、带缓存的 Mosaic 与 MixUp 的吞吐、上游那个严格的两阶段
流水线切换，以及把 norm 和 bias 参数的衰减置零的 paramwise weight-decay 覆写。

RTMDet-Ins 没有训练路径。对 `-seg` 检查点调用 `train()`，或者传 `task="segment"`，
都会抛出 `NotImplementedError`；实例分割只支持推理和验证。

`train()` 还接受一个 `pretrained` 参数，但方法内部从不读取它的值：训练总是从模型
构造时所带的权重继续，所以 `pretrained=False` 并不会重新初始化网络。

其余保持默认时，训练器用 AdamW 跑 300 轮，`lr0=0.004`、`weight_decay=0.05`，按余弦
调度做 1 轮预热，并在最后 20 轮关掉 Mosaic 和 MixUp。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和 mAP 50-95，
在任何与你训练所用格式相同的数据集上测量。

<code-tabs name="val" />

对 `-seg` 检查点来说，普通的 `metrics/mAP50-95` 键里放的是掩码分数，同一次运行还会
在 `(B)` 下报告检测框、在 `(M)` 下报告掩码，所以一次跑完两者都有。

## 导出

<export-matrix />

检测能导出到大多数格式；实例分割目前一种都导不出；上面的矩阵反映的就是这个分界。
导出的检测产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。在没有装 LibreYOLO 的裸运行时里跑
这张图也是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
