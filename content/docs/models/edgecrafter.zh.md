---
title: EdgeCrafter
families: [ec]
seo_title: "EdgeCrafter：在 LibreYOLO 里检测、估计姿态并分割"
description: "在 LibreYOLO 里用 EdgeCrafter 做检测、姿态和实例分割。安装、预测、验证并导出，代码采用 MIT 许可。"
lead: "一个紧凑的视觉 transformer，面向边缘硬件上的稠密预测，上游以三个同胞模型的形式发布：ECDet、ECPose 和 ECSeg。LibreYOLO 把三者作为一个家族加载，任务由检查点（checkpoint）携带。"
keywords: [EdgeCrafter, ECDet, ECPose, ECSeg, "轻量视觉 transformer", 目标检测, 姿态估计, 实例分割, "边缘设备 推理"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 姿态
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -pose 后缀会选中关键点 head，所以这里不需要传
        # task 参数
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50 imgsz=640 batch=8 lr0=5e-4
    - label: 姿态
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 需要一个单类别的关键点数据集，它的 data.yaml 里要声明
        # kpt_shape，并且 imgsz 取检查点的原生尺寸
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 需要多边形标注，并且 imgsz 取检查点的原生尺寸
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: 姿态
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 掩码
        print(metrics["metrics/mAP50-95(B)"])   # 检测框
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

EdgeCrafter 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

用 `lora=True` 做适配器微调是个例外，它需要 `lora` extra。

```bash
pip install "libreyolo[lora]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

任务来自文件名，所以 `-pose` 或 `-seg` 检查点会选中自己的 head，不需要传 task
参数。三者返回的都是每个家族都会返回的那个 `Results` 对象，姿态额外多出
`result.keypoints`，分割额外多出 `result.masks`。姿态只覆盖一个类别，也就是人，
带 17 个 COCO 关键点，数量在模型构建时就固定了。它没有检测框 head，所以每个姿态
框都是自身关键点的包围范围，而第三个关键点通道是一个常数，不是逐点的分数。

`conf` 和 `max_det` 过滤 query 的选择；`iou` 为了 API 一致性会被接受，但没有任何
作用，因为三个 head 解码的都是一组 query，没有 NMS 步骤。数据源、流式处理和结果
处理见[预测](/docs/predict)。

## 变体

四种尺寸。它们跑的输入分辨率都相同，所以表格按参数量和精度来区分。

<benchmark-table task="detect" />

<va-embed />

上游把 ECDet、ECPose 和 ECSeg 发布成三个独立的模型，而不是一个带三个 head 的
模型。它们共用 ECViT 骨干和混合编码器，只在 head 上有差别，所以 LibreYOLO 把它们
收进同一个家族，让检查点文件名来携带任务。因此同一个尺寸字母在三者之间意味着相同
的骨干和编码器，而且不管你加载的是哪一个，预测、验证和导出接受的参数都一样。

## 训练

三个任务都通过 `train()` 训练，它从加载的检查点里读出任务，再挑选对应的训练器。

<code-tabs name="train" />

检测和分割上已经验证过的是：与上游的推理一致性在 1e-5 以内，逐层、逐尺寸都比对过，
以及损失函数和单步训练能在合成输入上跑通。按 `train()` 自己的 docstring，尚未验证
的是：完整微调的收敛性、多卡训练、停止数据增强后重新加载最佳权重这一步，以及
Objects365 到 COCO 的类别重映射。姿态这条路走的是 DETRPose 已发布的配方，即在类别、
关键点 L1 和 OKS 代价上做匈牙利匹配，并配合对比式关键点去噪，它的收敛性同样没有
端到端验证过。

保持默认时，训练器以 `lr0=5e-4` 跑 74 轮，混合精度开启，遵循上游的配方：AdamW、
平坦的余弦调度、0.9999 的 EMA 以及 ImageNet 归一化的输入。姿态和分割都要求 `imgsz`
取检查点的原生尺寸，因为它们的评估锚点网格是在模型构建时建好的；换成别的值会在
运行开始之前就抛错。姿态还要求数据集是单类别的，它的 `data.yaml` 里要声明
`kpt_shape`，关键点数量要和 head 匹配。

`lora=True` 只适用于检测；姿态和分割碰到它会抛出 `ValueError`。在 Apple silicon
上，训练器把整个训练留在 GPU 上，只把一个操作发到 CPU：可变形注意力内部的
grid-sample 反向传播，PyTorch 没有在 Metal 里实现它。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个以指标名为键的字典，并在 `verbose` 保持开启时打印每个类别的结果。

<code-tabs name="val" />

姿态在 `metrics/keypoints_*` 下报告关键点的 OKS 指标。分割把掩码放在普通的
`metrics/mAP50-95` 键里，并在同一次运行中给出两种视角，检测框在 `(B)` 下，掩码在
`(M)` 下。

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。姿态和分割导出时用的是固定的
640×640 输入，而不是动态形状，几个检测目标同样是固定画布，包括 OpenVINO、Paddle、
MNN、ExecuTorch 和 Core AI。[导出](/docs/export)列出了每种格式接受的参数，以及
其中少数几种额外增加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box></provenance-box>

## 引用

<citation-block />
