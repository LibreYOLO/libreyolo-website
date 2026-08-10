---
title: D-FINE
families: [dfine]
seo_title: "D-FINE：在 MIT 许可下微调、验证并导出"
description: "在 LibreYOLO 里用 D-FINE 做目标检测和实例分割。安装、预测、微调、验证、导出，代码采用 MIT 许可。"
lead: "一个检测 transformer，它把检测框回归重新表述成每条框边上的概率分布，并在各解码器层之间逐步细化。LibreYOLO 支持它做目标检测和实例分割。"
keywords: [D-FINE, "检测 transformer", "实时目标检测", "实例分割", "d-fine 训练自己的数据集", DETR]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDFINEn.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 文件名里的 -seg 后缀会选中 mask head，所以这里不需要传
        # task 参数
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8, lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: 实例分割
      language: bash
      code: |
        # 从已发布的分割权重继续训练，其中包含 mask head
        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: 从检测权重做分割
      language: bash
      code: |
        # 检测权重不带 mask head，所以这是一次显式迁移：head 一开始
        # 未经训练，只有训练之后才有用；在这里传 task=segment
        # 就是对这次迁移的授权
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: 实例分割
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # 掩码
        print(metrics["metrics/mAP50-95(B)"])   # 检测框
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640
        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

D-FINE 不需要任何可选 extra。它导入的一切都在基础安装里。

```bash
pip install libreyolo
```

用 `lora=True` 做适配器（adapter）微调是个例外，它需要 `lora` extra。

```bash
pip install "libreyolo[lora]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是一行的
改动。`-seg` 文件名自己就能解析到分割任务，此时 `result.masks` 会在检测框之外带上
实例掩码。`conf` 和 `max_det` 过滤的是 query 的选择；`iou` 为了 API 一致性而被接受，
但没有任何作用，因为解码器是一个集合预测器，没有 NMS 步骤。数据源、流式处理和结果
处理见[预测](/docs/predict)。

## 变体

五种尺寸。它们都在相同的输入分辨率下运行，所以下表用参数量和精度来区分它们。

<benchmark-table task="detect" />

<va-embed />

分割复用了检测的骨干、编码器和解码器，再加上一个 mask head，所以 `-seg` 检查点
（checkpoint）接受的参数和它的检测版兄弟完全一样。LibreYOLO 的 RT-DETRv4 家族写成
了 D-FINE 包装类的子类：它继承了这条解码器路线，然后把自己的任务列表钉回检测，因为
它不带 mask head。

## 训练

两个任务的训练都从已发布的检查点开始。

<code-tabs name="train" />

保持默认时，训练器以 `lr0=2e-4`、`amp=False`、批大小 16 跑 132 轮，并在 50 轮没有
提升后早停。检测权重可以作为分割训练的合法起点，但只能作为一次显式迁移，因为 mask
head 一开始未经训练，否则会返回毫无意义的掩码。给 CLI 传 `task=segment` 就是对它的
授权。Python 这条路更窄：必须直接构造 `LibreDFINE` 并传
`allow_detect_to_segment_transfer=True`，因为 `LibreYOLO()` 工厂不接受这个参数；而
直接构造不会下载，所以权重文件必须已经在磁盘上。

`lora=True` 适用于检测。分割训练会拒绝它，并转而指向 `freeze='backbone'`，因为
mask head 还没有和适配器一起测试过。在 Apple silicon 上，训练器会把整个训练过程搬到
CPU：Integral 的分箱矩阵乘法在反向传播时会撞上 Metal 编译失败。MPS 上的推理不受影响。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个以指标名为键的字典，并在 `verbose` 保持开启时打印每个类别的结果。

<code-tabs name="val" />

对 `-seg` 检查点来说，普通的 `metrics/mAP50-95` 键里放的是掩码分数，同一次运行还会
在 `(B)` 下报告检测框、在 `(M)` 下报告掩码，所以一次跑完两者都有。

## 导出

<export-matrix />

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine` 文件
的表现和检查点一样，返回同样的 `Results`。OpenVINO、Paddle、MNN 和 Core AI 这几条
路径导出时用的是固定画布，而不是动态形状。[导出](/docs/export)列出了每种格式接受的
参数，以及其中少数几种额外增加的参数。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

分割权重还有第二个上游：它们的掩码解码器、掩码匹配和掩码损失函数来自
ArgoHA/D-FINE-seg，同样采用 Apache-2.0 许可，其维护者已同意在署名的前提下复用。

</provenance-box>

## 引用

<citation-block />
