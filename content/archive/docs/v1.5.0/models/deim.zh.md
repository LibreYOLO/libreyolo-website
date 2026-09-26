---
title: DEIM
families:
  - deim
seo_title: 在 LibreYOLO 里使用 DEIM 和 DEIMv2
description: 用 LibreYOLO 跑 DEIM 和 DEIMv2 做目标检测。安装、预测、训练、验证和导出，从五十万参数的尺寸往上都有。
lead: >-
  一个用密集一对一匹配训练的检测 transformer，收敛所需的轮数远少于它所基于的 DETR 方案。LibreYOLO
  收录了它的两个版本，靠你加载的检查点（checkpoint）区分。
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - 检测 transformer
  - DETR
  - 目标检测 python
  - 实时目标检测
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 视频
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 版本是文件名的一部分，工厂又按检查点分发，所以两者的加载方式
        # 完全一样
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # 库接受的任何数据源：文件、文件夹、URL、摄像头索引、
        # RTSP 流，或者一个 .streams 列表
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml 在首次使用时下载一份 128 张图的样本；真正要跑的时候，
        # 把 `data` 指向你自己的数据集 YAML
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # 不设置时，epochs、batch、imgsz 和 lr0 取自所加载尺寸对应的
        # 已发布方案
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # 需要 lora extra：pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: 多卡训练
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() 返回的是普通 dict，不是对象
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: 在 COCO 上验证
      language: bash
      code: |
        # coco-val-only.yaml 会取回 5000 张 val2017 图片并跳过训练集；
        # 它带着一个内嵌的下载脚本，所以除非数据集已经在本地，否则
        # 需要显式放行
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # 需要 onnx extra：pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## 安装

两个版本都不需要任何可选 extra。它们导入的一切都在基础安装里。

```bash
pip install libreyolo
```

用 `lora=True` 做适配器微调是例外，它需要 `lora` extra。

```bash
pip install "libreyolo[lora]"
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象和每个家族返回的都是同一个，所以换成另一个检测器只是改一
行的事。`conf` 和 `max_det` 过滤的是在 query 和类别上做的 top-k 解码；没有 NMS
步骤需要调，`iou` 会被接受但不会用到。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

版本 1 提供五种尺寸，输入尺寸都相同。版本 2 保留了这五个名字，又加了三个更小
的——`atto`、`femto` 和 `pico`，其中前两个的原生输入尺寸比其余的更低。因此有五
个尺寸代号在两个版本里都存在，指向的却是不同的模型；版本写在检查点的文件名里。

<benchmark-table task="detect" />

<va-embed />

版本 1 沿用 D-FINE 的架构，把它的分类目标换成了密集一对一方案里的
matchability-aware 损失函数，所以这两个家族几乎共享全部 state dict 键，靠检查点
里的元数据区分。版本 2 保持同样的训练约定，并混用骨干：`s` 以下是 HGNetv2，`s`
及以上是带空间调优适配器的 DINOv3 vision transformer。正是这个骨干给那四个检查
点加上了第二份许可证，所以在你把其中之一发布出去之前，先读一读
[许可证](#licensing)。

## 训练

训练从一个已发布的检查点开始。`pretrained` 永远到不了训练器：版本 1 会警告这个
键未知然后忽略它，版本 2 会把它删掉。两者都不会给你一个随机初始化的模型。

<code-tabs name="train" />

在版本 1 上要自己传 `lr0`。它的 Python `train()` 签名默认是 `4e-4`，也就是已发布
的 COCO 方案里的学习率，而这个家族的训练配置文件把 `1e-4` 作为微调默认值，参数
缺省时 CLI 解析出来的正是这个更低的值。配置文件记录了背后的实测：在微调实际会
用到的批大小下，在小数据集上，COCO 的学习率明显拖累了迁移效果。

版本 2 自己解析这些默认值。不设置 `epochs`、`batch`、`imgsz` 和 `lr0`，它就会为
所加载的那个尺寸从发布的方案里逐个读取，所以小尺寸不用你交代就会用自己的输入分
辨率训练，而你传入的值会覆盖方案。`imgsz` 是它会加约束的那个参数：必须是 32 的
正整数倍，否则版本 2 会在运行开始前抛错。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和
mAP 50-95，在任何与你训练时格式相同的数据集上测量。

<code-tabs name="val" />

上面基准测试表里的行来自 LibreYOLO 的基准测试框架；表下的说明记录了它们出自哪个
数据集，并链接到运行记录。

## 导出

<export-matrix />

这份矩阵把两个版本放在同一页上：两者对某种格式的支持不一致时，单元格显示的是两
者中较弱的那个，所以不论你加载的是哪个版本，这里都不会夸大。

导出的产物按文件后缀经由 `LibreYOLO()` 重新加载，所以一个 `.onnx` 或 `.engine`
文件的表现和检查点一样，返回同样的 `Results`。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>
从 S 往上的四个 DEIMv2 尺寸，骨干取自 DINOv3，所以它们的权重仓库同时带有
Apache-2.0 和 Meta 的 DINOv3 License，LibreYOLO 随库提供的 DINOv3 骨干源码也适用
同一份许可。这个家族的其余部分，包括 S 以下的每个 DEIMv2 尺寸，只采用 Apache-2.0
许可。
</provenance-box>

## 引用

<citation-block />

DEIMv2 是另一篇论文，在
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation)
有它自己的引用块；如果你用的是版本 2 的检查点，就引用那一篇。
