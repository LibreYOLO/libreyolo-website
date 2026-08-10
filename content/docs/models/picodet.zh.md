---
title: PicoDet
families: [picodet]
seo_title: "PicoDet 在 LibreYOLO 里：预测、训练与导出"
description: "在 LibreYOLO 里用 PicoDet 做移动端目标检测。安装、预测、训练、验证并导出，采用 Apache-2.0 许可。"
lead: "PicoDet 是一个为移动端和边缘 CPU 打造的单阶段检测器：ESNet 骨干、CSP-PAN neck，以及一个共享的 Generalized Focal Loss head。LibreYOLO 支持它做目标检测。"
keywords: [PicoDet, PP-PicoDet, "目标检测", "移动端目标检测", "边缘设备目标检测", "picodet 部署", ESNet, "Generalized Focal Loss"]
last_verified: "1.5.0"
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePICODETs.pt source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: |
        # imgsz 值得显式设置：CLI 把它默认成 640，而 s 检查点
        # 原生就是 320
        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320 epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320
        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320 half=True
    - label: 使用导出的文件
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 工厂按文件后缀分发，所以导出的产物加载方式和任何检查点一样，
        # 返回的也是同一个 Results 对象
        model = LibreYOLO("LibrePICODETs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
---

## 安装

PicoDet 在基础包之外不需要任何 extra。

```bash
pip install libreyolo
```

## 预测

权重在首次使用时从 Hugging Face 下载，并缓存在本地。

<code-tabs name="predict" />

返回的 `Results` 对象就是每个家族都返回的那一个，所以换用另一个检测器只是一行
改动。`conf` 设置置信度阈值，`iou` 设置 NMS 阈值。数据源、流式处理和结果处理见
[预测](/docs/predict)。

## 变体

三种尺寸，每种都跑在各自固定的输入分辨率上：`s` 最小，`l` 最大。分辨率随尺寸一起
增长，所以更大的检查点（checkpoint）除了参数量更多，每张图像的运行开销也更高。

<benchmark-table task="detect" />

<va-embed />

## 训练

<code-tabs name="train" />

损失的各个分量和分配器（assigner）沿用上游的配方：VFL、DFL、GIoU 和 SimOTA，配合
分类质量加权和动态 IoU 的 VFL 目标。在同一个检查点上，推理与上游逐位一致。

按 `train()` 自己的 docstring，尚未验证的是：完整数据集上的收敛性、多卡训练的表现，
以及除水平翻转之外的任何数据增强。在库用来测试小规模微调的那个 30 张图像、两个类别的
fixture 上，`s` 检查点以它原生的 320 分辨率也没能稳定通过 LibreYOLO 的精度下限。那个
尺寸更适合完整 COCO 这种规模。

`train()` 还接受一个 `pretrained` 参数，但方法内部从来不读这个值：训练总是从构造模型
时所用的那份权重继续，所以 `pretrained=False` 并不会重新初始化网络。在 Python 里不设
`imgsz`，它就取所加载检查点的原生分辨率，`s` 是 320，`m` 是 416，`l` 是 640。CLI 总会
发一个 `imgsz`，默认是 640，所以要在那边把它设成和检查点匹配。

其余不动的话，训练器会用 SGD 跑 300 轮，`lr0=0.01`，动量 0.9，权重衰减 4e-5，并在余弦
调度上做 1 轮预热。水平翻转是唯一施加的数据增强。

数据集、数据增强、多卡训练和日志记录器见[训练](/docs/train)。

## 验证

`val()` 返回一个由 `metrics/` 键组成的字典，涵盖查准率、查全率、mAP 50 和 mAP 50-95，
在任何采用你训练时所用格式的数据集上测量。

<code-tabs name="val" />

## 导出

<export-matrix />

导出的产物会按文件后缀通过 `LibreYOLO()` 加载回来，所以一个 `.onnx` 或 `.engine`
文件的行为和检查点一样，返回的也是同一个 `Results`。在裸运行时里跑这张计算图、完全
不装 LibreYOLO，同样是支持的，但那样预处理和后处理就得你自己写。

<code-tabs name="export" />

## 检查点

这个家族已发布的全部权重文件。

<checkpoint-table />

## 许可证

<provenance-box>

LibreYOLO 的移植跟随 Bo396543018/Picodet_Pytorch，它是 PaddleDetection 原版
PP-PicoDet 的一个 PyTorch 重实现，剥掉了 mmcv，并把每一个激活函数都精确对齐，这样
经由 Bo 的流水线转换过来的 PaddlePaddle 检查点加载时不会有任何数值漂移。两个来源都
沿用了和论文作者相同的 Apache-2.0 条款。

</provenance-box>

## 引用

<citation-block />
