---
title: 快速开始
seo_title: LibreYOLO 快速开始
description: 在一张图片上跑检测器，在小数据集上微调，再导出为 TorchScript 或 ONNX，全程只用 CPU，大约十行 Python 代码。
lead: 走通 LibreYOLO 的最短路径：在一张图片上预测，在小数据集上训练，然后导出结果。这里的每条命令都能在 CPU 上跑。
keywords:
  - libreyolo 快速开始
  - libreyolo 教程
  - libreyolo 预测
  - libreyolo 训练自己的数据集
  - libreyolo 导出 onnx
  - yolo python 示例
last_verified: 1.5.0
meta:
  - label: 安装
    value: pip install libreyolo
    mono: true
  - label: 检查点
    value: LibreYOLO9t.pt
    mono: true
  - label: 硬件
    value: 本页的所有内容用 CPU 就够了
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # 首次使用时下载检查点，之后缓存在 weights/ 下
        model = LibreYOLO("LibreYOLO9t.pt")

        # 单张图片返回一个 Results 对象
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 视频与流
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True 每帧产出一个 Results，而不是先构建一个列表
        # 把路径换成摄像头编号、RTSP URL 或一个文件夹
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # coco8 是库自带的 8 张图片的数据集，首次使用时从 URL 下载，
        # 因此不需要执行任何脚本
        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )

        print(results["save_dir"])
        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: 验证
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() 返回一个普通 dict，不是对象
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # export() 返回它写出的路径
        path = model.export(format="torchscript")
        print(path)

        # 工厂函数按文件后缀分发，所以导出产物可以像检查点一样加载回来，
        # 并返回同样的 Results 对象
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## 安装

```bash
pip install libreyolo
```

下面的预测和训练两节需要的就这些。导出为 ONNX 要多装一个 extra；完整列表见[安装](/docs/install)。

## 预测

<code-tabs name="predict" />

`LibreYOLO()` 是一个工厂函数。它读取文件，判断这份权重属于哪个家族，然后返回该家族的模型，所以换用另一个检测器只是改一行的事。传入不带目录的 `LibreYOLO9t.pt`，会相对工作目录去找 `weights/LibreYOLO9t.pt`，文件不在时就下载到那里。下载规则和离线使用方式见[检查点与权重](/docs/weights)。

`save=True` 会在 `runs/detect/` 下写出一份带标注的副本，放进每次运行递增编号的 `predict` 目录。返回的 `Results` 带有 `boxes`，`names` 把类别索引映射到它的标签。单张图片路径返回一个 `Results`；目录、图片列表或 `stream=True` 返回它们的列表或生成器。

## 训练

<code-tabs name="train" />

`data` 是一份数据集 YAML。`coco8.yaml` 随库一起提供，所以这段代码片段粘贴过去就能跑；不是自带的名字会按路径读取。数据集在 `~/datasets` 下解析，设置了 `LIBREYOLO_DATASETS_DIR` 时则在该变量指向的位置解析。

一次运行写到 `project/name`，默认是 `runs/train` 下的一个目录，里面有 `weights/best.pt` 和 `weights/last.pt`。`train()` 返回一个字典，包含 `save_dir`、`best_checkpoint`、`last_checkpoint`、每轮的损失和每轮的验证指标。训练得到的检查点（checkpoint）通过 `LibreYOLO()` 加载，和预训练的完全一样。

并不是每个家族都能训练。只提供推理的家族在调用 `train()` 时会抛出 `NotImplementedError` 并说明这一点。[核心概念](/docs/concepts)解释了各个支持等级分别代表什么。

## 导出

<code-tabs name="export" />

TorchScript 除了基础安装之外不需要任何东西。其他目标各有自己的 extra，而且覆盖范围是按家族、按任务来的，并不统一：见[导出与部署](/docs/export)。

每种格式都接受的参数包括 `imgsz`（一个整数，或者高和宽一对值）、`batch`（默认 1）、`half`、`int8`（配合一份用于校准的 `data` YAML）、`dynamic`（默认 True）、`simplify`（默认 True）、`opset`、`device` 和 `output_path`。省略 `output_path` 时，文件写到 `weights/` 下，名字从检查点派生。

## 下一步看什么

- [核心概念](/docs/concepts)：任务、家族、尺寸和检查点命名。
- [检查点与权重](/docs/weights)：自动下载、离线使用和加载安全。
- 如果你已经有上游项目的检查点，看[导入已有权重](/docs/migrate)。
- [全部模型](/docs/models)：找到适合你问题的那个家族。
- [训练](/docs/train)、[预测](/docs/predict)和[导出](/docs/export)：完整的工作流程。
