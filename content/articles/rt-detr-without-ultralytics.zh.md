---
title: "RT-DETR 不用 Ultralytics：v1、v2 和 v4 的 Apache-2.0 版本"
description: "在 MIT 许可的库中运行和微调 RT-DETR、RT-DETRv2 与 RT-DETRv4，使用 Apache-2.0 权重，通过同一套 API 完成预测、训练、验证和导出。"
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "RT-DETR 可以免费商用吗？"
    a: "可以，前提是使用宽松许可的实现。原始 RT-DETR 和 RT-DETRv2 代码（lyuwenyu/RT-DETR）以及 RT-DETRv4（RT-DETRs/RT-DETRv4）采用 Apache-2.0，LibreYOLO 则在 MIT 许可的库中运行这三个版本，并使用 Apache-2.0 权重。ultralytics 包中的 RT-DETR 实现采用 AGPL-3.0，闭源使用的替代方案是付费的企业许可（Enterprise License）。以上是一般性信息，不是法律意见。"
  - q: "RT-DETRv4 采用什么许可？"
    a: "Apache-2.0。github.com/RT-DETRs/RT-DETRv4 仓库中的 LICENSE 文件标明采用 Apache-2.0。RT-DETRv4 只在训练时使用 DINOv3 教师模型；发布的学生模型权重不含 DINOv3 参数，因此 Meta 的 DINOv3 许可不适用于这些权重。"
  - q: "可以不使用 Ultralytics 微调 RT-DETR 吗？"
    a: "可以。LibreYOLO 可以通过 model.train() 微调 RT-DETR、RT-DETRv2 和 RT-DETRv4 检测检查点，也可以在命令行中运行 libreyolo train，从已发布的 COCO 权重开始训练。RT-DETRv2 的旋转框模型仅支持推理。"
  - q: "LibreYOLO 提供哪些 RT-DETR 权重？"
    a: "RT-DETR 提供 r18、r34、r50、r50m、r101、l 和 x；RT-DETRv2 提供 r18、r34、r50、r50m 和 r101；RT-DETRv4 提供 s、m、l 和 x，全部是在 640 px 分辨率下进行的 COCO 目标检测。RT-DETRv2 还提供 n、s、m、l 和 x 旋转框模型，这些模型在 1024 px 分辨率下使用 DOTA v1.0 训练。所有文件都采用 Apache-2.0。"
---

RT-DETR 是百度推出的实时目标检测 transformer。它解码固定数量的查询，而不是密集网格，因此无需调节 NMS 步骤。搜索 RT-DETR 时，你最先找到的实现之一可能就在 `ultralytics` Python 包中。这会引出一个模型本身从未带来的许可问题。

## RT-DETR 的许可取决于仓库

许可属于仓库，而不是模型。RT-DETR 有多个仓库，它们采用的许可并不相同：

| 仓库 | 涵盖内容 | 许可 |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR 和 RT-DETRv2，PyTorch 与 Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | `ultralytics` 包中的 RT-DETR | AGPL-3.0 或企业许可 |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR、RT-DETRv2 和 RT-DETRv4 | MIT 许可的代码，Apache-2.0 权重 |

Ultralytics 的实现很成熟。其 [RT-DETR 页面](https://docs.ultralytics.com/models/rtdetr/)列出了预训练的 `rtdetr-l.pt` 和 `rtdetr-x.pt`，支持训练、验证、推理和导出。它随采用 AGPL-3.0 许可的包一起发布；对于不想公开整个项目源代码的团队，Ultralytics 提供付费的企业许可。如果两者都不适合你，模型架构也可以从其作者处以 Apache-2.0 许可获取。[YOLO 许可说明](/articles/yolo-licenses-explained)介绍了 YOLO 家族中相同的许可模式，[商业许可指南](/articles/yolo-commercial-license)则解释了 AGPL-3.0 对闭源产品意味着什么。

## 使用 LibreYOLO 运行 RT-DETR

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # 首次使用时从 Hugging Face 下载
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

命令行也可以运行相同的调用：

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` 和 `max_det` 会筛选基于查询和类别进行 top-k 解码的结果。由于没有 NMS，`iou` 参数会被接受，但不会使用。所有 LibreYOLO 模型都返回同一种 `Results` 对象，因此更换检测器只需改一行代码。

## 在一个加载器中使用 RT-DETR、RT-DETRv2 和 RT-DETRv4

版本号包含在文件名中，`LibreYOLO()` 会根据检查点选择对应模型，因此三个版本的加载方式相同：

* **RT-DETR：** `LibreRTDETR` 提供 r18、r34、r50、r50m、r101（ResNet 骨干）以及 l、x（HGNetv2）。
* **RT-DETRv2：** `LibreRTDETRv2` 提供 r18、r34、r50、r50m 和 r101。它沿用 v1 的架构，改变了可变形注意力的采样方式。
* **RT-DETRv4：** `LibreRTDETRv4` 提供 s、m、l 和 x。它复用了 D-FINE 的架构，其权重来自将 DINOv3 教师模型蒸馏到 HGNetv2 学生模型的过程。

所有检测权重都使用 COCO 训练，运行分辨率为 640 px。作为参考，上游 README 报告 RT-DETR-R18 在 COCO 上达到 46.5 AP，RT-DETR-L 达到 53.0 AP；RT-DETRv4-S 达到 49.8 AP，RT-DETRv4-X 达到 57.0 AP。[RT-DETR 文档页](/docs/models/rt-detr)列出了 LibreYOLO 每个检查点自己的基准测试表。

RT-DETRv2 还支持旋转框。`LibreRTDETRv2n-obb.pt` 到 `LibreRTDETRv2x-obb.pt` 是官方 DOTA v1.0 检查点，包含 15 个航拍类别，分辨率为 1024 px，并已转换为 LibreYOLO 格式：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[旋转目标检测](/docs/tasks/oriented-detection)介绍了标注格式和指标。

## 使用自己的数据微调 RT-DETR

训练从已发布的检查点开始：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

实际训练时，将 `data` 指向你自己的数据集 YAML 文件。每个版本都有自己的默认学习率；按照原始训练方案，版本 1 和 2 的骨干学习率设为 `lr0` 的二十分之一。命令行中的 `device=0,1` 可用于多卡训练；安装 `lora` extra 后，使用 `lora=True` 即可进行 [LoRA](/docs/train/lora) 适配器微调。RT-DETRv2 的旋转框模型仅支持推理。关于数据集、数据增强和日志记录器，请参阅[训练文档](/docs/train)。

## 验证和导出

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` 返回普通字典。导出目标包括 ONNX、TorchScript、ExecuTorch、TensorRT 和 OpenVINO；[模型页](/docs/models/rt-detr)上的导出矩阵列出了每种任务已验证支持的格式。导出的 `.onnx` 或 `.engine` 文件可以通过 `LibreYOLO()` 重新加载，并返回相同的 `Results`。各格式的指南请参阅[导出文档](/docs/export)。

## 哪些情况下仍应使用原始仓库

如果你需要使用作者的精确配置复现论文结果、需要 Paddle 版本，或想自行运行 RT-DETRv4 的 DINOv3 教师模型蒸馏，请使用上游仓库。LibreYOLO 会微调已发布的 v4 学生模型，不会运行教师模型。如果你的项目已经采用 AGPL-3.0，或已获得 Ultralytics 企业许可，则没有许可方面的理由需要迁移。

## 关于许可证的说明

LibreYOLO 的代码采用 MIT 许可。每个 RT-DETR 权重文件都采用 Apache-2.0；每个 Hugging Face 权重仓库都附带上游 LICENSE 和 NOTICE，Apache-2.0 要求你在再分发权重时保留这些文件。你在自己的数据上训练得到的权重归你所有。RT-DETRv4 只在训练期间将 DINOv3 用作教师模型，发布的学生模型不含 DINOv3 参数。以上是一般性信息，不是法律意见。发布前请检查当前的 LICENSE 文件。

## 试试看

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO 采用 MIT 许可，可运行于 Linux、Mac 和 Windows，并且无需更改代码即可在 GPU、Apple Silicon 和普通 CPU 上运行。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [RT-DETR 文档](/docs/models/rt-detr)
