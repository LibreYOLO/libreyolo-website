---
title: 如何用 LibreYOLO 运行 YOLOX
description: YOLOX 采用 Apache 2.0 许可，可放心用于商业用途，且至今仍有竞争力。原始仓库已被弃置。下面教你改用 LibreYOLO 来运行它。
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, yolox, object-detection, tutorial]
---

YOLOX 是旷视（Megvii）推出的无锚框（anchor-free）检测器，共有六种模型尺寸，参数量从 0.91M（Nano）到 99.1M（X）不等。large 变体在 COCO val 上达到 49.7 mAP。它的代码和权重都采用 Apache 2.0 许可，这使它成为少数几个可以毫无限制地用于商业产品的、有竞争力的检测器之一。

<div style="position:relative;width:100%;padding-top:62.5%">
  <iframe
    src="https://www.visionanalysis.org/embed/scatter?highlight=yolox-nano%2Cyolox-tiny%2Cyolox-s%2Cyolox-m%2Cyolox-l%2Cyolox-x&title=YOLOX%3A%20accuracy%20vs.%20model%20size"
    style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;border-radius:12px"
    loading="lazy"
    title="YOLOX 精度与参数量对比 - visionanalysis.org">
  </iframe>
</div>

问题出在原始库上。`pip install yolox` 给你装的是 2022 年 4 月的 0.3.0 版本，此后再没有更新过。做一次推理可不是一行代码的事：

* 你要实例化一个 `Exp` 对象来配置模型，

* 手动把 checkpoint 加载进网络结构，

* 通过 `ValTransform` 对图像做预处理，

* 把它送入模型，再调用 `postprocess()`，才能从原始输出张量中得到检测框和分数。

它的 ONNX 导出用的是 `torch.onnx._export`，这是一个在 PyTorch 2.x 中已被移除的私有 PyTorch API。`requirements.txt` 里被死锁的 `onnx-simplifier==0.4.10` 会与任何较新的 `onnx` 安装发生冲突。CoreML 导出则根本不存在。这个仓库自 2022 年年中以来就一直处于「仅维护」状态。

LibreYOLO 通过你早已熟悉的同一套 API，为你提供同样的 YOLOX 权重：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")  # 首次运行时自动下载
results = model("image.jpg", save=True)
```

全部六种尺寸都可用：Nano（`n`）、Tiny（`t`）、Small（`s`）、Medium（`m`）、Large（`l`）、Extra-large（`x`）。Nano 和 Tiny 以 416 px 输入运行，其余的以 640 px 运行。LibreYOLO 会自动处理这一点。

```python
print(results[0].boxes.xyxy)  # xyxy 坐标
print(results[0].boxes.conf)  # 置信度分数
```

它的导出范围比原始库更广。LibreYOLO 可以把 YOLOX 导出为 ONNX、TorchScript、CoreML、OpenVINO、NCNN 和 TensorRT。原始仓库没有 CoreML 路径，而且它的 ONNX 导出在现代 PyTorch 上已经无法工作。

训练同样可用：

```python
model.train(data="your_dataset.yaml", epochs=100, batch=16)
```

## 关于许可证的说明

YOLOX 的权重采用 Apache 2.0 许可。这意味着没有非商用限制，不需要联系任何人，也没有平台锁定。

LibreYOLO 自身的代码采用 MIT 许可。

## 上手试试

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且无需改动代码即可在 GPU、Apple Silicon 和普通 CPU 上工作。一套 API 覆盖 RF-DETR、D-FINE、DEIM、YOLO-NAS、YOLOX、分割、姿态、深度等更多任务。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://libreyolo.com/docs)
