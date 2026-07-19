---
title: 如何用 LibreYOLO 运行 YOLO-NAS
description: YOLO-NAS 是目前精度最高的实时检测器之一，但原始仓库已无人维护。下面教你如何用 LibreYOLO 运行它。
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, yolo-nas, object-detection, tutorial]
---

在所有实时检测器中，YOLO-NAS 拥有数一数二的精度-速度表现：large 变体在 COCO 上达到 52.2 mAP，同时仍能实时运行。它有 3 种尺寸变体（S、M、L），以及两种任务变体（检测、姿态）。

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS 精度与参数量对比 - visionanalysis.org">
</iframe>

想从官方仓库（supergradients）把 YOLO-NAS 跑起来，需要花费比预期更多的功夫：

* 安装 super-gradients，它会连带拉入 hydra、omegaconf、boto3、tensorboard，以及一个被锁定的 `torchmetrics==0.8`，后者会与较新的技术栈发生冲突，

* 为了加载一个模型并跑一次预测，要从三四个不同的子模块里分别导入，

* 还要通过 `ImagesDetectionPrediction` 把输出层层拆开，才能拿到检测框和分数。

官方仓库的另一个问题是：自从创建 supergradients 的公司（DeciAI）被英伟达收购以来，它就完全无人维护了。这意味着会出现一些典型问题，比如缺乏支持，以及与较新技术栈的冲突。 \
\
LibreYOLO 是 supergradients 的一个可靠替代品：它有人维护，并且通过你对其他每个模型都在用的那套简单易用的 API，加载同样的 YOLO-NAS 权重：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # 首次运行时自动下载
results = model("image.jpg", save=True)
```

S、M、L 变体在检测和姿态任务上都可用。

LibreYOLO 可以把 YOLO-NAS 导出为 ONNX、TorchScript、OpenVINO、NCNN 和 TFLite，这比原始库提供了更好的支持。

## 什么情况下 super-gradients 仍是更好的选择

**CoreML 导出**：LibreYOLO 不支持把 YOLO-NAS 导出为 CoreML。它的 CoreML 导出器仅限于另一组模型家族。如果你要在 Apple 设备上发布产品并需要 `.mlpackage`，请继续使用 super-gradients。

**TensorRT**：super-gradients 为 YOLO-NAS 提供了有文档、经过测试的 TensorRT 路径，并明确说明了已知的 batch-size 相关问题。而 LibreYOLO 对 YOLO-NAS 的 TensorRT 支持尚未经过测试。

## 关于权重的说明

这些权重属于 Deci，而非 LibreYOLO。无论你用哪个库来加载它们，适用的都是同一份非商用许可证。LibreYOLO 只是链接到 Deci 的 CDN，并不会重新托管任何内容。用于研究和非商业推理没有问题。但若要用于商业产品，无论如何你都需要去和 Deci 沟通。

LibreYOLO 的代码采用 MIT 许可。权重上的限制来自上游，并不是 LibreYOLO 附加的。我们计划在未来从零训练 YOLO-NAS 模型。

## 上手试试

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)   # xyxy 坐标
print(results[0].boxes.conf)   # 置信度分数
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且无需改动代码即可覆盖 GPU、Apple Silicon 和普通 CPU。一套 API 覆盖 RF-DETR、D-FINE、DEIM、YOLO-NAS、分割、姿态、深度等更多任务。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
