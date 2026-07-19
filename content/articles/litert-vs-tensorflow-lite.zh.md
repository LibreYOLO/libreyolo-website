---
title: "LiteRT 与 TensorFlow Lite：只是改名，不是新格式"
description: "Google 在 2024 年 9 月将 TensorFlow Lite 更名为 LiteRT。运行时相同，.tflite 文件相同，什么都不会坏。本文讲清楚改名的原因、实际变化，以及如何用 LibreYOLO 导出模型。"
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "TensorFlow Lite 被弃用了吗？"
    a: "被淘汰的是名字，不是技术。运行时以 LiteRT 的名义继续存在，API 相同，现有的 TFLite 模型和代码继续可用。新功能（如 CompiledModel API 和 NPU 加速）以 LiteRT 的名义发布。"
  - q: "我需要为 LiteRT 重新导出模型吗？"
    a: "不需要。为 TensorFlow Lite 导出的 .tflite 文件就是 LiteRT 模型。文件本身没有任何变化，因此无需重新导出或迁移。"
---

**LiteRT 就是改了名字的 TensorFlow Lite。它不是新的运行时，也不是新的文件格式。模型仍然是 `.tflite` 文件，之前能在 TensorFlow Lite 上运行的每个 `.tflite` 文件都能在 LiteRT 上原样运行。**

如果你搜索的是"LiteRT vs TensorFlow Lite"、"TensorFlow Lite 是否被弃用"或"什么是 LiteRT"，上面那段话就是答案。本页剩下的部分简要说明改名的来龙去脉，以及如何用 LibreYOLO 导出模型。

## Google 为什么改名

TensorFlow Lite 于 2017 年发布，最初是在手机和嵌入式设备上运行 TensorFlow 模型的方案。这些年它的范围远远超出了这个定位：Google 构建了从 PyTorch、JAX 和 Keras 的转换路径，到 2024 年，运行在它上面的模型中有很大一部分从未接触过 TensorFlow。

到那时，这个名字已经在造成误导。PyTorch 用户会跳过它，因为听起来像一个只服务 TensorFlow 的工具。于是 2024 年 9 月，Google [将其更名为 LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/)，即"Lite Runtime"的缩写。故事就这么简单：同一个团队，同一套代码，一个框架中立的名字。

## 实际变化有哪些

非常少，而且没有任何破坏性变化：

* 代码搬到了新家：[github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert)。
* 文档搬到了 [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)。
* 运行模型的 Python 包现在是 `ai-edge-litert`。旧的 `tflite-runtime` 包已停滞；新包提供相同的 `Interpreter` 类。
* 改名之后的新功能以 LiteRT 的名义发布：更简洁的 `CompiledModel` API，以及 Google 于 [2026 年 1 月宣布生产可用](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/)的 GPU/NPU 加速。

经典的 Interpreter API 也原样可用。确实有一个新扩展名 `.litertlm`，但它只是端侧 LLM 的打包格式，对计算机视觉模型来说你可以当它不存在。所以当一份文档写"TFLite"、另一份写"LiteRT"时，它们说的是同一种文件。

## 用 LibreYOLO 导出模型到 LiteRT

LibreYOLO 在 v1.3.0 中加入了 TFLite/LiteRT 导出路径。需要 Python 3.12+ 和一个可选依赖：

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # 首次运行自动下载
model.export(format="tflite")        # 生成一个 .tflite 文件
```

或者用 CLI：

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

目前验证过的路径是 YOLO9 检测和 RF-DETR 检测、分割、姿态，均仍标记为实验性。完整的支持矩阵见[文档](/docs/v1.3.0)。

在目标设备上运行导出的文件，使用 Google 的 `ai-edge-litert` 包：

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC，这里放你预处理后的图像
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

注意输入是 NHWC（通道在最后）：从 ONNX 转换时会转置布局，这对该格式来说是正常的。

## 我们在 LibreYOLO 中改了什么

改动不多，因为本来就没什么需要改。文档现在将该格式标注为"TFLite (LiteRT)"，两个名字都能搜到；API 中的格式名保持 `format="tflite"`。

## 常见问题

**TensorFlow Lite 被弃用了吗？**
被淘汰的是名字，不是技术。运行时以 LiteRT 的名义继续存在，API 相同，现有的 TFLite 模型和代码继续可用。新功能（如 CompiledModel API 和 NPU 加速）以 LiteRT 的名义发布。

**我需要为 LiteRT 重新导出模型吗？**
不需要。为 TensorFlow Lite 导出的 `.tflite` 文件就是 LiteRT 模型。文件本身没有任何变化，因此无需重新导出或迁移。
