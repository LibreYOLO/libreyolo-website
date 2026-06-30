---
title: LibreYOLO 亮相 CVPR 2026
description: 在丹佛举办的 CVPR 2026 上，来自 Jabra 和哥本哈根信息技术大学的团队在他们的「Edge AI in Action」教程中使用 LibreYOLOXs 作为示例模型，并将它运行在 Hailo-8L 和 Snapdragon 上。
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
---

![CVPR 2026，美国科罗拉多州丹佛，6 月 3-7 日](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO 亮相了 CVPR 2026。

CVPR 被公认为全世界最具声望的计算机视觉会议。今年它在丹佛举办，来自 Jabra 和哥本哈根信息技术大学（IT University of Copenhagen）的一支团队在会上做了一场名为「Edge AI in Action: Mastering On-Device Inference」的教程。他们选用 LibreYOLOXs 作为示例模型，演示如何把目标检测部署到边缘芯片上。

## 他们做了什么

这场教程完整走通了从头到尾的边缘部署流水线。先拿到一个 LibreYOLOXs 模型。把它导出为 ONNX。然后把这个 ONNX 编译到两种截然不同的硬件上：一块 Hailo-8L 加速器，以及高通骁龙（Qualcomm Snapdragon）。

## 在 Hailo-8L 上实时运行

他们用 Hailo Dataflow Compiler 把 ONNX 编译成 HEF，然后在一块装了 Hailo-8L AI HAT 的树莓派 5 上运行。

![LibreYOLOXs 在装有 Hailo-8L 的树莓派 5 上运行，在繁忙的街道上检测行人和手提包](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

它跑出了每帧 19.0 毫秒、52.6 FPS 的成绩，而且只是在一块树莓派上。

## 在 Snapdragon 上实时运行

在高通这边，他们把 LibreYOLOXs 量化为 INT8，并通过 SNPE、QAIRT 和 AI Hub 这套技术栈运行，在 QCS6490 上做了基准测试。

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs 在 Snapdragon 手机的 EdgeVision AI 应用中实时检测电视、笔记本电脑、键盘、鼠标和手机" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

它在 HTP/DSP 上跑到了 6.69 毫秒。

## 致谢

非常感谢 Sai Narsi Reddy Donthi Reddy、Fabricio Batista Narcizo、Elizabete Munzlingera 和 Shan Ahmed Shaffi 对本项目的采用与展示。

- 教程与幻灯片：[Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- 高通量化版 LibreYOLOXs 模型：[在 Hugging Face 上](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## 上手试试

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # 然后为你的边缘目标硬件编译
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且无需改动代码即可在 GPU、Apple Silicon 和普通 CPU 上工作。一套 API 覆盖 YOLOX、RF-DETR、D-FINE、DEIM、YOLO-NAS、分割、姿态、深度等更多任务。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://libreyolo.com/docs)
