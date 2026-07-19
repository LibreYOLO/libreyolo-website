---
title: LibreYOLO showed up at CVPR 2026
description: At CVPR 2026 in Denver, a Jabra and IT University of Copenhagen team used LibreYOLOXs as the example model in their "Edge AI in Action" tutorial, running it on a Hailo-8L and on Snapdragon.
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
---

![CVPR 2026, Denver, Colorado, June 3-7](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO showed up at CVPR 2026.

CVPR is widely recognized as the most prestigious computer vision conference in the world. This year it was in Denver, and a team from Jabra and the IT University of Copenhagen ran a tutorial there called "Edge AI in Action: Mastering On-Device Inference". They picked LibreYOLOXs as their example model for getting object detection onto edge chips.

## What they built

The tutorial walked the full edge pipeline, end to end. Start with a LibreYOLOXs model. Export it to ONNX. Then compile that ONNX down to two very different pieces of hardware: a Hailo-8L accelerator, and Qualcomm Snapdragon.

## Real-time on a Hailo-8L

They compiled the ONNX to a HEF with the Hailo Dataflow Compiler, then ran it on a Raspberry Pi 5 with the Hailo-8L AI HAT.

![LibreYOLOXs running on a Raspberry Pi 5 with a Hailo-8L, detecting people and handbags on a busy street](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

It ran at 19.0 ms per frame, 52.6 FPS, on a Raspberry Pi.

## Real-time on Snapdragon

On the Qualcomm side they quantized LibreYOLOXs to INT8 and ran it through the SNPE, QAIRT, and AI Hub stack, benchmarked on a QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs detecting a TV, laptop, keyboard, mouse and cell phone live in the EdgeVision AI app on a Snapdragon phone" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

It came in at 6.69 ms on the HTP/DSP.

## Thank you

Huge thanks to Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera, and Shan Ahmed Shaffi for featuring the project.

- Tutorial and slides: [Edge AI in Action: Mastering On-Device Inference](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Qualcomm-quantized LibreYOLOXs models: [on Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Try it

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO is MIT-licensed, runs on Linux, Mac, and Windows, and works on GPU, Apple Silicon, and plain CPU with no code change. One API spans YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentation, pose, depth, and more.

Star it on GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Docs: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
