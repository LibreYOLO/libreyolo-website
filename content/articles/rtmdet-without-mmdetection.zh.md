---
title: 如何在不使用 mmdetection 的情况下运行 RTMDet
description: RTMDet 是目前速度最快、精度也很高的检测器之一。但它的官方安装方式在过去两年发布的任何 PyTorch 上都无法正常工作。这里提供另一种方案。
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "为什么在较新的 PyTorch 上安装 mmcv 会失败？"
    a: "最新的 mmcv wheel 是 2024 年 4 月发布的 2.2.0，预编译二进制只支持到 torch 2.4 / CUDA 12.1。在更新的 PyTorch 上，mmcv 必须从源码编译 C++/CUDA 算子，耗时 10 到 30 分钟，还需要匹配的 CUDA 工具链、nvcc 和兼容的 C++ 编译器。诸如找不到 mmcv._ext 模块之类的报错正是由此而来。"
  - q: "可以不安装 mmdetection 就运行 RTMDet 吗？"
    a: "可以。LibreYOLO 加载的 RTMDet 权重由上游 OpenMMLab 检查点转换而来，在同一检查点上推理结果与 mmdetection 逐位一致。按名称加载 LibreRTMDets.pt 即可自动下载；五个尺寸（Tiny 到 X）均以 640 px 运行。"
  - q: "RTMDet 可以免费商用吗？"
    a: "可以。MMDetection 和 RTMDet 均为 Apache 2.0，转换后的权重沿用同样的 Apache 2.0 条款，LibreYOLO 的代码是 MIT 许可。没有商业限制。"
  - q: "什么情况下仍应选择 mmdetection？"
    a: "如果你需要用完整的 MMDetection 数据增强管线训练或微调 RTMDet，或者你已经深入 OpenMMLab 生态且 mmcv 环境正常，就留在原生态。至于推理和部署，LibreYOLO 是更好的选择。"
---

RTMDet 是 OpenMMLab 推出的实时检测器，它在 COCO 上取得了很高的精度，同时保持着足以用于部署的速度。它的架构很简洁，安装过程却并不简洁。

想从官方源码把 RTMDet 跑起来，意味着要拼装一套四层的 OpenMMLab 技术栈：

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

版本约束非常苛刻。mmdet 3.3.0 要求 `mmcv < 2.2.0`，但 `mim install mmcv>=2.0.0` 却会顺手装上 2.2.0，结果在运行时触发断言失败。你必须手动指定版本。

接下来是更大的问题。最新的 mmcv wheel 是 2.2.0，发布于 2024 年 4 月，此后再未更新。它只提供截至 **torch 2.4 / CUDA 12.1** 的预编译二进制包。而今天全新执行 `pip install torch` 会给你装上 PyTorch 2.12。这个版本差意味着 mmcv 必须从源码编译它的 C++/CUDA 算子：耗时 10 到 30 分钟，还需要配套的 CUDA 工具链、`nvcc` 以及兼容的 C++ 编译器。各种有据可查的报错就堆积在这一步：

* `ModuleNotFoundError: No module named 'mmcv._ext'`：编译出来的算子没有构建成功，或者版本不匹配，

* 在任何 RTX 30 系列显卡上出现 `nvcc fatal: Unsupported gpu architecture 'compute_86'`，

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`：这套技术栈逼着你退回到 Python 3.8，

* 由于 GCC 版本不匹配，导入时出现段错误（segmentation fault）。

OpenMMLab 自己的 FAQ 告诉你用 pip 而不是 mim 来安装 mmcv，以避开版本陷阱。而他们的「Get Started」页面又告诉你要用 mim。两份文档都是官方的，却彼此矛盾。

即便技术栈终于跑起来了，做一次推理仍然需要：挑选一个文件名里编码着训练配方的 config 文件，再加上一个单独下载、文件名带哈希戳的 checkpoint，并通过一个你得先学会的注册表（registry）把它们串接起来。

LibreYOLO 把这一切都替换掉了：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # 首次运行时自动下载
results = model("image.jpg", save=True)
```

不需要 `mim`，不需要四个包的版本矩阵，不需要从源码编译，不需要 config 文件，不需要满世界找带哈希的 checkpoint，也不需要把 Python 锁死在 3.8。这些权重是从上游 OpenMMLab 的 checkpoint 转换而来的，在相同 checkpoint 下，推理结果与 mmdetection 逐比特一致。

共有五种尺寸可选：Tiny、Small、Medium、Large 和 X。它们全部以 640 px 运行。

```python
print(results[0].boxes.xyxy)  # xyxy 坐标
print(results[0].boxes.conf)  # 置信度分数
```

## 什么情况下原版仍是更好的选择

如果你需要借助完整的 MMDetection 数据增强流水线来训练或微调 RTMDet，或者你已经深度使用 OpenMMLab 生态、并且 mmcv 已经正常工作，那就继续留在那边。而在推理和部署方面，LibreYOLO 是更好的选择。

## 关于许可证的说明

MMDetection 和 RTMDet 采用 Apache 2.0 许可。LibreYOLO 的代码采用 MIT 许可。权重沿用上游相同的 Apache 2.0 条款。没有商用限制。

## 上手试试

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且无需改动代码即可在 GPU、Apple Silicon 和普通 CPU 上工作。一套 API 覆盖 RTMDet、RT-DETR、RF-DETR、D-FINE、YOLOX、YOLO-NAS、分割、姿态、深度等更多任务。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
