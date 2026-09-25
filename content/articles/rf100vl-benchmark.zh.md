---
title: "RF100-VL 基准测试：YOLOv9、YOLOX、YOLO-NAS、EdgeCrafter 和 RF-DETR 跨数据集泛化能力"
description: RF100-VL 结果衡量 YOLOv9、YOLOX、YOLO-NAS、EdgeCrafter 和 RF-DETR 在 100 个真实数据集上微调后的泛化能力。
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL 要回答的是检测器在 COCO 之外的数据上适应得如何。我们分别在 100 个差异很大的数据集上微调每个模型，其中包括红外、X 射线、显微镜、体育、无线电频谱图像、微小物体和电子游戏数据集。我们对 17 种 LibreYOLO 配置进行了基准测试：总计 **1700 次微调**。

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## 结果

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L 在这次比较中以 **61.76** 领先。M 以 61.13 紧随其后，S 为 60.41。四项 RF-DETR 结果与 [Roboflow 发布的数据](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 接近，这为 LibreYOLO 中的 RF-DETR 训练提供了有用的验证。

模型尺寸并不能预测所有结果。YOLO-NAS-S 和 M 的成绩几乎持平，分别为 58.00 和 57.99。EdgeCrafter-M 得分为 57.93，高于 S 的 55.99 和 L 的 56.11。我们计划调查这些回退现象。这轮测试不是受控的缩放实验：分辨率、预训练权重、精度和训练方案各不相同。

使用骨干 LoRA 的 RF-DETR-S 得分为 57.19，而全量微调为 60.41。全量微调在 95 个数据集上胜出。记录的中位耗时只多 7 分钟，而作业总耗时几乎没有变化。

YOLOv9 的结果是在几项重要训练修复之前得到的。异常的 M 结果帮助我们发现了 PGI 缺失、不同的 letterbox 约定以及 100 个标签的训练上限。归档的数字反映了当时实际运行的代码，不能据此评判 YOLOv9 架构。

选择下方的模型，查看它在全部 100 个数据集上的得分。

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## 方法

对于每个数据集，我们都在 `train` 上训练，选择验证集 AP50:95 最佳的检查点，并在 `test` 上评估一次。总分是这 100 个测试结果的非加权平均值。

| 设置 | 测试规则 |
|---|---|
| 训练 | 100 轮；不提前停止 |
| 有效批大小 | 16 |
| 随机种子 | 0；每个数据集完成一次运行 |
| 检查点 | 使用 EMA 权重的最佳验证集 AP50:95 |
| 测试评分 | pycocotools，`maxDets=500` |
| 调参 | 每个家族固定一套训练方案；不按数据集调参 |

每个家族都使用自己的训练方案：

| 家族 | 分辨率 | 精度 | 增强概述 |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic、HSV、翻转；最后 15 轮关闭强增强 |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic、mixup、HSV、仿射变换、翻转；最后 15 轮为收尾阶段 |
| YOLOX S/M | 640 | FP32 | 使用同一家族的训练方案，并按尺寸设置学习率 |
| YOLO-NAS S/M | 640 | FP32 | Mixup、HSV 和翻转；关闭 mosaic |
| EdgeCrafter S/M/L | 640 | FP16 | 翻转；LibreYOLO 家族默认设置 |
| RF-DETR N/S/M/L 和 S LoRA | 384/512/576/704；LoRA 使用 512 | BF16 | 多尺度、裁剪/调整尺寸和翻转 |

这些结果都只用了一个随机种子。微小差异不能证明有所提升。RF-DETR M 和 L 使用梯度累积以适应可用内存，一些密集数据集则需要较小的实际批大小。公开的 RF-DETR 训练方案也从 COCO 检查点开始，而 Roboflow 历史表格使用的是私有 Objects365 检查点。

训练耗时是测试活动的记录，不是受控的速度基准。作业有时会共用 GPU、重试或恢复运行。我们没有执行可选的 T4 单制品延迟测试流程。YOLO-NAS 使用 Deci 单独许可的[非商用预训练权重](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md)。

## 工作负载带来的改进

小规模测试可以证明训练能够启动，但无法复现跨许多架构和数据集持续数周的微调。在这样的规模下，缓慢路径、内存压力和正确性问题都很难被忽略。

- **图像加载和验证。** 我们在数据增强前缓存确定性缩放结果，复用验证器工作线程和缓冲区，并在支持的情况下绘制验证前向传播图。[PR #677](https://github.com/LibreYOLO/libreyolo/pull/677)、[PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **训练 CUDA graph。** 捕获支持从 YOLOv9 和 RF-DETR 扩展到了 24 个家族。我们还修复了 DataLoader 的 pin-memory 竞争问题，以及训练阶段转换时的 graph 失效问题。在一项 YOLOv9-T 测试中，耗时从 428.4 秒降至 367.7 秒，报告的 AP 相同。[PR #671](https://github.com/LibreYOLO/libreyolo/pull/671)、[PR #681](https://github.com/LibreYOLO/libreyolo/pull/681)、[PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **COCO 评分。** 在密集数据集上，评估有时比训练耗时更长。集成 faster-coco-eval 后，对全部 100 个测试划分中已保存的预测评分只需 8.4 秒，而原来需要 131.4 秒。在 1,400 个指标值中，1,381 个逐位一致；最大差异为 2.22e-16，汇总 AP 不变。[PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR 和共享训练路径。** 更快的 L1 匹配、更少的设备同步、融合 AdamW 和有界的匹配器内存，将一次记录的 RF-DETR-S 测试步耗时从 266 毫秒降至 234 毫秒。同一轮调查还改进了另外五个匹配器、YOLOv9 日志记录和 EdgeCrafter 张量复用。[PR #761](https://github.com/LibreYOLO/libreyolo/pull/761)、[PR #762](https://github.com/LibreYOLO/libreyolo/pull/762)、[PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention。** 我们将符合条件的 attention 路由到 PyTorch SDPA，集成了一种 Apache-2.0 CUDA 实现，并修复了混合精度执行问题。我们还在代码库中为推理编写了 Triton deformable-attention kernel。在文档记录的测试中，它单独运行时约快 4 倍，RF-DETR-N 端到端运行则约快 7%。[PR #712](https://github.com/LibreYOLO/libreyolo/pull/712)、[PR #713](https://github.com/LibreYOLO/libreyolo/pull/713)、[PR #760](https://github.com/LibreYOLO/libreyolo/pull/760)、[PR #784](https://github.com/LibreYOLO/libreyolo/pull/784)、[PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **训练正确性。** 我们修复了 YOLOX BatchNorm 重建问题，并恢复了 YOLOv9 PGI、letterbox 元数据、标签容量和动量预热。基准测试提供的检查点和故障模式揭示了这两个问题。[PR #700](https://github.com/LibreYOLO/libreyolo/pull/700)、[PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

这些数字来自不同工作负载下的独立测试，不能相乘后合并成一个总体加速比。与这轮测试活动开始前相比，LibreYOLO 现在处理真实训练工作负载的速度更快、可靠性也更高。

## 测试工具和产物

公开的[训练测试工具](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness)会跨 GPU 调度数据集、恢复中断的作业、处理 OOM 恢复，并记录训练方案、数据集版本、日志、检查点和预测结果。

[run-rf100vl-benchmark skill](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) 为 AI 编码代理提供测试流程和 Vast.ai 操作说明。[结果归档](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main)包含所有已发布的运行结果。

## 感谢 Roboflow

我写下自己想在 COCO 之外进行基准测试、但无力承担这些运行费用后，Joseph Nelson 提供了 GPU 支持。Matvei Popov 分享了参考设置，解释了评估设置，并持续关注后续结果。

这份支持让我们得以验证 17 种配置下的 LibreYOLO 训练、发布人们选模型时可以参考的证据、开放测试工具，并对库进行压力测试，直到其薄弱环节清晰显现。

感谢 Joseph、Matvei 和 Roboflow 团队提供算力、时间和指导。
