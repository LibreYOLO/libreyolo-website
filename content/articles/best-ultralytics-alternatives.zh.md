---
title: 2026 年最佳 Ultralytics 替代方案
description: "一份务实的指南，梳理 2026 年 Ultralytics YOLO 的最佳开源替代方案：它们的许可、任务覆盖与部署限制，以及最完整的 MIT 许可 YOLO 库 LibreYOLO 在其中的位置。"
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLO 可以免费商用吗？"
    a: "仅在 AGPL-3.0 下可以，而 AGPL-3.0 要求你公开基于它构建的整个应用的源代码，包括通过网络提供服务的情形。闭源商用需要购买 Ultralytics 的商业许可。采用宽松许可（MIT、Apache-2.0）的替代方案则没有这个问题。"
  - q: "哪个是许可最干净的 Ultralytics 替代方案？"
    a: "如果你需要一套可以随处部署的宽松许可栈：LibreYOLO（代码为 MIT）、RF-DETR（Apache-2.0）和 YOLOX（Apache-2.0）。它们都避开了 AGPL 的传染性。"
  - q: "2026 年什么在取代 YOLO？"
    a: "越来越多是实时 transformer 检测器：RT-DETR、RF-DETR，以及 D-FINE 和 DEIM 系列。在算力足够的硬件上，它们在相近延迟下达到或超过 YOLO 的精度。而在小型边缘设备上，YOLO 这类 CNN 仍占优势，因为那些 transformer 在上面运行很吃力，也缺少成熟的 NCNN 或 CPU 路径。"
  - q: "这些能在树莓派或 NPU 上运行吗？"
    a: "取决于导出能力。像 YOLOX 和 RTMDet 这类 CNN 检测器可以导出为 NCNN 并在树莓派上运行，其中一些（YOLOX）还能上 Hailo NPU。而像 RF-DETR 这样的 transformer 检测器不行，它们需要 GPU 或性能强劲的 CPU。"
---

声明：LibreYOLO 是我们自己的项目，因此它排在这份榜单的第一位。榜单里其他每一个工具都是我们在实际工作中真正使用的，我们也会指出它们在哪些方面胜过 LibreYOLO。

大多数团队离开 Ultralytics，通常是出于以下三个原因之一：

- **许可。** Ultralytics 的 YOLO 模型采用 AGPL-3.0（YOLOv5 自 2023 年起，以及 v8 之后的版本）。套路很熟悉：你做了一个产品，把它发布出去，然后一次法务审查或一位客户对模型提出质疑，于是选择就变成了：要么开源你的整个应用，要么购买一份商业许可。只要你分发或对外提供服务，AGPL 的传染性就会触及你的代码。这是人们另寻他路最常见的原因。
- **开放性。** 模型、训练代码和条款都由同一家厂商掌握。有些团队希望权重和代码采用宽松许可，可以在其上自由构建而无需征得任何人同意。
- **模型本身。** YOLO 并不总是某个任务的最佳架构。像 RT-DETR、RF-DETR 和 D-FINE 这样的实时 transformer 检测器，可以在相同延迟下更精确。问题在于它们散落在各个研究仓库里，而正如你接下来会看到的，LibreYOLO 用一套 API 就能运行这三者，所以这是一个换库的理由，而不是彻底离开的理由。

下面的每一个工具都从三点来评判：一份你能放心发布的许可、能否在当前版本的 PyTorch 上安装并运行，以及能否导出到你真正要部署的硬件上。

读下去你会发现一个反复出现的规律：每个替代方案本身都不错，但单独用起来都很痛苦。YOLOX 几乎装不上，MMDetection 是版本锁死的炼狱，RT-DETR 家族是没有支持渠道的研究代码，Detectron2 自 2021 年起就已冻结。LibreYOLO 排在榜首，是因为它已经把它们中的大多数封装了起来：持续维护、MIT 许可、跑在当前技术栈上、藏在一套你熟悉的 API 之后。与其说它是这份榜单上的一项，不如说它是这份榜单之上的一层。

## 速览

| 工具 | 许可 | 覆盖任务 | 最适合 | 边缘导出 |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT（代码） | 检测、分割、姿态、分类、深度、视线、跟踪 | 用一套 API 承载 20 多个模型家族的 MIT 中枢 | ONNX、TensorRT、OpenVINO、NCNN、CoreML、TFLite |
| **RF-DETR** | Apache-2.0（N/S/M/L） | 检测、分割、姿态（预览） | 生产级检测与分割 | ONNX、TFLite、TensorRT；无 NCNN 或 Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | 预训练、蒸馏 | 无标注数据，蒸馏 DINOv2/v3 | 不适用（非检测器） |
| **YOLOX** | Apache-2.0 | 检测 | 无锚框实时检测 | 上游安装困难，可经 LibreYOLO 运行 |
| **MMDetection / OneDL 分支** | Apache-2.0 | 全部（研究向） | 架构广度，复现论文 | 通过导出 |
| **Detectron2** | Apache-2.0 | 检测、分割、关键点 | Mask R-CNN 与 R-CNN 家族 | 手动 |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | 检测 | 最先进的实时 DETR | ONNX、TensorRT |
| **EdgeCrafter** | Apache-2.0 | 检测、分割、姿态 | 从 DINOv3 蒸馏的紧凑边缘 ViT | 研究级 |
| **RT-DETR（v1-v4）** | Apache-2.0 | 检测 | 前沿实时检测 | ONNX、TensorRT |

## 1. LibreYOLO

许可：MIT（代码）。最适合：最完整的 MIT 许可 YOLO 库，以及一个可直接替换 Ultralytics 的方案。

**LibreYOLO 是 MIT 许可的 YOLO 库，用一套 API 运行每一个模型。** 它是我们自己的项目，而在看完这份榜单的其余部分之后，它的定位应该已经很清楚了：它就是那个运行这些替代方案的中枢。上面几乎每一个令人头疼的仓库，YOLOX、RTMDet、RF-DETR、D-FINE、DEIM、RT-DETR 系列，LibreYOLO 都已经封装在一套熟悉且持续维护的 API 之后，让你拿到模型，却不必经历安装考古，也不必背上那份许可。

这里还有一个超越便利性的理由。YOLO 起初是开放的研究成果：Joseph Redmon 的 Darknet，从 v1 到 v3，任何人都可以自由使用和在其上构建。AGPL 时代把它圈了起来。LibreYOLO 的存在，就是要让这份工作重新变得触手可及，回到它的创造者最初所期望的样子：宽松许可、社区维护，且不会偷偷回传数据。于是，两个故事。

**其一：它是 Ultralytics YOLO 的 MIT 许可替代方案。** 它保留了你已经熟悉的那套 API，所以你的 YOLO 格式数据集和脚本只需小幅改动即可迁移过来，但它是 MIT 而不是 AGPL：没有传染到你代码里的 copyleft，无需购买商业许可，也不会像 Ultralytics 默认那样偷偷回传遥测数据。如果你是为许可而来，这正是关键所在。YOLO9 和 RF-DETR 是经过大量测试、可用于生产的默认选择；更广的模型库则较新，并被明确标注为实验性，我们宁愿如实相告，也不愿假装它们全都久经考验。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # 或 LibreRFDETRl.pt、LibreDFINEl.pt 等
results = model("image.jpg", save=True)
```

**其二：它覆盖的范围远超 Ultralytics。** 一套 API 横跨 20 多个模型家族，CNN 版 YOLO、transformer 版 DETR、ViT 骨干、SAM 等等，而不是某一家厂商的产品线。而且它不只有检测器：

- **任务：** 实例分割与语义分割、姿态、分类和旋转框，还有两项 Ultralytics 根本没有的任务，单目深度（Depth Anything V2）和视线估计（L2CS）。
- **实用层：** 带持久 ID 的多目标跟踪（ByteTrack 和 OC-SORT）、支持抽帧和实时预览的视频推理、面向无人机和卫星影像的切片推理、一个拖拽式的浏览器 UI（`libreyolo ui`），以及一个在你浪费一次训练之前先检查数据集的 `doctor` 命令。
- **训练很扎实：** 梯度累积、层冻结、断点续训、测试时增强（TTA）、LoRA/DoRA 微调、多卡训练，以及 TensorBoard/MLflow/Weights & Biases 日志。
- **导出：** 七种格式（ONNX、TorchScript、TensorRT、OpenVINO、NCNN、CoreML、TFLite），支持 INT8/FP16 量化、内嵌 NMS 和内嵌模型元数据。
- **随处运行：** 接受路径、URL、PIL、NumPy、张量或原始字节，并且无需改动代码即可在 CUDA、Apple Silicon（MPS）或普通 CPU 上运行。

当某个强力检测器自己的仓库被弃置或难以安装时，LibreYOLO 会在当前技术栈上、以持续维护的方式运行它的权重。

## 2. RF-DETR

许可：Apache-2.0（Nano、Small、Medium、Large）。最适合：生产级检测与分割。

RF-DETR 来自 Roboflow，发表于 ICLR 2026，是这份榜单上最具生产可用性的模型。它工程做得扎实，出自一支真正在交付产品的团队，当你需要能在真实系统中放心使用、切实可用的检测或分割时，它是一个很强的首选。`rfdetr` 包以及 Nano 到 Large 的权重都是 Apache-2.0。更大的 XL 和 2XL 权重则使用 Roboflow 的 PML 许可。

它的局限在于范围和部署。它覆盖检测和分割，并有一个预览版的关键点/姿态模型，但没有分类。而且它触及不到受限硬件。官方导出是 ONNX 和 TFLite；TensorRT 是需要 CUDA GPU 的 ONNX 转换。它没有 NCNN 路径，而那通常是上树莓派的常规途径，也不支持 Hailo NPU。在树莓派 5 的 CPU 上，它大约每帧要一秒甚至更久，谈不上实时。Roboflow 自己给出的实时建议是：用 GPU、上 TensorRT、选最小的模型、降低分辨率。有 GPU 或 Jetson 级别的板子时这没问题，但对廉价的边缘 SoC 或 NPU 来说就成了难题。它公布的延迟数据是 NVIDIA T4 上的 TensorRT FP16，而不是 CPU 数据。

## 3. Lightly

最适合：无标注数据、自监督预训练，以及蒸馏一个 DINO 骨干。

Lightly 不是检测器。它是让你从尚未标注的数据中榨取价值的工具，由许可不同的两部分组成。

- **[`lightly`](https://github.com/lightly-ai/lightly)（LightlySSL），MIT。** 一个自监督学习组件框架：损失函数、各种 head、数据增强、memory bank，以及二十多种方法的参考实现（SimCLR、MoCo、BYOL、DINO、DINOv2、MAE 等）。训练循环由你来写；它把从零在无标注图像上预训练一个表征所需的一切都给你。
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train)，AGPL-3.0（提供商业版和免费社区版选项）。** 开箱即用的那一款，也是多数人所指的工作流。把它指向一个基础模型（如 DINOv2 或 DINOv3），再指向你的无标注数据，它就会把那个骨干蒸馏成一个更小、可部署的模型（YOLO、RT-DETR、ViT 或自定义网络），几行代码、无需标注。注意其许可：AGPL-3.0 正是那种把人们推向寻找 Ultralytics 替代方案的 copyleft，所以如果你打算发布闭源产品，请仔细阅读条款，或者购买商业许可。

当瓶颈在于「需要一个好骨干，但没有标注」时，就该考虑 Lightly。它与本榜单上的检测器是互补关系，而非竞争关系。榜单中最新的那些也印证了这一点：RT-DETRv4 把视觉基础模型蒸馏直接融入了训练，DEIMv2 则把 DINOv3 骨干直接内置进了检测器。

## 4. YOLOX

许可：Apache-2.0。最适合：无锚框实时检测，前提是你能把它装上。

上游仓库自最后一个版本 v0.3.0（2022 年 4 月）以来就已冻结，未解决的 issue 超过 700 个，而且它在 2026 年的技术栈上极难安装。它没有 `pyproject.toml`，`requirements.txt` 里锁死了 `onnx-simplifier==0.4.10`，而后者在 Python 3.10 之后没有预编译 wheel，所以在较新的解释器上它会转为源码编译，需要 cmake 和一套 C++ 工具链。NumPy 也没有被固定版本，这会招致与较旧的 pycocotools 和 torch 之间的 NumPy 2.0 ABI 冲突。核心检测在你理清依赖链之后是能跑起来的，但走到那一步本身就是代价。

模型本身依然不错：一个来自旷视（Megvii）的无锚框实时检测器，代码和权重都是 Apache-2.0，即便有更新的检测器已经超越了它，它仍是一个合理的基线。如今复活一个被弃置但设计良好的仓库，是编码 agent 一个下午就能搞定的活儿，所以安装状态并不像看上去那么是道墙。LibreYOLO 也可以直接在当前技术栈上运行 YOLOX 的权重。无论走哪条路，这个架构都值得一用。我们写了一篇更完整的教程，[点这里](/articles/yolox-with-libreyolo)。

## 5. MMDetection 生态

许可：大多为 Apache-2.0。最适合：架构广度与复现论文。

多年来，MMDetection 一直是几乎每篇检测论文的官方实现所在：Faster R-CNN、DINO、Grounding-DINO、RTMDet、Mask R-CNN，还有数百个配置文件。到了 2026 年，OpenMMLab 已经把 mm 系列逐步收尾。[MMDetection](https://github.com/open-mmlab/mmdetection) 最后一个版本是 2024 年 1 月的 v3.3.0，issue 无人回应，而且整个栈被 `mmcv` 的版本锁死，其预编译 wheel 落后于当前的 PyTorch 和 CUDA，于是一次全新安装往往会一路报错，直到你把所有东西都降级。我们在[这里](/articles/rtmdet-without-mmdetection)讲过这份痛苦。

一家荷兰公司 VBTI 让它活了下来。他们的 [OneDL 分支](https://github.com/VBTI-development/onedl-mmdetection)把整个栈以 `onedl-` 前缀重新发布（`onedl-mmdetection`、`onedl-mmcv`、`onedl-mmengine` 等），并针对当前的 PyTorch 2.x、CUDA 和 Python 3.10+ 重新构建，解决了版本锁死的问题。它是 Apache-2.0 且在积极维护，2026 年 5 月发布了 v3.5.1。想要 MMDetection 的广度又不想折腾安装，就用这个分支。它是个小团队，如果你依赖它，请回馈社区。

另外两个相关工具：

- **[Detectron2](https://github.com/facebookresearch/detectron2)**（Meta，Apache-2.0）处于维护模式，自 2021 年的 v0.6 起就没有打过新版本标签，但它可靠，而且仍是获取 Mask R-CNN 与 R-CNN 家族（用于实例分割和全景分割）最干净的来源。
- **[TorchVision](https://github.com/pytorch/vision)**（BSD-3-Clause，积极维护）提供 Faster R-CNN、RetinaNet、FCOS、SSD、Mask R-CNN 和 Keypoint R-CNN。没有现代 YOLO 或 DETR，训练循环也得你自己写，但它是零额外依赖的基线。

关于商用有一点要提醒：**MMYOLO**，OpenMMLab 的 YOLO 复现中枢，采用的是 GPL-3.0 而非 Apache-2.0，并且自 2023 年 8 月起就已冻结。

## 6. RT-DETR 家族

许可：全线 Apache-2.0。最适合：以研究级代码实现最先进的实时检测。

实时检测的当前前沿，是一簇彼此相关的 transformer 检测器，而不是某个 YOLO。它们大多源自 RT-DETR，共享许多相同的骨干和编码器代码，且几乎都是 Apache-2.0，所以在它们之间切换很容易。它们大多只做检测，唯一的例外（EdgeCrafter）把同样的蒸馏思路延伸到了分割和姿态。它们共同的代价是：这些是研究仓库，配置文件式训练、和 Ultralytics 迥异的使用体验，且没有支持渠道。但模型很强，而且 ONNX 和 TensorRT 导出通常是一流的。

- **[D-FINE](https://github.com/Peterande/D-FINE)**（中科大，ICLR 2025 Spotlight）。把框回归重新表述为带自蒸馏的分布细化。在同等延迟下精度很强（D-FINE-X 在 T4 上约 13 ms 达到约 55.8 AP），尺寸从 N 到 X，并已集成进 Hugging Face Transformers，这让它成为这里最容易在其自身仓库之外加载和微调的一个。
- **[DEIM 和 DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)**（Intellindust AI Lab）。DEIM（CVPR 2025）是一套训练配方（Dense O2O 匹配），可以嫁接到 D-FINE 或 RT-DETRv2 上，并将训练时间大致减半。DEIMv2 加入了 DINOv3 特征，并向下缩放到面向移动端的、参数量不到 1M 的 Atto 档；DEIMv2-S 是首个突破 50 AP 的 10M 以下模型。它是这一组里维护最积极的，一直更新到 2026 年。
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)**（Intellindust AI Lab，2026）。同一实验室的最新成果，也是这里唯一走出检测的一个：它涵盖检测（ECDet）、实例分割（ECSeg）和姿态（ECPose），每种都有 S/M/L/X 尺寸，全部 Apache-2.0。它把一个经 DINOv3 预训练的大型 ViT 改造成任务专用的教师模型，再蒸馏进为边缘打造的紧凑学生骨干。就其体量而言，数字很强：ECDet-S 仅用不到 10M 参数、只在 COCO 上就达到 51.7 AP，ECPose-X 达到 74.8 AP（领先于 YOLO26Pose-X 的 71.6），其实例分割也能与 RF-DETR 相媲美。它非常新，所以请把它当作研究版本对待，但它是少见的、能横跨这三项任务的紧凑模型。
- **[RT-DETR 和 RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)**（百度）。最初那篇「DETR 在实时检测上击败 YOLO」的工作（CVPR 2024），也是整个家族的根：无 NMS、端到端、易于导出。v2 是 2024 年的一次「bag-of-freebies」升级，在相同延迟下可直接替换。Paddle 原版和权威的 PyTorch 移植共用这个仓库，且已进入 HF Transformers。
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)**（百度，WACV 2025 Oral）和 **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)**（北京大学与清华大学，2025 年底）。v3 仅支持 PaddlePaddle。v4 是这一系列当前最好的（X 接近 57 AP），基于 PyTorch，其做法是把一个视觉基础模型蒸馏进一个轻量检测器，且不增加推理开销。它的代码库还能通过改配置复现 D-FINE、DEIM 和 RT-DETRv2。
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)**（百度）。一个纯 ViT 的 DETR，被定位为 YOLO 的 transformer 替代品，尺寸从 tiny 到 xlarge，支持 ONNX 和 TensorRT 导出。它还是开放词汇的 OVLW-DETR 的底座。近期活跃度较低；把它当作一个稳定的研究版本即可。

LibreYOLO 已经把其中许多（D-FINE、DEIM、DEIMv2、RT-DETRv2、RT-DETRv4、RTMDet、EdgeCrafter）封装在一套 API 之后，并以 YOLO9 和 RF-DETR 作为测试最充分的模型。若需要参考实现和最新的研究检查点，上面那些源仓库才是权威出处。

## 常见问题

**Ultralytics YOLO 可以免费商用吗？**
仅在 AGPL-3.0 下可以，而 AGPL-3.0 要求你公开基于它构建的整个应用的源代码，包括通过网络提供服务的情形。闭源商用需要购买 Ultralytics 的商业许可。采用宽松许可（MIT、Apache-2.0）的替代方案则没有这个问题。

**哪个是许可最干净的 Ultralytics 替代方案？**
如果你需要一套可以随处部署的宽松许可栈：LibreYOLO（代码为 MIT）、RF-DETR（Apache-2.0）和 YOLOX（Apache-2.0）。它们都避开了 AGPL 的传染性。

**2026 年什么在取代 YOLO？**
越来越多是实时 transformer 检测器：RT-DETR、RF-DETR，以及 D-FINE 和 DEIM 系列。在算力足够的硬件上，它们在相近延迟下达到或超过 YOLO 的精度。而在小型边缘设备上，YOLO 这类 CNN 仍占优势，因为那些 transformer 在上面运行很吃力，也缺少成熟的 NCNN 或 CPU 路径。

**这些能在树莓派或 NPU 上运行吗？**
取决于导出能力。像 YOLOX 和 RTMDet 这类 CNN 检测器可以导出为 NCNN 并在树莓派上运行，其中一些（YOLOX）还能上 Hailo NPU。而像 RF-DETR 这样的 transformer 检测器不行，它们需要 GPU 或性能强劲的 CPU。

## 上手试试

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO 采用 MIT 许可，可在 Linux、Mac 和 Windows 上运行，并且无需改动代码即可在 GPU、Apple Silicon 和普通 CPU 上工作。一套 API 覆盖 RF-DETR、D-FINE、DEIM、YOLOX、YOLO-NAS、RTMDet、RT-DETR、EdgeCrafter 等更多模型，横跨检测、分割、姿态、分类、深度、视线和跟踪。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://libreyolo.com/docs)
