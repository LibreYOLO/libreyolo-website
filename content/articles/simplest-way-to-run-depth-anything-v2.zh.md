---
title: 如何用 LibreYOLO 运行 Depth Anything v2
description: Depth Anything V2 能提供业界领先的单目深度估计。下面教你如何用 LibreYOLO 两行代码把它跑起来。
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
---

![毕尔巴鄂古根海姆博物馆与其深度图的对比](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 能生成目前业界最好的单目深度图之一。

如果你来自 YOLO 的世界，官方仓库并不太容易上手。光是从中得到一张深度图，就要经过好几个手动步骤：

* 手动下载正确的 checkpoint，并把它放到正确的文件夹里，

* 让 encoder 与它的配置相匹配（`encoder="vitl"`、`features=256`、`out_channels=...`），

* 自己编写归一化和上色（colorize）的步骤，

* 自己处理设备分配，让它不只能在 CUDA 上跑，也能在 Mac 或 CPU 上跑，

* 还要学习一套和你其余技术栈完全不像的推理 API。

这些都不难，只是徒增摩擦，而这些你都可以跳过。

LibreYOLO 加载的是同样的 Depth Anything V2 权重，并把深度任务变成一次 `predict()` 调用。只要写出模型名称，它就会在首次使用时自动下载：

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # 首次运行时自动下载
model.predict("image.jpg", save=True)     # 把上色后的深度图写入磁盘
```

就这么简单。如果你用过标准的 YOLO API，这个写法会很熟悉：按名称加载模型，调用 `predict`，让 `save=True` 写出可视化结果。这和你做目标检测时用的是完全相同的调用，只不过结果里带的是深度图而不是检测框。色彩映射、归一化、设备分配都已经替你处理好了。它在搭配 CUDA 的 Linux、搭配 Apple Silicon 的 Mac，或是普通 CPU 上都以同样方式运行，无需改动代码。

在此基础上，同一个调用还能做更多：你可以取出原始深度值直接使用。Depth Anything V2 本身的训练仍然留在原仓库里；LibreYOLO 负责推理、视频和验证。

完全相同的一行调用，应用在差异极大的各种场景上：

![跑酷示例图与其深度图的对比：前景中的跳跃者从身后的混凝土墙中凸显出来。](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![多诺斯蒂亚（Donostia）拉孔查（La Concha）海湾的航拍图与其深度图的对比：船只和海岸线被判为近景，开阔水域则向远处退去。](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![格尔尼卡议事堂（Casa de Juntas de Gernika）带列柱的庭院与其深度图的对比，展现出层层退远的建筑结构。](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![多诺斯蒂亚宪法广场（Plaza de la Constitucion）夜间节庆人群与其深度图的对比，近处一排排人群从身后被灯光照亮的立面中凸显出来。](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

关于权重的说明：LibreYOLO 托管了转换后的 Depth Anything V2 checkpoint，并在首次使用时自动拉取，因此无需手动下载任何东西。上游的许可证依然适用：Small encoder 为 Apache-2.0，而 Base、Large 和 Giant 为 CC-BY-NC-4.0（非商用），因此这几个更强的 checkpoint 仅供非商业用途。想要离线使用，或自己转换权重？一次性的转换脚本依然在那儿：

```bash
# 可选：自己转换官方 checkpoint，而不使用自动下载
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## 上手试试

```bash
pip install libreyolo
```

LibreYOLO 是你能通过 pip 安装到的最完整的计算机视觉库。一套熟悉的 API 覆盖了不断扩充的最先进模型阵容：目标检测（RF-DETR、D-FINE、DEIM）、分割、姿态、旋转框，现在还加上了基于 Depth Anything V2 的单目深度估计，以及对支持这些任务的训练和验证。它可在所有主流操作系统、GPU 或 CPU 上运行，并且采用完全的 MIT 许可，因此你可以毫无附加条件地将它用于商业产品。

在 GitHub 上点个 star：[github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | 文档：[libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
